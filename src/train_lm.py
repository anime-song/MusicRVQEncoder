import tensorflow as tf
from tensorflow.keras import layers as L
from tensorflow.keras.callbacks import LearningRateScheduler

from model import MusicRVQAE, MusicRVQLM
from util import DataGeneratorBatch, load_from_npz
from config import MusicRVQAEConfig, MusicRVQLMConfig


def allocate_gpu_memory(gpu_number=0):
    physical_devices = tf.config.experimental.list_physical_devices('GPU')

    if len(physical_devices) > 0:
        try:
            print("Found {} GPU(s)".format(len(physical_devices)))
            tf.config.experimental.set_memory_growth(
                physical_devices[gpu_number], True)
            print("#{} GPU memory is allocated".format(gpu_number))
        except RuntimeError as e:
            print(e)
    else:
        print("Not enough GPU hardware devices available")


import numpy as np
from tensorflow.keras import backend as K


def lr_warmup_cosine_decay(global_step,
                           warmup_steps,
                           hold = 0,
                           total_steps=0,
                           start_lr=0.0,
                           target_lr=1e-3):
    learning_rate = 0.5 * target_lr * (1 + np.cos(np.pi * (global_step - warmup_steps - hold) / float(total_steps - warmup_steps - hold)))
    warmup_lr = target_lr * (global_step / warmup_steps)
    if hold > 0:
        learning_rate = np.where(global_step > warmup_steps + hold,
                                 learning_rate, target_lr)
    learning_rate = np.where(global_step < warmup_steps, warmup_lr, learning_rate)
    return learning_rate


class WarmupCosineDecay(tf.keras.callbacks.Callback):
    def __init__(self, total_steps=0, warmup_steps=0, start_lr=0.0, target_lr=1e-3, hold=0, global_steps=0):

        super(WarmupCosineDecay, self).__init__()
        self.start_lr = start_lr
        self.hold = hold
        self.total_steps = total_steps
        self.global_step = global_steps
        self.target_lr = target_lr
        self.warmup_steps = warmup_steps
        self.lrs = []

    def on_batch_end(self, batch, logs=None):
        self.global_step = self.global_step + 1
        lr = model.optimizer.lr.numpy()
        self.lrs.append(lr)

    def on_batch_begin(self, batch, logs=None):
        lr = lr_warmup_cosine_decay(global_step=self.global_step,
                                    total_steps=self.total_steps,
                                    warmup_steps=self.warmup_steps,
                                    start_lr=self.start_lr,
                                    target_lr=self.target_lr,
                                    hold=self.hold)
        K.set_value(self.model.optimizer.lr, lr)


if __name__ == "__main__":
    # GPUメモリ制限
    allocate_gpu_memory()

    model_name = "music_rvq_lm"

    epochs = 100
    batch_size = 8
    patch_len = 2048
    cache_size = 100
    initial_epoch = 0
    initial_value_loss = None
    log_dir = "./logs/music_rvq_lm"

    x_train, x_test = load_from_npz()
    
    monitor = 'val_loss'

    model_input = L.Input(shape=(None, 12 * 3 * 7, 2))
    music_rvq_ae = MusicRVQAE(config=MusicRVQAEConfig(), batch_size=batch_size, seq_len=patch_len)
    music_rvq_ae_out = music_rvq_ae(model_input)
    music_rvq_ae_model = tf.keras.Model(inputs=[model_input], outputs=music_rvq_ae_out)
    music_rvq_ae_model.load_weights("./model/music_rvq_ae.h5")
    music_rvq_ae.trainable = False

    model = MusicRVQLM(music_rvq_ae, config=MusicRVQLMConfig(), batch_size=batch_size)
    model_out = model(model_input)
    model = tf.keras.Model(inputs=[model_input], outputs=model_out)

    # ジェネレーター作成
    train_gen = DataGeneratorBatch(
        x_train,
        batch_size=batch_size,
        patch_length=patch_len,
        initialepoch=initial_epoch,
        max_queue=1,
        cache_size=cache_size)

    test_gen = DataGeneratorBatch(
        x_test,
        batch_size=batch_size,
        patch_length=patch_len,
        validate_mode=True,
        cache_size=cache_size)

    tensorboard_callback = tf.keras.callbacks.TensorBoard(
        log_dir=log_dir, histogram_freq=1)

    ckpt_callback_best = tf.keras.callbacks.ModelCheckpoint(
        filepath="./model/{}.h5".format(model_name),
        monitor=monitor,
        verbose=1,
        save_best_only=True,
        save_weights_only=True,
        initial_value_threshold=initial_value_loss
    )

    lrs = WarmupCosineDecay(
        len(train_gen) * epochs, warmup_steps=len(train_gen) * epochs * 0.1, target_lr=1e-4, global_steps=len(train_gen) * initial_epoch)

    callbacks = [
        ckpt_callback_best,
        lrs,
        tensorboard_callback,
    ]

    optimizer = tf.keras.optimizers.Adam(learning_rate=1e-4)

    model.compile(optimizer)
    model.summary()

    history = model.fit(
        x=train_gen,
        validation_data=test_gen,
        initial_epoch=initial_epoch,
        epochs=epochs,
        shuffle=False,
        callbacks=callbacks
    )
