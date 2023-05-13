import tensorflow as tf
from tensorflow.keras import layers as L
from tensorflow.keras.callbacks import LearningRateScheduler

from model import MusicRVQAE
from util import DataGeneratorBatch, load_from_npz
from config import MusicRVQAEConfig


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


def step_decay(epochs, lr=0.001):
    def wrap(epoch):
        learning_rate = lr
        if epoch >= 50:
            learning_rate *= 0.1
        if epoch >= 75:
            learning_rate *= 0.1

        return learning_rate
    return wrap


@tf.function
def maskbacc(y_true, y_pred):
    y_true_boolean_mask = tf.not_equal(y_true[:, :, :, 0], -1)
    y_true = tf.boolean_mask(y_true, y_true_boolean_mask)
    y_pred = tf.boolean_mask(y_pred, y_true_boolean_mask)

    return tf.keras.metrics.categorical_accuracy(y_true, y_pred)


if __name__ == "__main__":
    # GPUメモリ制限
    allocate_gpu_memory()

    model_name = "music_rvq_ae"

    epochs = 100
    batch_size = 8
    patch_len = 2048
    cache_size = 100
    initial_epoch = 0
    initial_value_loss = None
    log_dir = "./logs/music_rvq_ae"

    x_train, x_test = load_from_npz()
    
    loss = "mse"
    accuracy = [maskbacc]
    monitor = 'val_loss'

    model_input = L.Input(shape=(None, 12 * 3 * 7, 2))
    model = MusicRVQAE(config=MusicRVQAEConfig(), batch_size=batch_size, seq_len=patch_len)
    model_out = model(model_input)
    model = tf.keras.Model(inputs=[model_input], outputs=model_out)

    optimizer = tf.keras.optimizers.Adam()
    lrs = LearningRateScheduler(step_decay(epochs))

    model.compile(
        optimizer,
        loss,
        metrics=accuracy)
    model.summary()

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

    callbacks = [
        ckpt_callback_best,
        lrs,
        tensorboard_callback,
    ]

    history = model.fit(
        x=train_gen,
        validation_data=test_gen,
        initial_epoch=initial_epoch,
        epochs=epochs,
        shuffle=False,
        callbacks=callbacks
    )
