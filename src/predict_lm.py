import tensorflow as tf
from tensorflow.keras import layers as L
import librosa
import numpy as np
import cv2


from config import MusicRVQAEConfig, MusicRVQLMConfig
from model import MusicRVQAE, MusicRVQLM
from preprocess import fft

tf.config.set_visible_devices([], 'GPU')

model_input = L.Input(shape=(None, 12 * 3 * 7, 2))
music_rvq_ae = MusicRVQAE(config=MusicRVQAEConfig(), batch_size=1, seq_len=8192)
music_rvq_ae_out = music_rvq_ae(model_input)
music_rvq_ae_model = tf.keras.Model(inputs=[model_input], outputs=music_rvq_ae_out)
music_rvq_ae.trainable = False

rvqlm_config = MusicRVQLMConfig()
model = MusicRVQLM(music_rvq_ae, config=rvqlm_config, batch_size=1)
model_out, attention_scores = model(model_input, return_scores=True, add_loss=False)
model_out = [model_out]
model_out.extend(attention_scores)

model = tf.keras.Model(inputs=[model_input], outputs=model_out)
model.load_weights("./model/music_rvq_lm.h5")

file = input("楽曲：")
y, sr = librosa.load(file, sr=22050, mono=False)

S = fft.cqt(y, sr=sr)
S = S.transpose(1, 2, 0)
pad_num = int(1024 - S.shape[0] % 1024)
S = np.pad(S, [(0, pad_num), (0, 0), (0, 0)])
S = np.array([S[:8192]])

pred = model.predict(S)

for i in range(rvqlm_config.num_layers):
    for j in range(rvqlm_config.num_heads):
        layer = pred[1 + i][0][j, :, :].transpose(1, 0)
        cv2.imwrite("./img/attentions/score_{}_{}.jpg".format(i, j), cv2.flip(fft.minmax(layer) * 255, 0))
cv2.imwrite("./img/latent.jpg", cv2.flip(fft.minmax(pred[-1].transpose(1, 0)) * 255, 0))
