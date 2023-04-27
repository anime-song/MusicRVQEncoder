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
music_rvq_ae = MusicRVQAE(config=MusicRVQAEConfig(), batch_size=1, seq_len=8182)
music_rvq_ae_out = music_rvq_ae(model_input)
music_rvq_ae_model = tf.keras.Model(inputs=[model_input], outputs=music_rvq_ae_out)
music_rvq_ae_model.load_weights("./model/music_rvq_ae.h5")

model = MusicRVQLM(music_rvq_ae, config=MusicRVQLMConfig(), batch_size=1)
model_out = model(model_input)
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

cv2.imwrite("./img/original_l.jpg", cv2.flip(fft.minmax(S[0][:, :, 0].transpose(1, 0)) * 255, 0))
cv2.imwrite("./img/original_r.jpg", cv2.flip(fft.minmax(S[0][:, :, 1].transpose(1, 0)) * 255, 0))
cv2.imwrite("./img/reconstract_l.jpg", cv2.flip(fft.minmax(pred[0][:, :, 0].transpose(1, 0)) * 255, 0))
cv2.imwrite("./img/reconstract_r.jpg", cv2.flip(fft.minmax(pred[0][:, :, 1].transpose(1, 0)) * 255, 0))
