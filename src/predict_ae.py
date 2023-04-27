import tensorflow as tf
from tensorflow.keras import layers as L
import librosa
import numpy as np
import cv2


from config import MusicRVQAEConfig
from model import MusicRVQAE
from preprocess import fft

tf.config.set_visible_devices([], 'GPU')

model_input = L.Input(shape=(None, 12 * 3 * 7, 2))
model = MusicRVQAE(config=MusicRVQAEConfig(), batch_size=1, seq_len=8192)
model_out = model(model_input)
outputs = [model_out]
model = tf.keras.Model(inputs=[model_input], outputs=outputs)

model.load_weights("./model/music_rvq_ae.h5")

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
