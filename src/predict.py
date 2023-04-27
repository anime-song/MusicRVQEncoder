import tensorflow as tf
from tensorflow.keras import layers as L
import librosa
import numpy as np
import cv2


from config import MusicRVQEncoderConfig
from model import MusicRVQEncoderModel
from preprocess import fft

tf.config.set_visible_devices([], 'GPU')

model_input = L.Input(shape=(None, 12 * 3 * 7, 2))
model = MusicRVQEncoderModel(config=MusicRVQEncoderConfig(), batch_size=1, seq_len=8192)
model_out, quantized = model(model_input)
outputs = [model_out]
#outputs.extend(attention_scores)
outputs.extend(quantized['quantized_out'])
outputs.extend(quantized['quantized_list'])
model = tf.keras.Model(inputs=[model_input], outputs=outputs)

model.load_weights("./model/music_rp.h5")

file = input("楽曲：")
y, sr = librosa.load(file, sr=22050, mono=False)

S = fft.cqt(y, sr=sr)
S = S.transpose(1, 2, 0)
pad_num = int(1024 - S.shape[0] % 1024)
S = np.pad(S, [(0, pad_num), (0, 0), (0, 0)])
S = np.array([S[:8192]])

pred = model.predict(S)

# for i in range(4):
#     for j in range(8):
#         layer = pred[i + 1][0][j]
#         cv2.imwrite("./img/attentions/score_{}_{}.jpg".format(i, j), cv2.flip(fft.minmax(layer.transpose(1, 0)) * 255, 0))

cv2.imwrite("./img/original_l.jpg", cv2.flip(fft.minmax(S[0][:, :, 0].transpose(1, 0)) * 255, 0))
cv2.imwrite("./img/original_r.jpg", cv2.flip(fft.minmax(S[0][:, :, 1].transpose(1, 0)) * 255, 0))
cv2.imwrite("./img/reconstract_l.jpg", cv2.flip(fft.minmax(pred[0][0][:, :, 0].transpose(1, 0)) * 255, 0))
cv2.imwrite("./img/reconstract_r.jpg", cv2.flip(fft.minmax(pred[0][0][:, :, 1].transpose(1, 0)) * 255, 0))

cv2.imwrite("./img/quantized.jpg", cv2.flip(fft.minmax(pred[1].transpose(1, 0)) * 255, 0))

for i in range(8):
    cv2.imwrite("./img/quantized_imed_{}.jpg".format(i), cv2.flip(fft.minmax(pred[2 + i][0].transpose(1, 0)) * 255, 0))
