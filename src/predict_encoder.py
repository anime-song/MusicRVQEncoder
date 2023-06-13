import tensorflow as tf
from tensorflow.keras import layers as L
import librosa
import numpy as np
import cv2


from config import MusicRVQEncoderConfig
from model import MusicRVQEncoder
from preprocess import fft

tf.config.set_visible_devices([], 'GPU')

model_input = L.Input(shape=(None, 12 * 3 * 7, 2))
music_rvq_ae = MusicRVQEncoder(config=MusicRVQEncoderConfig(), batch_size=1, seq_len=8192)
model_out, encoder_out, scores = music_rvq_ae(model_input, return_attention_scores=True, add_loss=True)
model_out = [model_out]
model_out.extend(scores)
model_out.extend(encoder_out)
model = tf.keras.Model(inputs=[model_input], outputs=model_out)
model.load_weights("./model/music_encoder/music_encoder.ckpt")

file = input("楽曲：")
y, sr = librosa.load(file, sr=22050, mono=False)

if len(y.shape) == 1:
    y = np.array([y, y])

S = fft.cqt(y, sr=sr)
S = S.transpose(1, 2, 0)
pad_num = int(1024 - S.shape[0] % 1024)
S = np.pad(S, [(0, pad_num), (0, 0), (0, 0)])
S = np.array([S[:8192]])

pred = model.predict(S)

cv2.imwrite("./img/original_l.jpg", cv2.flip(fft.minmax(S[0][:, :, 0].transpose(1, 0)) * 255, 0))
cv2.imwrite("./img/original_r.jpg", cv2.flip(fft.minmax(S[0][:, :, 1].transpose(1, 0)) * 255, 0))

for i in range(4):
    for j in range(8):
        cv2.imwrite("./img/attentions/attention_{}_{}.jpg".format(i, j), cv2.flip(fft.minmax(pred[i + 1][0][j].transpose(1, 0)) * 255, 0))

cv2.imwrite("./img/latent.jpg", cv2.flip(fft.minmax(pred[-1].transpose(1, 0)) * 255, 0))
cv2.imwrite("./img/reconstract.jpg", cv2.flip(fft.minmax(pred[0][0][:, :, 0].transpose(1, 0)) * 255, 0))
