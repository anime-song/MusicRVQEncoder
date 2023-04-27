import librosa
import numpy as np
from joblib import Parallel, delayed
import os

import fft

DATA_SET_PATH = "./Dataset/Processed"


def create_dataset(files):
    for train_data in files:
        try:
            f = train_data

            music_name = f.split("\\")[-1].split(".mp3")[0]
            print(music_name)

            if os.path.exists(os.path.join(DATA_SET_PATH, music_name) + ".npz"):
                continue

            # 音声読み込み
            y, sr = librosa.load(f, sr=22050, mono=False)
            
            # スペクトル解析
            S = fft.cqt(
                y,
                sr=sr,
                n_bins=12 * 3 * 7,
                bins_per_octave=12 * 3,
                hop_length=256 + 128 + 64,
                Qfactor=25.0)

            S = (S * np.iinfo(np.uint16).max).astype("uint16")
            np.savez(
                os.path.join(DATA_SET_PATH, music_name) + ".npz",
                S=S)

        except Exception as e:
            print(music_name, e)


if __name__ == '__main__':
    files = librosa.util.find_files("./Dataset/Source", ext=["mp3"])

    multi = True
    if multi:
        n_proc = 6
        N = int(np.ceil(len(files) / n_proc))
        y_split = [files[idx:idx + N] for idx in range(0, len(files), N)]

        Parallel(
            n_jobs=n_proc,
            backend="multiprocessing",
            verbose=1)([
                delayed(create_dataset)(
                    files=[f]) for f in files])
