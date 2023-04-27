import time
import random
import librosa
import threading
import copy
import os
from tensorflow.keras.utils import Sequence
import numpy as np
from sklearn.model_selection import train_test_split


def load_from_npz(directory="./Dataset/Processed"):
    files = librosa.util.find_files(directory, ext=["npz"])
    train_files, test_files = train_test_split(files, test_size=0.2, random_state=42)

    return train_files, test_files


def load_data(self):
    while not self.is_epoch_end:
        should_added_queue = len(self.data_cache_queue) < self.max_queue
        while should_added_queue:
            self._load_cache(self.file_index)
            should_added_queue = len(self.data_cache_queue) < self.max_queue

        time.sleep(0.1)


class DataLoader:
    def __init__(
            self,
            files,
            validate_mode,
            seq_len,
            max_queue=1,
            cache_size=100):
        self.files_index = files.index
        self.files = sorted(set(copy.deepcopy(files)), key=self.files_index)
        self.file_index = 0
        self.data_cache = {}
        self.data_cache_queue = []

        self.validate_mode = validate_mode
        self.cache_size = cache_size
        self.max_queue = max_queue
        self.is_epoch_end = False
        self.seq_len = seq_len
        self.start()

    def _load_cache(self, start_idx):
        cache = {}
        for i in range(self.cache_size):
            idx = start_idx + i
            if idx >= len(self.files):
                idx = len(self.files) - 2

            data = (np.load(self.files[idx])["S"] / np.iinfo(np.uint16).max).astype("float32")
            cache[self.files[idx]] = [data, 0]

        self.data_cache_queue.append(cache)
        self.file_index += self.cache_size

    def on_epoch_end(self):
        self.is_epoch_end = True
        self.load_segment_next.join()
        
        self.is_epoch_end = False
        if not self.validate_mode:
            self.files = random.sample(self.files, len(self.files))
        
        self.file_index = 0
        self.data_cache.clear()
        self.data_cache_queue.clear()

        self.start()

    def start(self):
        self.load_segment_next = threading.Thread(
            target=load_data, args=(self,))
        self.load_segment_next.start()

    def select_data(self):
        while len(self.data_cache) <= 0:
            if len(self.data_cache_queue) >= 1:
                self.data_cache = self.data_cache_queue.pop(0)
                break

            time.sleep(0.1)
            
        file_name, data = random.choice(list(self.data_cache.items()))
        spectrogram_data = data[0]
        n_frames = spectrogram_data.shape[1]

        start = self.data_cache[file_name][-1]
        
        x = spectrogram_data[:, start: start + self.seq_len]

        self.data_cache[file_name][-1] += self.seq_len
        if self.data_cache[file_name][-1] >= n_frames:
            del self.data_cache[file_name]

        return x

    def __len__(self):
        return len(self.files)


class DataGeneratorBatch(Sequence):
    def __init__(self,
                 files,
                 batch_size=32,
                 patch_length=128,
                 initialepoch=0,
                 validate_mode=False,
                 max_queue=1,
                 cache_size=500):

        print("files size:{}".format(len(files)))
        self.dataloader = DataLoader(
            files,
            validate_mode,
            seq_len=patch_length,
            max_queue=max_queue,
            cache_size=cache_size)

        self.batch_size = batch_size
        self.patch_length = patch_length

        total_seq_length = 0
        if validate_mode:
            if os.path.exists("./total_length_validate.txt"):
                with open("./total_length_validate.txt", mode="r") as f:
                    total_seq_length = int(f.read())

        else:
            if os.path.exists("./total_length.txt"):
                with open("./total_length.txt", mode="r") as f:
                    total_seq_length = int(f.read())
        
        if total_seq_length == 0:
            for file in files:
                length = np.load(file)["S"].shape[1]
                total_seq_length += length
            if validate_mode:
                with open("./total_length_validate.txt", mode="w") as f:
                    f.write(str(total_seq_length))
            else:
                with open("./total_length.txt", mode="w") as f:
                    f.write(str(total_seq_length))

        self.batch_len = int((total_seq_length // self.patch_length // self.batch_size)) + 1

        # データ読み込み
        self.epoch = initialepoch

    def on_epoch_end(self):
        self.dataloader.on_epoch_end()
        self.epoch += 1

    def __getitem__(self, index):
        X = np.full(
            (self.batch_size,
             self.patch_length,
             252,
             2),
            -1,
            dtype="float32")

        for batch in range(self.batch_size):
            data = self.dataloader.select_data()
            X[batch, :data.shape[1], :, 0] = data[0]
            X[batch, :data.shape[1], :, 1] = data[1]
        
        return [X], X

    def __len__(self):
        return self.batch_len
