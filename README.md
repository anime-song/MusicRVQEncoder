# MusicRVQEncoder
Residual-VQを使ったオートエンコーダ

# 前提条件
Python 3.9

# データ作成
訓練データを作成するには Dataset/Source フォルダにオーディオデータを配置します。
オーディオデータはすべてmp3である必要があります。

以下を実行すると Dataset/Processed フォルダに訓練データが作成されます。
```
python preprocess/convert_spectrogram.py
```

# 学習済みモデル
オートエンコーダモデルは992.6hの音楽データで100epoch訓練されています。

[学習済みオートエンコーダ重み](https://huggingface.co/anime-song/MusicRVQEncoder/resolve/main/music_rvq_ae.h5)

上記をダウンロード、解凍しmodelフォルダー内に入れてください。

# 訓練
学習を安定させるためにオートエンコーダを初めに訓練します。
```
python train_ae.py
```

その後学習済みオートエンコーダを使用してベクトル量子化レイヤーを含むモデルを訓練します。
```
python train_lm.py
```

パラメータなどは調整できるようにする予定です。


# 予測
オートエンコーダの結果を確認するには以下を実行します。
```
python predict_ae.py
```

量子化レイヤーを含むモデルの結果を確認するには以下を実行します。
```
python predict_lm.py
```
