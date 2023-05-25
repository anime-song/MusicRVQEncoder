# MusicRVQEncoder
Residual-VQを使った音楽データのエンコーダ

# 前提条件
Python 3.9

# データ作成
訓練データを作成するには Dataset/Source フォルダにオーディオデータを配置します。
オーディオデータはすべてmp3である必要があります。

以下を実行すると Dataset/Processed フォルダに訓練データが作成されます。
```
python preprocess/convert_spectrogram.py
```

# 訓練
学習を行うには以下を実行します。
```
python train_encoder.py
```

学習はかなり不安定で高確率で出力が同じになる崩壊を起こします。
target_lrを下げる、accum_stepsを上げる、データセットサイズを上げるなどで崩壊しないように調節できます。


# 予測
学習結果を確認するには以下を実行します。
```
python predict_encoder.py
```
