# 配布データと応募用ファイル作成方法の説明

本コンペティションで配布されるデータと応募用ファイルの作成方法や投稿する際の注意点について説明する.

1. [配布データ](#配布データ)
1. [応募用ファイルの作成方法](#応募用ファイルの作成方法)
1. [投稿時の注意点](#投稿時の注意点)

## 配布データ

配布されるデータは以下の通り.

- [Readme](#readme)
- [学習用データ](#学習用データ)
- [検証用データ](#検証用データ)
- [動作確認用のプログラム](#動作確認用のプログラム)
- [応募用サンプルファイル](#応募用サンプルファイル)
- [チュートリアルコード](#チュートリアルコード)

### Readme

本ファイル(readme.md)で, 配布用データの説明と応募用ファイルの作成方法を説明したドキュメント. マークダウン形式で, プレビューモードで見ることを推奨する.

### 学習用データ

`train.zip`を解凍して得られる. pngフォーマットの高解像画像データ(RGB, 8ビットの符号なし整数型)である. ディレクトリ構造は以下の通り.

```bash
train
├── 1.png
├── 2.png
└── ...
```

### 検証用データ

`validation.zip`を解凍して得られる. pngフォーマットの高解像画像データと低解像画像データである(RGB, 8ビットの符号なし整数型). ディレクトリ構造は以下の通り.

```bash
valiation
├── 0.25x
│   ├── 1.png
│   └── ...
└── original
    ├── 1.png
    └── ...
```

低解像画像(0.25x)と高解像画像(original)のそれぞれのファイル名は対応していて, 低解像画像は高解像画像を(縦と横をそれぞれ)25%にしたものである.

### 動作確認用のプログラム

動作確認用プログラム一式は`run_test.zip`であり, 解凍すると以下のようなディレクトリ構造のデータが生成される.

```bash
run_test
├── src                    Pythonのプログラムを置くディレクトリ
│   ├── generator.py
│   ├── predictor.py
│   └── runner.py
├── submit                 投稿用のモデルファイルを置くディレクトリ
│   └── model.onnx
├── docker-compose.yml     分析環境を構築するためのdocker-composeファイル
├── Dockerfile             分析環境元のDockerファイル
├── input.json             検証用データの情報
├── requirements.txt       分析環境でインストールされる主なPythonライブラリ一覧
└── run.py                 実装した推論プログラムを実行するプログラム
```

使い方の詳細は[応募用ファイルの作成方法](#応募用ファイルの作成方法)を参照されたい.

### 応募用サンプルファイル

応募用サンプルファイルは`sample_submit.zip`として与えられる. 解凍すると以下のようなディレクトリ構造のデータが生成される.

```bash
sample_submit
└── model.onnx
```

実際に作成する際に参照されたい.

### チュートリアルコード

様々な環境下で学習を実行できるサンプルプログラム一式. モデルを学習する際に参考にされたい. `sample_script.zip`として与えられる. 解凍すると以下のようなディレクトリ構造のデータが生成される.

```bash
sample_script
├── model.onnx           学習済みONNXモデル(サンプルスクリプトにて生成)
├── model.pth            学習済み重み(サンプルスクリプトにて生成)
├── train_colab.ipynb    Google Colaboratory向けnotebook版学習・ONNXモデル生成サンプルスクリプト
├── train.ipynb          notebook版学習・ONNXモデル生成サンプルスクリプト
├── TRAIN.md             学習サンプルの説明資料
└── train.py             pythonスクリプト版学習・ONNXモデル生成サンプルスクリプト
```

## 応募用ファイルの作成方法

応募用ファイルは学習済みモデルをzipファイルでまとめたものとする.

### ディレクトリ構造

以下のようなディレクトリ構造となっていることを想定している.

```bash
submit
└── model.onnx
```

- 学習済みモデルの名前は必ず"model.onnx"とすること.
- 学習済みモデルは`onnx-runtime`で読み込めるフォーマットとすること.

### 環境構築

評価システムと同じ環境を用意する. Dockerイメージが[こちら(タグ名はbase_env)](https://hub.docker.com/r/signate/runtime-gpu)で公開されているので, pullしてコンテナを作成して環境構築を行うことを推奨する(GPUが搭載されている環境で構築することが望ましい). Dockerから環境構築する場合, Docker Engineなど, Dockerを使用するために必要なものがない場合はまずはそれらを導入しておく. [Docker Desktop](https://docs.docker.com/get-docker/)を導入すると必要なものがすべてそろうので, 自身の環境に合わせてインストーラをダウンロードして導入しておくことが望ましい. 現状, Linux, Mac, Windowsに対応している. そして, `/path/to/run_test`に同封してある`docker-compose.yml`で定義されたコンテナを, 以下のコマンドを実行することで立ち上げる.

```bash
$ cd /path/to/run_test
$ docker compose up -d
...
```

`docker-compose.yml`は好きに編集するなりして, 自身が使いやすいように改造してもよい. GPUが使えてCUDAを有効化したい場合は以下のように編集することでコンテナ内で使用することができる.

```yaml
version: "3"
services:
  dev1:
    image: signate/runtime-gpu:base_env
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: 1
              capabilities: [gpu]
    container_name: analysis_axell
    ports:
      - "8080:8080"
    volumes:
      - .:/workspace
    tty: true
```

インストールされている主なライブラリは以下の通り.

```bash
tensorflow-gpu==2.11.0
keras==2.11.0
torch==2.2.1
torchvision==0.17.1
onnxruntime-gpu==1.17.1
opencv-python==4.8.0.74
```

CPUで動作するDockerイメージも[こちら(タグ名はonnx_env)](https://hub.docker.com/r/signate/runtime-cpu)に用意してあるので, 使用したい場合は`docker-compose.yml`の`services`->`dev1`->`image`を`signate/runtime-cpu:onnx_env`と編集したうえで構築する. `tensorflow==2.13.0`, `keras==2.13.0`, `onnxruntime==0.17.1`以外のライブラリは同様である.

無事にコンテナが走ったら, 必要に応じてデータをコンテナへコピーする.

```bash
$ docker cp /path/to/some/file/or/dir {コンテナ名}: {コンテナ側パス}
... 
```

そして, 以下のコマンドでコンテナの中に入り, 分析や開発を行う.

```bash
$ docker exec -it {コンテナ名} bash
...
# コンテナに入った後
$ cd /workspace
```

`コンテナ名`には`docker-compose.yml`の`services`->`dev1`->`container_name`に記載の値を記述する. デフォルトでは`/path/to/run_test`をコンテナ側の`/workspace`へバインドマウントした状態(`/path/to/run_test`でファイルの編集などをしたらコンテナ側にも反映される. 逆もしかり.)となっている.

CUDA環境を構築した場合, 実際にCUDAがコンテナ内で有効になっているかどうかは以下のコマンドで確認できる.

```bash
# コンテナに入った後
$ python -c "import torch; print(torch.cuda.is_available())"
True
$ python -c "import tensorflow as tf; print(tf.config.list_physical_devices('GPU'))"
...{GPUデバイスのリスト}...
```

#### 補足

Dockerで環境構築が難しい場合は[Google Colab](https://colab.research.google.com)を活用してもよい. `requirements.txt`に記載のライブラリをインストールすれば概ねサーバーの環境は再現できる. `onnxruntime-gpu`については以下の方法でインストールするとCUDAを有効にすることができる(`onnxruntime-gpu`のバージョンはサーバー側と厳密には一致しない).

```bash
!pip install ort-nightly-gpu --index-url=https://aiinfra.pkgs.visualstudio.com/PublicPackages/_packaging/ort-cuda-12-nightly/pypi/simple/
!pip install onnxruntime-gpu==1.17.0 --index-url=https://pkgs.dev.azure.com/onnxruntime/onnxruntime/_packaging/onnxruntime-cuda-12/pypi/simple/
```

### `model.onnx`の作成方法

`Pytorch`などで学習したモデルを変換することで`onnx`フォーマットの学習済みモデルを作成することが可能. 詳細は例えば[こちら](https://pytorch.org/tutorials/advanced/super_resolution_with_onnxruntime.html)を参照されたい. また, チュートリアルも公開されているので, 適宜参考にされたい.

推論の大まかな流れは以下の通り.

1. ONNX Runtimeにより`onnx`フォーマットで保存された学習済みモデル(ここでは`model.onnx`)を読み込んで, `inference session`を作成する.
1. 高解像化の対象となる低解像画像が`numpy.ndarray`型で渡される. チャンネルはB,G,Rの順で, 8ビットの符号なし整数型である.
1. モデルへ入力するための前処理として, 低解像画像データをRGBへ変換し, CHWの順に転置し, バッチ数1のミニバッチとして拡大して255で除算する.
1. モデルへ入力して4倍に高解像化された画像データを得る. 構造はNCHW(チャンネルはR,G,Bの順)で, 0~1のfloat32型である.
1. 精度評価を行うための後処理として, HWCの順に転置し, 値を255倍して8ビットの符号なし整数型に変換してチャンネルをB,G,Rの順にする.

`run_test/src/predictor.py`も参照されたい.

前処理や後処理などモデルへの入力以外の処理は固定となる. 以上の流れから, モデルの要件はNCHW(チャンネルはR,G,Bの順)で各要素の値が0~1のデータを入力として, 同じ構造で各要素の値が同じ範囲の4倍に高解像化された画像データ(HとWがそれぞれ4倍)を出力することとなる. また, 入力画像の大きさ(HとW)は任意となるのでその対応も可能にする必要がある. 実際に学習を実行してモデルを作成する際にこれらのことを参考にされたい.

### 推論テスト

学習済みモデルが作成できたら, 正常に動作するか確認する.

以下, コンテナ内での作業とする.

[動作確認用のプログラム](#動作確認用のプログラム)を用いて検証用のデータに対して推論を実行する.

```bash
$ python run.py  --exec-dir /path/to/submit --input-data-dir /path/to/validation --input-params-path /path/to/input.json --result-dir /path/to/result --result-name result_name
...
```

- `--exec-dir`には学習済みモデル(`model.onnx`)が存在するパス名を指定する. デフォルトでは`./submit`.
- `--input-data-dir`には配布された検証用データ"validation"のパス名を指定する. 解凍後のパス名を指定すればよい. デフォルトでは`./validation`.
- `--input-params-path`検証用データに関するファイル名などの情報ファイルパスを指定する. デフォルトでは`./input.json`.
- `--result-dir`には各画像のPSNRや推論時間の結果の格納先のディレクトリを指定する. デフォルトは`./results`.
- `--result-name`には評価結果ファイルの名前を指定する. デフォルトは`scores.json`.

実行に成功すると, 平均PSNRと推論時間が出力され, `{result_dir}/{result_name}`として結果ファイルが保存される. 中身は以下のような形となる.

```json
{
    "1.png": {
        "psnr": 29.257987307660155,
        "runtime": 0.0
    },
    "10.png": {
        "psnr": 27.93552978155205,
        "runtime": 0.0019888877868652344
    },
    ...
}
```

"psnr"は対応する画像データに対するPSNR値, "runtime"は実行時間(単位は秒. 画像の読み込みや前処理と後処理は含まない)である.

投稿する前にエラーが出ずに実行が成功することを確認すること.

### 応募用ファイルの作成

上記の[ディレクトリ構造](#ディレクトリ構造)となっていることを確認して, zipファイルとして圧縮する.

## 投稿時の注意点

投稿する前に自身のローカル環境などで実行テストを行い, 学習済みモデルがonnx形式になっていることを前提としてエラーなく実行できるかを確認すること. 投稿時にエラーが出た場合, 以下のことについて確認してみる.

- 任意の大きさの画像に対して正しく4倍高解像化ができているか.
- 実行時間がかかりすぎていないか. 少数のサンプルで正常に動いても, 時間がかかりすぎるとランタイムエラーとなることがある. 使用メモリなども見直すこと.
