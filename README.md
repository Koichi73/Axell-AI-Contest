# Axell AI Contest 2024
https://signate.jp/competitions/1374

## ディレクトリ構造

    ├── README.md
    ├── requirements.txt
    ├── datasets/
    │   ├── df2k_no_person/     <- 配布データ+DIV2K+Flickr2K(without person)
    │   ├── div2k_no_person/    <- 配布データ+DIV2K(without person)
    │   ├── raw/                <- 配布データ
    │   └── sample/             <- サンプルデータ
    ├── outputs/                <- モデルの学習結果
    └── src/                    <- メインのソースコード
        ├── configs/            <- パラメータの設定
        ├── data/               <- データの前処理など
        ├── models/             <- モデル
        ├── utils/              <- ユーティリティ関数
        └── train.py            <- 学習の実行

## コンテスト概要
- 一般自然画像の4x超解像度
- 約1000枚の写真が配布される
- 推論時間0.035s/imageの制限
- PSNRで評価

## 解法(最終6位)
### モデル
シンプルかつ拡張性の高いEDSRを採用。推論時間に収まるようパラメータ数を調整。

### データ
配布データに、外部データとしてDIV2KとFlickr2Kを追加。配布データに人物写真がなかったため、yolov5で人がメインの画像を外部データから削除。

### 学習
データ拡張は反転のみ。500epochs。400epochから配布データのみで学習。  
`src/configs/240826_03.yaml`にその他の条件を記載。
