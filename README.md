# はじめに

このリポジトリは、データ分析コンペティションサイト"SIGNATE"の「ウェザーニューズ × SIGNATE: Weather Challenge：雲画像予測」(https://signate.jp/competitions/169)にて入賞候補となった際に、最終審査用提出物として作成したファイルが元となっています。

当コンペティションの解法の概要を以下のページに記述しています。合わせて御覧ください。
https://signate.jp/competitions/169/discussions/3

当コンペティションは、現在でも以下のページから、データのダウンロードが可能です。
https://signate.jp/competitions/169/data

## 実行環境

Windows10にて動作確認を行いました。
Python及び、各主要ライブラリのバージョンは以下のとおりです。

Python 3.6.9

tensorflow-gpu 1.14
numpy          1.17.3
pandas         0.25.3
opencv-python  4.1.1.26

## 実行方法

上記のWebページからダウンロードしたZIPデータを展開し、

- testフォルダ
- trainフォルダ(add_met_data内の３つのファイルを正しい場所に追加したもの)
- inference_terms.csv
- sample_submit.csv

が存在するディレクトリ内で、

1. preprocess.py
2. train.py
3. predict.py

の順番にpythonファイルを実行することで、提出フォーマットに沿った予測ファイルが生成されるようになっています。
(preprocess.pyを実行すると、PCの容量を数百ギガバイト消費します。
また、train.pyの実行には、７つのモデルが含まれており、それらを全て実行するには数日の時間を要します。ご注意ください。)

## 各ファイルについての説明

preprocess.pyは、前処理を行い、データセットを作成する処理を行うファイルです。

train.pyは、モデルの学習を行い、学習済みのモデルをhdf5ファイルとして出力するファイルです。

predict.pyは、学習済みのモデルを使ってテストデータの予測を行い、提出用フォーマットに則ったファイルを出力するファイルです。

## その他

当コンペティションは、参加規約上で、参加者によるソースコードの公開が許可されています。(https://signate.jp/competitions/169#terms)
