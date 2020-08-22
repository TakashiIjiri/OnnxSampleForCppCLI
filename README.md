# OnnxSampleForCppCLI
Train a VAE with PyTorch, export the model as onnx, and use the decoder of VAE in C++CLI (VS2019).

ONNXをC++/CLI環境で利用するサンプルです．  
1. PyTorchでMNISTデータの変分オートエンコーダ（VAE）を訓練し（Python + PyTorch）,  
2. 学習したモデルをONNX形式で掃き出し（Python + PyTorch），  
3. C++CLI環境でONNXを読み込みVAEのデコーダを実行して手書き数字を出力します．  

![img](https://github.com/TakashiIjiri/OnnxSampleForCppCLI/blob/master/fig.png)



##利用環境  
- Windows 10 
- GeForce 1660ti 
  
## 準備1 Cudaのインストール  
- GPUのドライバの更新
  - nvidiaのページから最新のものをDLしてインストール
 
- Cudaのインストール
  - 本家ページに行く（https://developer.nvidia.com/cuda-toolkit-archive）
  - CUDA Toolkit 10.1 update2　を選択
  - OS等を選んでダウンロードしてインストール
  
コマンドプロンプトで　$nvcc -V でcuda　のバージョン確認できる  
  
  
## 準備2 Python関連の環境構築
- Python 3.8.5をインストール
  - 本家（https://www.python.org/）よりインストーラをダウンロード
  - トップページのリンクは32bit版なので注意!!! Windows x86-64 executable installer を探してインストールする
  - インストーラ開始時のAdd Python 3.8 to PATHへのチェック忘れない

- PipとSetupToolの更新
  - コマンドプロンプトを管理者権限で実行して
  - $python -m pip install -U pip setuptools
  - $python -m pip install -U setuptools

- 準備5 パッケージのインストール （numpy, scikit-learn, matplotlib, opencv）
  - $python -m pip install -U numpy
  - $python -m pip install -U scikit-learn
  - $pip install matplotlib
  - $pip install opencv-python


## 準備3 PyTorchのインストール
- 本家ページに行く https://pytorch.org/
  - ページの少し下のほうで，OS, cudaのバージョン情報を指定するとコマンドプロンプトからpipで打ち込むべきコマンドが出てくる
  - 出力されたコマンドをコマンドプロンプトで実行
  - $pip install torch==1.6.0+cu101 torchvision==0.7.0+cu101 -f https://download.pytorch.org/whl/torch_stable.html
  - ちなみにPythonが32bitだとここではまる

参考1 : https://www.kkaneko.jp/tools/win/pytorch.html
参考2 : https://qiita.com/hey_gt/items/cabf4dce9d8e39a5b4d4


## 準備4 Visual Studio 2019 コミュニティ インストール  
- 本家のページよりコミュニティ版のインストーラをDLしてインストール
  - C++によるデスクトップ開発にチェック
  - v142 ビルド ツール (14.27) の C++/CLI サポートにもチェック
  
  
  
## 実行方法 (VAEの学習)
- vae_mnist_pytorchフォルダにて 「$python vae_mnist.py」とすると学習が始まる
- モデルは「model_fin.pth」というファイルに書き出される
- 学習後「$python vae_mnist_export_onnx.py」とすると，model_fin.pthがONNX形式で出力される

## ONNXファイルの確認
- Neutron (https://www.electronjs.org/apps/netron)を利用すると，モデルの構造や入出力層の名前を確認できる
  
  
## 実行方法 (C++/CLIによる可視化)
- Visual Studio 2019により，OnnxSampleOnCppCLI.slnを開く
- ./OnnxForCppCLI\OnnxSampleOnCppCLIフォルダに上記のonnxをコピーする（このリポジトリではコピー済み）
- x64, Releaseモードでコンパイルし実行

## C++/CLIでONNX用のプロジェクトを作る際にやったこととはまったところ
- 
- 





