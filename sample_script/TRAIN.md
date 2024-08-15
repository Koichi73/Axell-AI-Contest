# Axell AI Contest 2024 学習サンプルコード

## コンテストページでの配布物の構成
* readme.md: 配布データ全般の説明
* run_test.zip: 動作確認用プログラム
* train.zip: 学習用データ
* validation.zip: 検証用データ
* sample_submit.zip: 応募用サンプルファイル
* sample_script.zip: 学習サンプルコード

## 学習サンプルコードの構成
* TRAIN.md : 本ドキュメント
* train.py : pythonスクリプト版学習・ONNXモデル生成サンプルスクリプト
* train_colab.ipynb : Google Colaboratory向けnotebook版学習・ONNXモデル生成サンプルスクリプト
* train.ipynb : notebook版学習・ONNXモデル生成サンプルスクリプト
* model.pth : 学習済み重み(サンプルスクリプトにて生成)
* model.onnx : 学習済みONNXモデル(サンプルスクリプトにて生成)

## 学習用データセットについて
学習用データセットに関してはコンテストページ配布のreadme.mdを参照してください。  
このサンプルでは配布されているtrain.zipとvalidation.zipをdatasetフォルダーに展開した前提として説明をしております。  

```bash
dataset
├── valiation
│   ├── 0.25x
│   │   ├── 1.png
│   │   └── ...
│   └── original
|       ├── 1.png
|       └── ...
└── train
    ├── 1.png
    ├── 2.png
    └── ...
```


## 環境構築方法

### Docker Imageを利用する(推奨)

SIGNATE様の評価環境と同じ環境をDockerを用いて構築することが可能です。  
Docker環境の構築方法につきましてはSIGNATE様提供のreadme.mdも参照してください。

#### Windowsをお使いの場合
Windowsをお使いの場合はWSL2上でDockerを動かしてください。

インストールの流れは以下のようになります。

1. コマンドプロンプトなどを開き、以下のコマンドでWSLをインストール

    ```ps
    wsl --install
    ```

2. PCを再起動

3. PC再起動後WSLの初期化を行なうターミナルが表示されるので、WSLに新規に追加するユーザーのユーザー名・パスワードを入力

4. Dockerを導入  
以下のコマンドでDockerを導入します。  
詳細につきましては https://docs.docker.com/engine/install/ https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html を参照してください。  

    ```shell
    sudo apt-get update
    sudo apt-get install -y ca-certificates curl unzip
    sudo install -m 0755 -d /etc/apt/keyrings
    sudo curl -fsSL https://download.docker.com/linux/ubuntu/gpg -o /etc/apt/keyrings/docker.asc
    sudo chmod a+r /etc/apt/keyrings/docker.asc
    echo "deb [arch=$(dpkg --print-architecture) signed-by=/etc/apt/keyrings/docker.asc] https://download.docker.com/linux/ubuntu $(. /etc/os-release && echo "$VERSION_CODENAME") stable" | sudo tee /etc/apt/sources.list.d/docker.list > /dev/null
    sudo apt-get update
    sudo apt-get install -y docker-ce docker-ce-cli containerd.io docker-buildx-plugin docker-compose-plugin
    ```

5. 現在のユーザーをdockerグループへ追加し、WSLを再起動します

    ```shell
    sudo usermod -aG docker $USER
    ```

6. 次のコマンドでnvidia-container-toolkitをインストールします

    ```shell
    curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey | sudo gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg && curl -s -L https://nvidia.github.io/libnvidia-container/stable/deb/nvidia-container-toolkit.list | sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' | sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list
    sudo apt-get update
    sudo apt-get install -y nvidia-container-toolkit
    ```

7. 次のコマンドでnvidia-container-toolkitを有効化します

    ```shell
    sudo nvidia-ctk runtime configure --runtime=docker
    sudo systemctl restart docker
    ```

8. run_test.zipを解凍し、コンテナを構築します

    ```shell
    unzip run_test.zip
    cd run_test/
    docker compose up -d
    ```

9. 次の項目を参照し、依存ライブラリを導入します。

#### pythonスクリプトで学習をする場合
サンプルのPythonスクリプトで学習をする場合、追加で依存ライブラリの導入が必要です。

```shell
docker exec -it analysis_axell python3 -m pip install onnx==1.13.1
```


#### notebookで学習をする場合
notebookで学習をする場合は追加で依存ライブラリの導入が必要です。

```shell
docker exec -it analysis_axell python3 -m pip install notebook ipywidgets onnx==1.13.1
```


### google colaboratoryを利用する

Googleアカウントを作成し、ログインを行ってください。

### Python+venvで環境を構築する

手元のPythonの実行環境を用いて学習用環境を構築することが可能です。  
なお、NVIDIA製GPUを用いて学習を行う場合、事前に最新のGPUドライバー/CUDA/CuDNNの導入が必要です。  

**NOTE**
2024/07現在、torchがサポートしているCUDAは11.8か12.1となります。  
また、onnxruntiemがサポートしているCuDNNは8.9系となりますので、CUDA11.8+CuDNN8.9.7もしくはCUDA12.1+CuDNN8.9.7を導入することをおすすめします。  

CUDAは https://developer.nvidia.com/cuda-toolkit-archive より  
CuDNNは https://developer.nvidia.com/rdp/cudnn-archive より入手可能です。

1. venvを作成  
    `python -m venv <venv名>` でvenvを作成します  
    例: 
    ```
    python -m venv axell-ai-contest
    ```
  
2. venvを有効化します  
    例 (linux):  
    ```shell
    source axell-ai-contest/bin/activate
    ```  
    
    例 (windows):  
    ```ps
    axell-ai-contest/Scripts/Activate.ps1
    ```

3. パッケージを導入します  
    なお、torch及びonnxruntimeは導入したCUDAのバージョンによりインストール方法が異なります。  
    詳細公式はドキュメントを参照してください。  
    pytorch: https://pytorch.org/get-started/locally/
    onnxruntiem: https://onnxruntime.ai/getting-started
    
    ```shell
    # 共通ライブラリの導入
    python -m pip install Pillow tqdm onnx tensorboard opencv-python
    # torchのインストール(利用する環境に応じて内容は変更)
    python -m pip install torch torchvision torchaudio
    # onnxruntimeのインストール(推論用/利用する環境に応じて内容は変更)
    python -m pip install onnxruntime-gpu 
    ```
    
    例(cuda11.8の場合): 
    cuda11.8の場合、まずpytorchのサイト(https://pytorch.org/get-started/locally/)上で、
    * PyTorch Build = Stable
    * Your OS = お使いのOS
    * Package = Pip
    * Language = Python
    * Compute Platform = CUDA 11.8
    を選択します。  
    環境情報を入力するとインストールのためのコマンドが表示されます。  
    次にonnxruntimeのサイト(https://onnxruntime.ai/getting-started)で  
    * Platform = お使いのOS
    * API = Python
    * Architecture = お使いのCPUのアーキテクチャー(通常はX64)
    * Hardware Acceleration = CUDA
    を選択すると、インストール手順が表示されます。 cuda11.8の場合はFor CUDA 11.Xにかかれているものを利用します。
    以上よりサンプルのインストールスクリプトは以下のようになります。
    ```shell
    python -m pip install Pillow tqdm onnx tensorboard opencv-python
    ## torchのインストール(https://pytorch.org/get-started/locally/で表示されたもの)
    pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
    ## onnxruntimeのインストール(https://onnxruntime.ai/getting-startedで表示されたもの)
    pip install onnxruntime-gpu
    ```
    
#### notebookで学習をする場合
notebookで学習をする場合は追加で依存ライブラリの導入が必要です。

```shell
python -m pip install notebook ipywidgets
```


## 学習方法

### 評価環境で学習をする場合(pythonスクリプト版)
1. 環境構築時に展開したrun_testフォルダーに学習用データセットとサンプルコード(py.ipynb)を配置します。  
    なお、この際データセットとサンプルコードは以下のような配置となるようにしてください。  
    ```
    run_test    # 環境構築時に展開したrun_testフォルダー
    ├── train.py # 学習サンプルに含まれる学習用コード
    └──dataset  # 事前準備したデータセット
        ├── valiation
        │   ├── 0.25x
        │   │   ├── 1.png
        │   │   └── ...
        │   └── original
        |       ├── 1.png
        |       └── ...
        └── train
            ├── 1.png
            ├── 2.png
            └── ...
    ```

    **NOTE**
    評価環境では環境構築時に展開したrun_testフォルダーとdockerコンテナ上の/workspaceがバインドされています。  
    そのため、ホスト環境のrun_testフォルダーにスクリプトやデータセットを配置するとコンテナ内で利用が可能です。

2. 次のコマンドでサンプルコード(train.py)を実行します。  
    ```shell
    docker exec -w /workspace -it analysis_axell python3 train.py
    ```
    なお、学習が完了すると、学習済みモデル(model.onnx)がサンプルコード(train.py)と同じ場所へ生成されます。


### 評価環境で学習をする場合(notebook版)

1. 環境構築時に展開したrun_testフォルダーに学習用データセットとサンプルコード(train.ipynb)を配置します。  
    なお、この際データセットとサンプルコードは以下のような配置となるようにしてください。  
    ```
    run_test        # 環境構築時に展開したrun_testフォルダー
    ├── train.ipynb # 学習サンプルに含まれる学習用notebook
    └──dataset      # 事前準備したデータセット
        ├── valiation
        │   ├── 0.25x
        │   │   ├── 1.png
        │   │   └── ...
        │   └── original
        |       ├── 1.png
        |       └── ...
        └── train
            ├── 1.png
            ├── 2.png
            └── ...
    ```
    
    **NOTE**
    評価環境では環境構築時に展開したrun_testフォルダーとdockerコンテナ上の/workspaceがバインドされています。  
    そのため、ホスト環境のrun_testフォルダーにスクリプトやデータセットを配置するとコンテナ内で利用が可能です。

2. 次のコマンドでnotebookを起動します。  

    ```
    docker exec -it analysis_axell python3 -m notebook --port 8080 --ip=* --notebook-dir=/workspace
    ```

    **NOTE**
    Docker環境では8080版ポートがホストとマッピングされています。  それ以外のポートを利用したい場合はdocker-compose.ymlを修正してください。  
    また、ローカルマシン以外で起動する場合は`--ip=*`引数を追加してください。

3. Webブラウザでnotebookを開きます。  
    notebook起動時に表示されるURLへアクセスしてください。  
    例: `http://localhost:8080/tree?token=XXXXXXXXXXXXXX`
    
    
4. サンプルコード(train.ipynb)を開きます

5. 先頭のセルから順番に実行します

6. 最後のセルまで到達すると学習済みモデル(model.onnx)がサンプルコード(train.ipynb)と同じ場所へ生成されますのでこちらをダウンロードします。

### Google Colaboratoryで学習をする場合

1. 学習用データセットを Google Drive へアップロードします  
    Google Driveにデータセット用フォルダー(例: dataset)を作成し、以下のレイアウトでデータセットを配置してください。
    ```
    dataset  # 作成したフォルダー
    ├── valiation
    │   ├── 0.25x
    │   │   ├── 1.png
    │   │   └── ...
    │   └── original
    |       ├── 1.png
    |       └── ...
    └── train
        ├── 1.png
        ├── 2.png
        └── ...
    ```

    フォルダー名をdataset以外にする場合、適宜サンプルnotebookを修正してください。

2. Google Colaboratory( https://colab.research.google.com )へアクセスします  

3. ノートブックを開くからアップロードを選択し、参照をクリックします  
    ファイル選択ダイアログでサンプルnotebook(train_colab.ipynb)を選択します。
    
4. ランタイムからランタイムのタイプを変更を選択し、ハードウェアアクセラレーターから任意のGPUを選択し保存を選択します。

5. 先頭のセルから順番に実行します  
    なお、Google Driveマウント時にアクセス許可の画面が表示されますのでログインして許可をしてください。

6. 最後のセルまで到達すると学習済みモデル(model.onnx)がGoogle Drive直下に生成されますのでGoogle Driveを開きダウンロードします。

### Python+venvで学習をする場合(pythonスクリプト版)
1. venvを有効化します
    例 (linux):  
    ```shell
    source axell-ai-contest/bin/activate
    ```  
    
    例 (windows):  
    ```ps
    axell-ai-contest/Scripts/Activate.ps1
    ```
    python3 train.py
    
2. サンプルコードとデータセットを任意の場所へ配置します。  
    なお、この際データセットとサンプルコードは以下のような配置となるようにしてください。  
    ```
    train.py # 学習サンプルに含まれる学習用コード
    dataset  # 事前準備したデータセット
    ├── valiation
    │   ├── 0.25x
    │   │   ├── 1.png
    │   │   └── ...
    │   └── original
    |       ├── 1.png
    |       └── ...
    └── train
        ├── 1.png
        ├── 2.png
        └── ...
    ```

3. サンプルコード(train.py)を実行します。  
    ```shell
    python3 train.py
    ```
    なお、学習が完了すると、学習済みモデル(model.onnx)がサンプルコード(train.py)と同じ場所へ生成されます。

### Python+venvで学習をする場合(notebook版)

1. venvを有効化します
    例 (linux):  
    ```shell
    source axell-ai-contest/bin/activate
    ```  
    
    例 (windows):  
    ```ps
    axell-ai-contest/Scripts/Activate.ps1
    ```

2. 作業ディレクトリへ移動し、学習用データセットとサンプルコード(train.ipynb)を配置します  
    なお、この際データセットとサンプルコードは以下のような配置となるようにしてください。  
    ```
    train.ipynb # 学習サンプルに含まれる学習用notebook
    dataset     # 事前準備したデータセット
    ├── valiation
    │   ├── 0.25x
    │   │   ├── 1.png
    │   │   └── ...
    │   └── original
    |       ├── 1.png
    |       └── ...
    └── train
        ├── 1.png
        ├── 2.png
        └── ...
    ```

3. 次のコマンドでnotebookを起動します。  

    ```shell
    python3 -m notebook 
    ```

    **NOTE**
    デフォルトのポートが利用されますが、指定したポートを利用したい場合は`--port <ポート番号>`引数を追加してください。  
    また、ローカルマシン以外で起動する場合は`--ip=*`引数を追加してください。

4. Webブラウザでnotebookを開きます。  
    notebook起動時に表示されるURLへアクセスしてください。  
    例: `http://localhost:8888/tree?token=XXXXXXXXXXXXXX`

    
5. サンプルコード(train.ipynb)を開きます

6. 先頭のセルから順番に実行します

7. 最後のセルまで到達すると学習済みモデル(model.onnx)がサンプルコード(train.ipynb)と同じ場所へ生成されますのでこちらをダウンロードします。


## 評価環境での評価方法

1. 環境構築時に展開したrun_testフォルダーに学習用データセットと学習済みモデル(model.onnx)を配置します  
    なお、この際データセットと学習済みモデルは以下のような配置となるようにしてください。  
    ```
    run_test            # 環境構築時に展開したrun_testフォルダー
    ├── train.py        # 学習サンプルに含まれる学習用notebook
    ├── dataset         # 事前準備したデータセット
    │   ├── valiation
    │   │   ├── 0.25x
    │   │   │   ├── 1.png
    │   │   │   └── ...
    │   │   └── original
    │   |       ├── 1.png
    │   |       └── ...
    │   └── train
    │       ├── 1.png
    │       ├── 2.png
    │       └── ...
    └── submit          # 提出用フォルダー
        └── model.onnx  # 学習をしたモデル
    ```
    
2. 評価スクリプトを実行します  
    ```shell
    docker exec -w /workspace -it analysis_axell python3 run.py --input-data-dir dataset/validation
    ```

    実行完了後、標準出力にPSNRと処理時間が表示されます。

