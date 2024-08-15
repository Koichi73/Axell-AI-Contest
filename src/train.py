# 学習パラメーター
batch_size = 50
num_workers = 0
num_epoch = 1
learning_rate = 1e-3

# スクリプト本体
import sys
import torch
from torch import Tensor
from torch.utils import data
from torch.optim import Adam
from torch.optim.lr_scheduler import MultiStepLR
from torch.nn import MSELoss
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms
from tqdm import tqdm, trange
import onnxruntime as ort
import numpy as np
import datetime
from typing import Tuple
from pathlib import Path
import cv2
from tqdm import tqdm
from utils.models import ESPCN4x
from utils.datasets import TrainDataSet, ValidationDataSet

output_dir = Path("outputs/sample")
output_dir.mkdir(exist_ok=True, parents=True)

def get_dataset() -> Tuple[TrainDataSet, ValidationDataSet]:
    return TrainDataSet(Path("./dataset/train"), 850 * 10), ValidationDataSet(Path("./dataset/validation/original"), Path("./dataset/validation/0.25x"))

# 学習
# 定義したモデルをpytorchで学習します。  
# バッチサイズなどのパラメーターはお使いのGPUのVRAMに合わせて調整をしてください。  
# 学習時のログはlogフォルダーに保存されます。  
# 学習後、ONNXモデルへ変換するためtorch.onnx.exportを呼び出しています。  
# この際、opset=17、モデルの入力名はinput、モデルの出力名はoutput、モデルの入力形状は(1, 3, height, width)となるように dynamic_axes を設定します。  
# (この例では(1, 3, 128, 128)のダミー入力を設定後、shape[2]、shape[3]にdynamic_axesを設定することで、モデルの入力形状を(1, 3, height, width)としています。)
def train():
    to_image = transforms.ToPILImage()
    def calc_psnr(image1: Tensor, image2: Tensor):
        image1 = cv2.cvtColor((np.array(to_image(image1))).astype(np.uint8), cv2.COLOR_RGB2BGR)
        image2 = cv2.cvtColor((np.array(to_image(image2))).astype(np.uint8), cv2.COLOR_RGB2BGR)

        return cv2.PSNR(image1, image2)
        

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = ESPCN4x()
    model.to(device)
    writer = SummaryWriter("log")

    train_dataset, validation_dataset = get_dataset()
    train_data_loader = data.DataLoader(train_dataset,
                                batch_size=batch_size,
                                shuffle=True,
                                num_workers=num_workers,
                                pin_memory=True)
    validation_data_loader = data.DataLoader(validation_dataset,
                                batch_size=1,
                                shuffle=False,
                                num_workers=num_workers)

    optimizer = Adam(model.parameters(), lr=learning_rate)
    scheduler = MultiStepLR(optimizer, milestones=[30, 50, 65, 80, 90], gamma=0.7) 
    criterion = MSELoss()

    for epoch in trange(num_epoch, desc="EPOCH"):
        try:
            # 学習
            model.train()
            train_loss = 0.0 
            validation_loss = 0.0 
            train_psnr = 0.0
            validation_psnr = 0.0
            for idx, (low_resolution_image, high_resolution_image ) in tqdm(enumerate(train_data_loader), desc=f"EPOCH[{epoch}] TRAIN", total=len(train_data_loader)):
                low_resolution_image = low_resolution_image.to(device)
                high_resolution_image = high_resolution_image.to(device)
                optimizer.zero_grad()
                output = model(low_resolution_image)
                loss = criterion(output, high_resolution_image)
                loss.backward()
                train_loss += loss.item() * low_resolution_image.size(0)
                for image1, image2 in zip(output, high_resolution_image):   
                    train_psnr += calc_psnr(image1, image2)
                optimizer.step()
            scheduler.step()
            
            # 検証
            model.eval()
            with torch.no_grad():
                for idx, (low_resolution_image, high_resolution_image ) in tqdm(enumerate(validation_data_loader), desc=f"EPOCH[{epoch}] VALIDATION", total=len(validation_data_loader)):
                    low_resolution_image = low_resolution_image.to(device)
                    high_resolution_image = high_resolution_image.to(device)
                    output = model(low_resolution_image)
                    loss = criterion(output, high_resolution_image)
                    validation_loss += loss.item() * low_resolution_image.size(0)
                    for image1, image2 in zip(output, high_resolution_image):   
                        validation_psnr += calc_psnr(image1, image2)
            writer.add_scalar("train/loss", train_loss / len(train_dataset), epoch)
            writer.add_scalar("train/psnr", train_psnr / len(train_dataset), epoch)
            writer.add_scalar("validation/loss", validation_loss / len(validation_dataset), epoch)
            writer.add_scalar("validation/psnr", validation_psnr / len(validation_dataset), epoch)
            writer.add_image("output", output[0], epoch)
        except Exception as ex:
            print(f"EPOCH[{epoch}] ERROR: {ex}")

    writer.close()

    # モデル生成
    torch.save(model.state_dict(), output_dir / "model.pth")

    model.to(torch.device("cpu"))
    dummy_input = torch.randn(1, 3, 128, 128, device="cpu")
    torch.onnx.export(model, dummy_input, output_dir / "model.onnx",
                    opset_version=17,
                    input_names=["input"],
                    output_names=["output"],
                    dynamic_axes={"input": {2: "height", 3:"width"}})


# ONNXモデルによる推論(SIGNATE上で動作させるものと同等)
# pytorchで学習・変換したモデルをonnxruntimeで推論して確認します。  
# 推論結果の画像はoutputフォルダーに生成されます。  
# また、簡易的ですが、手元環境での処理時間の計測も行います。
def inference_onnxruntime():
    input_image_dir = Path("./dataset/validation/0.25x")
    output_image_dir = output_dir / "inference"
    output_image_dir.mkdir(exist_ok=True, parents=True)

    sess = ort.InferenceSession(output_dir / "model.onnx", providers=['CUDAExecutionProvider', 'CPUExecutionProvider'])
    input_images = []
    output_images = []
    output_paths = []

    print("load image")
    for image_path in input_image_dir.iterdir():
        output_iamge_path = output_image_dir / image_path.relative_to(input_image_dir)
        input_image = cv2.imread(str(image_path))
        input_image = np.array([cv2.cvtColor(input_image, cv2.COLOR_BGR2RGB).transpose((2,0,1))], dtype=np.float32)/255
        input_images.append(input_image)
        output_paths.append(output_iamge_path)

    print("inference")
    start_time = datetime.datetime.now()
    for input_image in input_images:
        output_images.append(sess.run(["output"], {"input": input_image})[0])
    end_time = datetime.datetime.now()

    print("save image")
    for output_path, output_image in zip(output_paths, output_images):
        output_image = cv2.cvtColor((output_image.transpose((0,2,3,1))[0]*255).astype(np.uint8), cv2.COLOR_RGB2BGR)
        cv2.imwrite(str(output_path), output_image)

    print(f"inference time: {(end_time - start_time).total_seconds() / len(input_images)}[s/image]")

# PSNR計算(従来手法との比較付き)
# onnxruntimeで推論した結果の画像に対してPSNRの計測を行います。  
# また、このスクリプトでは従来手法との比較も行います。 
def calc_and_print_PSNR():
    input_image_dir = Path("./dataset/validation/0.25x")
    output_image_dir = output_dir / "inference"
    original_image_dir = Path("./dataset/validation/original")
    output_label = ["ESPCN", "NEAREST", "BILINEAR", "BICUBIC"]
    output_psnr = [0.0, 0.0, 0.0, 0.0]
    original_image_paths = list(original_image_dir.iterdir())
    for image_path in tqdm(original_image_paths):
        input_image_path = input_image_dir / image_path.relative_to(original_image_dir)
        output_iamge_path = output_image_dir / image_path.relative_to(original_image_dir)
        input_image = cv2.imread(str(input_image_path))
        original_image = cv2.imread(str(image_path))
        espcn_image = cv2.imread(str(output_iamge_path))
        output_psnr[0] += cv2.PSNR(original_image, espcn_image)
        h, w = original_image.shape[:2]
        output_psnr[1] += cv2.PSNR(original_image, cv2.resize(input_image, (w, h), interpolation=cv2.INTER_NEAREST))
        output_psnr[2] += cv2.PSNR(original_image, cv2.resize(input_image, (w, h), interpolation=cv2.INTER_LINEAR))
        output_psnr[3] += cv2.PSNR(original_image, cv2.resize(input_image, (w, h), interpolation=cv2.INTER_CUBIC))
    for label, psnr in zip(output_label, output_psnr):
        print(f"{label}: {psnr / len(original_image_paths)}")

if __name__ == "__main__":
    train()
    inference_onnxruntime()
    calc_and_print_PSNR()
