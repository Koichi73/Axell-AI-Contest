# スクリプト本体
import sys
import numpy as np
import cv2
from typing import Tuple
from pathlib import Path
import torch
from torch import Tensor
from torch.utils import data
from torch.optim import Adam
from torch.optim.lr_scheduler import MultiStepLR
from torch.nn import MSELoss
from torchvision import transforms
from torch.cuda.amp import autocast, GradScaler
import onnxruntime as ort
from timm.scheduler import CosineLRScheduler
import yaml
import datetime
from tqdm import tqdm
from utils.models import ESPCN4x
from utils.datasets import TrainDataSet, ValidationDataSet
from utils.early_stopping import EarlyStopping
from utils.train_helper import check_and_make_directory, load_checkpoint, save_checkpoint, create_csv, plot_learning_curve, plot_psnr_curve

# データセットの取得
def get_dataset(dataset_dir) -> Tuple[TrainDataSet, ValidationDataSet]:
    return TrainDataSet(dataset_dir / "train"), ValidationDataSet(dataset_dir / "validation/original", dataset_dir / "validation/0.25x")

# PSNR計算
def calc_psnr(image1: Tensor, image2: Tensor):
    to_image = transforms.ToPILImage()
    image1 = cv2.cvtColor((np.array(to_image(image1))).astype(np.uint8), cv2.COLOR_RGB2BGR)
    image2 = cv2.cvtColor((np.array(to_image(image2))).astype(np.uint8), cv2.COLOR_RGB2BGR)
    return cv2.PSNR(image1, image2)

# 学習
def train(model, device, batch_size, num_workers, epochs, lr, scheduler, dataset_dir, output_dir, pretrained=None):
    scaler = GradScaler()
    train_dataset, validation_dataset = get_dataset(dataset_dir)
    train_data_loader = data.DataLoader(train_dataset,
                                batch_size=batch_size,
                                shuffle=True,
                                num_workers=num_workers,
                                pin_memory=True)
    validation_data_loader = data.DataLoader(validation_dataset,
                                batch_size=1,
                                shuffle=False,
                                num_workers=num_workers)
    if pretrained is not None:
        model.load_state_dict(torch.load(pretrained))
    optimizer = Adam(model.parameters(), lr=lr)
    if scheduler == "cosine":
        scheduler = CosineLRScheduler(optimizer, t_initial=epochs, lr_min=0.000005)
    else:
        scheduler = MultiStepLR(optimizer, milestones=[30, 50, 65, 80, 90], gamma=0.7) 
    criterion = MSELoss()
    train_losses, validation_losses, train_psnres, validation_psnres = [], [], [], []
    early_stopping = EarlyStopping(output_dir, patience=100)
    start_epoch = 0

    if (output_dir / "checkpoint.pth").exists():
        start_epoch, model, optimizer, scheduler, scaler, train_losses, validation_losses, train_psnres, validation_psnres = load_checkpoint(model, optimizer, scheduler, scaler, train_losses, validation_losses, train_psnres, validation_psnres, output_dir / "checkpoint.pth")
        print(f"Loaded the checkpoint. Start epoch: {start_epoch+1}")

    for epoch in range(start_epoch, epochs):
        try:
            model.train()
            train_loss, validation_loss, train_psnr, validation_psnr = 0.0, 0.0, 0.0, 0.0
            for idx, (low_resolution_image, high_resolution_image ) in tqdm(enumerate(train_data_loader), desc=f"EPOCH[{epoch+1}/{epochs}] TRAIN", total=len(train_data_loader), leave=False):
                low_resolution_image = low_resolution_image.to(device)
                high_resolution_image = high_resolution_image.to(device)
                optimizer.zero_grad()
                with autocast(dtype=torch.float16):
                    output = model(low_resolution_image)
                    loss = criterion(output, high_resolution_image)
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
                train_loss += loss.item() * low_resolution_image.size(0)
                for image1, image2 in zip(output, high_resolution_image):   
                    train_psnr += calc_psnr(image1, image2)
            avarage_train_loss = train_loss / len(train_data_loader)
            avarage_train_psnr = train_psnr / len(train_data_loader)
            train_losses.append(avarage_train_loss)
            train_psnres.append(avarage_train_psnr)
            
            # 検証
            model.eval()
            with torch.no_grad():
                for idx, (low_resolution_image, high_resolution_image ) in tqdm(enumerate(validation_data_loader), desc=f"EPOCH[{epoch+1}/{epochs}] VALIDATION", total=len(validation_data_loader), leave=False):
                    low_resolution_image = low_resolution_image.to(device)
                    high_resolution_image = high_resolution_image.to(device)
                    output = model(low_resolution_image)
                    loss = criterion(output, high_resolution_image)
                    validation_loss += loss.item() * low_resolution_image.size(0)
                    for image1, image2 in zip(output, high_resolution_image):   
                        validation_psnr += calc_psnr(image1, image2)
            avarage_validation_loss = validation_loss / len(validation_data_loader)
            avarage_validation_psnr = validation_psnr / len(validation_data_loader)
            validation_losses.append(avarage_validation_loss)
            validation_psnres.append(avarage_validation_psnr)
            
            print(f"EPOCH[{epoch+1}] TRAIN LOSS: {avarage_train_loss:.4f}, VALIDATION LOSS: {avarage_validation_loss:.4f}, TRAIN PSNR: {avarage_train_psnr:.4f}, VALIDATION PSNR: {avarage_validation_psnr:.4f}")
            early_stopping(avarage_validation_loss, model)
            if early_stopping.early_stop:
                break
            scheduler.step()
            create_csv(train_losses, validation_losses, train_psnres, validation_psnres, output_dir)
            plot_learning_curve(train_losses, validation_losses, output_dir)
            plot_psnr_curve(train_psnres, validation_psnres, output_dir)
            save_checkpoint(epoch+1, model, optimizer, scheduler, scaler, train_losses, validation_losses, train_psnres, validation_psnres, output_dir)
            
        except Exception as ex:
            print(f"EPOCH[{epoch}] ERROR: {ex}")

# ONNXモデルへの変換
# ONNXモデルへ変換するためtorch.onnx.exportを呼び出しています。  
# この際、opset=17、モデルの入力名はinput、モデルの出力名はoutput、モデルの入力形状は(1, 3, height, width)となるように dynamic_axes を設定します。  
# (この例では(1, 3, 128, 128)のダミー入力を設定後、shape[2]、shape[3]にdynamic_axesを設定することで、モデルの入力形状を(1, 3, height, width)としています。)
def export_model_to_onnx(model, output_dir):
    model.load_state_dict(torch.load(output_dir / "model.pth"))
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
def inference_onnxruntime(dataset_dir, output_dir):
    input_image_dir = dataset_dir / "validation/0.25x"
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
def calc_and_print_PSNR(dataset_dir, output_dir):
    input_image_dir = dataset_dir / "validation/0.25x"
    output_image_dir = output_dir / "inference"
    original_image_dir = dataset_dir / "validation/original"
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

def main(config_file):
    # Load the configuration file
    with open(config_file, 'r') as f:
        config = yaml.load(f, Loader=yaml.FullLoader)

    dataset_dir = Path(config["dataset_dir"])
    output_dir = Path(config["output_dir"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = ESPCN4x().to(device)
    check_and_make_directory(output_dir)
    train(model, device, config["batch_size"], config["num_workers"], config["epochs"], config["lr"], config["scheduler"], dataset_dir, output_dir)
    export_model_to_onnx(model, output_dir)
    inference_onnxruntime(dataset_dir, output_dir)
    calc_and_print_PSNR(dataset_dir, output_dir)

if __name__ == "__main__":
    args = sys.argv
    # Validate the arguments
    if len(args) != 2:
        raise ValueError("ValueError: The number of arguments is invalid.")
    config_file = args[1]
    if not Path(config_file).exists():
        raise FileNotFoundError(f"File not found: {config_file}")
    if not config_file.endswith('.yaml'):
        raise ValueError("ValueError: The configuration file must be in yaml format.")
    
    # Run the main function
    main(config_file)
