"""
DIV2K_train_HRのイメージセットから、人がメインで写っている画像をoutputフォルダに移動するスクリプト
"""
import cv2
import os
import shutil
import torch

# YOLOv5を使用した人検出モデルをロード
model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)

def calculate_person_area(image, results):
    labels = results.xyxyn[0][:, -1].cpu().numpy()
    boxes = results.xyxyn[0][:, :-1].cpu().numpy()
    person_area = 0
    for label, box in zip(labels, boxes):
        if int(label) == 0:  # "person"クラス
            x1, y1, x2, y2 = box[:4]
            person_area += (x2 - x1) * (y2 - y1)
    return person_area

def is_person_main(image_path, threshold=0.1):
    img = cv2.imread(image_path)
    results = model(img) # 人検出
    person_area = calculate_person_area(img, results)
    return person_area > threshold

def main():
    input_folder = 'datasets/external/DIV2K_train_HR'
    output_folder = 'datasets/div2k_no_person/train' # 人でない画像を移動する先のフォルダ
    threshold = 0.1 # 人がメインで写っているかどうかの閾値

    os.makedirs(output_folder, exist_ok=True)
    image_files = [f for f in os.listdir(input_folder) if os.path.isfile(os.path.join(input_folder, f))]
    for image_file in image_files:
        image_path = os.path.join(input_folder, image_file)
        if not is_person_main(image_path, threshold):
            shutil.copy(image_path, os.path.join(output_folder, image_file))
            print(f"{image_file} を {output_folder} にコピーしました。")
if __name__ == '__main__':
    main()