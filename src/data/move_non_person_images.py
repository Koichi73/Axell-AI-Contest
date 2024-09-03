import cv2
import os
import shutil
import torch

# Load YOLOv5 model
model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)

def calculate_person_area(results):
    """
    Calculate the total area occupied by persons in the image.

    Args:
        image (numpy.ndarray): The input image.
        results (torch.Tensor): The detection results from the YOLOv5 model.

    Returns:
        float: The total area occupied by persons in the image as a proportion of the image size.
    """
    labels = results.xyxyn[0][:, -1].cpu().numpy()
    boxes = results.xyxyn[0][:, :-1].cpu().numpy()
    person_area = 0
    for label, box in zip(labels, boxes):
        if int(label) == 0: # "person"class
            x1, y1, x2, y2 = box[:4]
            person_area += (x2 - x1) * (y2 - y1)
    return person_area

def is_person_main(image_path, threshold=0.1):
    """
    Determine if a person is the main subject in the image based on the threshold.

    Args:
        image_path (str): The file path of the image to be evaluated.
        threshold (float): The threshold for determining if a person is the main subject.

    Returns:
        bool: True if the person occupies more area than the threshold, False otherwise.
    """
    img = cv2.imread(image_path)
    results = model(img) # inference by YOLOv5
    person_area = calculate_person_area(results)
    return person_area > threshold

def move_non_person_images(input_folder, output_folder, threshold=0.1):
    """
    Move images that do not have a person as the main subject to the output folder.

    Args:
        input_folder (str): Path to the input folder containing image datasets such as DIV2K and Flickr2K.
        output_folder (str): Path to the output folder where non-person images will be copied.
        threshold (float): Threshold to judge whether the image has a person or not.
    """
    os.makedirs(output_folder, exist_ok=True)
    image_files = [f for f in os.listdir(input_folder) if os.path.isfile(os.path.join(input_folder, f))]
    for image_file in image_files:
        image_path = os.path.join(input_folder, image_file)
        if not is_person_main(image_path, threshold):
            shutil.copy(image_path, os.path.join(output_folder, image_file))
            print(f"Copy {image_file} to {output_folder}.")

if __name__ == '__main__':
    input_folder = 'datasets/external/DIV2K_train_HR'
    output_folder = 'datasets/div2k_no_person/train'
    threshold = 0.1
    move_non_person_images(input_folder, output_folder, threshold)
