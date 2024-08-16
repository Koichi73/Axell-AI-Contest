# データセット定義
# 提供されている学習用画像と評価用画像セット(高解像度＋低解像度)を読み出すクラスです。  
# 学習用画像は元画像を512px四方に切り出し正解画像とします。また、正解画像を1/4に縮小したものを入力画像として用います(TrainDataSet)。  
# 評価用画像は高解像度と低解像度がセットで提供されているため、低解像度のものを入力画像、高解像度のものを正解画像として用います(ValidationDataSet)。

from torch import Tensor
from torch.utils import data
from abc import ABC, abstractmethod
from pathlib import Path
import PIL
from PIL import Image
from torchvision import transforms
from typing import Tuple

class DataSetBase(data.Dataset, ABC):
    def __init__(self, image_path: Path):
        self.images = list(image_path.iterdir())
        
    def __len__(self) -> int:
        return len(self.images)
    
    @abstractmethod
    def get_low_resolution_image(self, image: Image, path: Path)-> Image:
        pass
    
    def preprocess_high_resolution_image(self, image: Image) -> Image:
        return image
    
    def __getitem__(self, index) -> Tuple[Tensor, Tensor]:
        image_path = self.images[index % len(self.images)]
        high_resolution_image = self.preprocess_high_resolution_image(PIL.Image.open(image_path))
        low_resolution_image = self.get_low_resolution_image(high_resolution_image, image_path)
        return transforms.ToTensor()(low_resolution_image), transforms.ToTensor()(high_resolution_image)

class TrainDataSet(DataSetBase):
    def __init__(self, image_path: Path):
        super().__init__(image_path)

    def get_low_resolution_image(self, image: Image, path: Path)-> Image:
        return transforms.Resize((image.size[0] // 4, image.size[1] // 4), transforms.InterpolationMode.BICUBIC)(image.copy())
    
    def preprocess_high_resolution_image(self, image: Image) -> Image:
        return transforms.Compose([
            transforms.RandomCrop(size = 512),
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip()
        ])(image)

class ValidationDataSet(DataSetBase):
    def __init__(self, high_resolution_image_path: Path, low_resolution_image_path: Path):
        super().__init__(high_resolution_image_path)
        self.high_resolution_image_path = high_resolution_image_path
        self.low_resolution_image_path = low_resolution_image_path

    def get_low_resolution_image(self, image: Image, path: Path)-> Image:
        return PIL.Image.open(self.low_resolution_image_path / path.relative_to(self.high_resolution_image_path))
