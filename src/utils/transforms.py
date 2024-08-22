import numpy as np
from PIL import Image
import random

class CutBlur(object):
    def __init__(self, prob=1.0, alpha=0.7):
        """
        Args:
            prob (float): Cutblurを適用する確率
        """
        self.prob = prob
        self.alpha = alpha

    def __call__(self, lr_img, hr_img):
        """
        Args:
            lr_img (Image): 低解像度画像
            hr_img (Image): 高解像度画像
        Returns:
            Image: Cutblurが適用された画像
        """
        if np.random.rand() > self.prob:
            return hr_img

        # Cutblurの領域をランダムに決定
        cut_ratio = np.random.normal(self.alpha, 0.01)
        w, h = hr_img.size
        ch, cw = int(h * cut_ratio), int(w * cut_ratio)
        cy = int(np.random.randint(0, h-ch+1))
        cx = int(np.random.randint(0, w-cw+1))

        # lr_imgのサイズをhr_imgに合わせる
        lr_img = lr_img.resize((w, h), Image.BICUBIC)
        
        # Cutblurを適用
        lr_img_np = np.array(lr_img)
        hr_img_np = np.array(hr_img)
        hr_img_np[cy:cy+ch, cx:cx+cw] = lr_img_np[cy:cy+ch, cx:cx+cw]
        hr_img = Image.fromarray(hr_img_np)
        return hr_img

class CutMix:
    def __init__(self, p: float = 1.0, alpha: float = 0.7):
        self.p = p
        self.alpha = alpha

    def __call__(self, image: Image, ref_image: Image) -> Image:
        if random.random() >= self.p:
            return image

        # 画像のサイズを取得
        w, h = image.size
        v = np.random.normal(self.alpha, 0.01)
        ch, cw = int(h * v), int(w * v)
        # ランダムに領域を選択
        fcy, fcx = random.randint(0, h - ch), random.randint(0, w - cw)
        tcy, tcx = random.randint(0, h - ch), random.randint(0, w - cw)
        # 参照画像から選択した領域を取得
        ref_region = ref_image.crop((fcx, fcy, fcx + cw, fcy + ch))
        # 元の画像の対応する領域に貼り付け
        image.paste(ref_region, (tcx, tcy))
        return image
