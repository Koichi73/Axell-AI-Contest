import numpy as np
from PIL import Image

class CutBlur(object):
    def __init__(self, prob=0.5, alpha=1.0, cut_min=0.2, cut_max=0.5):
        """
        Args:
            prob (float): Cutblurを適用する確率
            alpha (float): 高解像度と低解像度のブレンド比率
        """
        self.prob = prob
        # self.alpha = alpha
        self.cut_min = cut_min
        self.cut_max = cut_max

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
        cut_ratio = np.random.uniform(self.cut_min, self.cut_max)
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


