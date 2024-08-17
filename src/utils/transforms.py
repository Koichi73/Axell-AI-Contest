import numpy as np
from PIL import Image

class CutBlur(object):
    def __init__(self, prob=1, alpha=1.0):
        """
        Args:
            prob (float): Cutblurを適用する確率
            alpha (float): 高解像度と低解像度のブレンド比率
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
        cut_ratio = np.random.normal(loc=0.5, scale=0.1)
        cut_ratio = np.clip(cut_ratio, 0, 1)
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

# 使用例
if __name__ == "__main__":
    # 高解像度と低解像度の画像を用意
    lr_img = Image.open("datasets/raw/validation/0.25x/5.png")
    hr_img = Image.open("datasets/raw/validation/original/5.png")

    # CutBlurオブジェクトを作成
    cutblur = CutBlur()

    # # 画像にCutblurを適用
    output_img = cutblur(lr_img, hr_img)
    output_img.save("output.png")

