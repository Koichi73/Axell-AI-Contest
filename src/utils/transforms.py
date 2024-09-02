import numpy as np
from PIL import Image
import random

class CutBlur(object):
    def __init__(self, prob=1.0, alpha=0.7):
        """
        Args:
            prob (float): Probability of applying CutBlur.
            alpha (float): The ratio of the size of the cut region.
        """
        self.prob = prob
        self.alpha = alpha

    def __call__(self, lr_img, hr_img):
        """
        Args:
            lr_img (Image): Low-resolution image.
            hr_img (Image): High-resolution image.
        Returns:
            Image: Image with CutBlur applied.
        """
        if np.random.rand() > self.prob:
            return hr_img

        # Determine the CutBlur region randomly
        cut_ratio = np.random.normal(self.alpha, 0.01)
        w, h = hr_img.size
        ch, cw = int(h * cut_ratio), int(w * cut_ratio)
        cy = int(np.random.randint(0, h-ch+1))
        cx = int(np.random.randint(0, w-cw+1))

        # Resize lr_img to match hr_img
        lr_img = lr_img.resize((w, h), Image.BICUBIC)
        
        # Apply CutBlur
        lr_img_np = np.array(lr_img)
        hr_img_np = np.array(hr_img)
        hr_img_np[cy:cy+ch, cx:cx+cw] = lr_img_np[cy:cy+ch, cx:cx+cw]
        hr_img = Image.fromarray(hr_img_np)
        return hr_img

class CutMix:
    def __init__(self, p: float = 1.0, alpha: float = 0.7):
        """
        Args:
            p (float): Probability of applying CutMix.
            alpha (float): The ratio of the size of the mixed region.
        """
        self.p = p
        self.alpha = alpha

    def __call__(self, image: Image, ref_image: Image) -> Image:
        if random.random() >= self.p:
            return image

        # Determine the CutMix region randomly
        w, h = image.size
        v = np.random.normal(self.alpha, 0.01)
        ch, cw = int(h * v), int(w * v)
        # Randomly select the region of the image and the reference image
        fcy, fcx = random.randint(0, h - ch), random.randint(0, w - cw)
        tcy, tcx = random.randint(0, h - ch), random.randint(0, w - cw)
        ref_region = ref_image.crop((fcx, fcy, fcx + cw, fcy + ch))
        # Apply CutMix
        image.paste(ref_region, (tcx, tcy))
        return image
