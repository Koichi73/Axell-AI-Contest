import numpy as np
from PIL import Image
import random

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
