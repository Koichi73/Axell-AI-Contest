from transformers import Swin2SRForImageSuperResolution
from torch import nn

class Swin2SR(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.model = Swin2SRForImageSuperResolution.from_pretrained("caidas/swin2SR-classical-sr-x4-64")
    
    def forward(self, x):
        return self.model(x)