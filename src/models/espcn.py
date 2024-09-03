import torch
from torch import nn
from models.common import create_conv_layer

class ESPCN(nn.Module):
    """Efficient Sub-Pixel Convolutional Neural Network (ESPCN) model.
    
    Args:
        scale_factor (int, optional): Upscaling factor.

    References:
        https://github.com/Nhat-Thanh/ESPCN-Pytorch
    """
    def __init__(self, scale_factor=4):
        super().__init__()
        self.scale = scale_factor
        self.act = nn.ReLU()
        self.conv_1 = create_conv_layer(in_channels=1, out_channels=64, kernel_size=5, padding=2)
        self.conv_2 = create_conv_layer(in_channels=64, out_channels=32)
        self.conv_3 = create_conv_layer(in_channels=32, out_channels=32)
        self.conv_4 = create_conv_layer(in_channels=32, out_channels=(1 * self.scale * self.scale))
        self.pixel_shuffle = nn.PixelShuffle(self.scale)

    def forward(self, x):
        x = x.reshape(-1, 1, x.shape[-2], x.shape[-1])
        x = self.act(self.conv_1(x))
        x = self.act(self.conv_2(x))
        x = self.act(self.conv_3(x))
        x = self.conv_4(x)
        x = self.pixel_shuffle(x)
        x = x.reshape(-1, 3, x.shape[-2], x.shape[-1])
        x = torch.clamp(x, 0.0, 1.0)
        return x

if __name__ == "__main__":
    model = ESPCN()
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total number of parameters in the model: {total_params}")
