import torch
from torch import nn
from models.common import create_conv_layer, ResidualBlock

class EDSR(nn.Module):
    """Enhanced Deep Super-Resolution (EDSR) model.

    Args:
        num_residual_blocks (int, optional): Number of residual blocks.
        scale_factor (int, optional): Upscaling factor.
    """
    def __init__(self, num_residual_blocks=8, scale_factor=4):
        super().__init__()
        self.num_residual_blocks = num_residual_blocks
        self.scale = scale_factor
        self.conv_1 = create_conv_layer(in_channels=3, out_channels=32)
        self.residual_blocks = nn.Sequential(
            *[ResidualBlock(32) for _ in range(self.num_residual_blocks)]
        )
        self.conv_2 = create_conv_layer(in_channels=32, out_channels=32)
        self.conv_3 = create_conv_layer(in_channels=32, out_channels=(3 * self.scale * self.scale))
        self.pixel_shuffle = nn.PixelShuffle(self.scale // 2)

    def forward(self, x):
        x = self.conv_1(x)
        residual = x
        x = self.residual_blocks(x)
        x = self.conv_2(x)
        x += residual
        x = self.conv_3(x)
        x = self.pixel_shuffle(x)
        x = self.pixel_shuffle(x)
        x = torch.clamp(x, 0.0, 1.0)
        return x

if __name__ == "__main__":
    model = EDSR()
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total number of parameters in the model: {total_params}")
