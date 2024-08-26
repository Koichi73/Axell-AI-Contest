# 4倍拡大サンプルモデル(ESPCN)の構造定義
# 参考 https://github.com/Nhat-Thanh/ESPCN-Pytorch
# モデルへの入力はN, C, H, Wの4次元入力で、チャンネルはR, G, Bの順、画素値は0~1に正規化されている想定となります。  
# また、出力も同様のフォーマットで、縦横の解像度(H, W)が4倍となります。

import torch
from torch import nn

class ResidualBlock(nn.Module):
    def __init__(self, num_channels):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(num_channels, num_channels, kernel_size=3, padding=1)
        nn.init.normal_(self.conv1.weight, mean=0, std=0.001)
        nn.init.zeros_(self.conv1.bias)
        self.act = nn.ReLU()
        self.conv2 = nn.Conv2d(num_channels, num_channels, kernel_size=3, padding=1)
        nn.init.normal_(self.conv2.weight, mean=0, std=0.001)
        nn.init.zeros_(self.conv2.bias)

    def forward(self, x):
        residual = x
        out = self.act(self.conv1(x))
        out = self.conv2(out)
        out += residual
        return out

class EDSR(nn.Module):
    def __init__(self, num_residual_blocks=8, scale_factor=4) -> None:
        super().__init__()
        self.scale = scale_factor
        self.conv_1 = nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3, padding=1)
        nn.init.normal_(self.conv_1.weight, mean=0, std=0.001)
        nn.init.zeros_(self.conv_1.bias)

        # Residual blocks
        self.residual_blocks = nn.Sequential(
            *[ResidualBlock(32) for _ in range(num_residual_blocks)]
        )

        # Second convolution layer
        self.conv_2 = nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, padding=1)
        nn.init.normal_(self.conv_2.weight, mean=0, std=0.001)
        nn.init.zeros_(self.conv_2.bias)

        # Pixel Shuffle layer for upscaling
        self.conv_3 = nn.Conv2d(in_channels=32, out_channels=(3 * self.scale * self.scale), kernel_size=3, padding=1)
        nn.init.normal_(self.conv_3.weight, mean=0, std=0.001)
        nn.init.zeros_(self.conv_3.bias)

        self.pixel_shuffle = nn.PixelShuffle(self.scale//2)

    def forward(self, x):
        x = self.conv_1(x)
        residual = x
        x = self.residual_blocks(x)
        x = self.conv_2(x)
        x += residual # Adding skip connection
        x = self.conv_3(x)
        x = self.pixel_shuffle(x)
        x = self.pixel_shuffle(x)
        x_out = torch.clamp(x, 0.0, 1.0)
        return x_out

if __name__ == "__main__":
    model = EDSR()
    total_params = sum(p.numel() for p in model.parameters())
    print(f"モデルの総パラメータ数: {total_params}")