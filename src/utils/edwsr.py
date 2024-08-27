import torch
from torch import nn
from pytorch_wavelets import DWTForward, DWTInverse
from models import EDSR

class EDWSR(EDSR):
    def __init__(self):
        super(EDWSR, self).__init__()
        self.xfm = DWTForward(J=1, wave='db1')
        self.ifm = DWTInverse(mode='zero', wave='db1')

        # 入力チャンネル数を12に変更
        self.conv_1 = nn.Conv2d(in_channels=12, out_channels=32, kernel_size=3, padding=1)
        nn.init.normal_(self.conv_1.weight, mean=0, std=0.001)
        nn.init.zeros_(self.conv_1.bias)

        # 出力チャンネル数を変更
        self.conv_3 = nn.Conv2d(in_channels=32, out_channels=(12 * self.scale * self.scale), kernel_size=3, padding=1)
        nn.init.normal_(self.conv_3.weight, mean=0, std=0.001)
        nn.init.zeros_(self.conv_3.bias)

    def forward(self, x):
        yl, yh = self.xfm(x)

        batch_size, _, height, width = yl.size()
        yh_reshaped = yh[0].view(batch_size, -1, height, width)
        combined = torch.cat((yl, yh_reshaped), dim=1)

        x = self.conv_1(combined)
        residual = x
        x = self.residual_blocks(x)
        x = self.conv_2(x)
        x += residual
        x = self.conv_3(x)
        x = self.pixel_shuffle(x)
        x = self.pixel_shuffle(x)

        yl_reconstructed = x[:, :3, :, :]
        yh_reconstructed = x[:, 3:, :, :].view(batch_size, 3, 3, height, width)
        reconstructed_image = self.ifm((yl_reconstructed, [yh_reconstructed]))
        out = torch.clamp(reconstructed_image, 0.0, 1.0)
        return out


if __name__ == '__main__':
    xfm = DWTForward(J=1, wave='db1')
    ifm = DWTInverse(mode='zero', wave='db1')

    x = torch.randn(1, 3, 256, 256)
    yl, yh = xfm(x)

    yh_reshaped = yh[0].view(1, -1, 128, 128)  # [1, 9, 128, 128]
    combined = torch.cat((yl, yh_reshaped), dim=1)  # [1, 12, 128, 128]

    yl_reconstructed = combined[:, :3, :, :]  # [1, 3, 128, 128]
    yh_reconstructed = combined[:, 3:, :, :].view(1, 3, 3, 128, 128)  # [1, 3, 3, 128, 128]

    reconstructed_image = ifm((yl_reconstructed, [yh_reconstructed]))