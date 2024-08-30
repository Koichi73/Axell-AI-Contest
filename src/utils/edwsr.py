import torch
from torch import nn
from pytorch_wavelets import DWTForward, DWTInverse
from utils.models import EDSR, ResidualBlock

class EDWSR(EDSR):
    def __init__(self):
        super(EDWSR, self).__init__()
        self.xfm = DWTForward(J=1, wave='db1')
        self.ifm = DWTInverse(mode='zero', wave='db1')

        # 入力チャンネル数を12に変更
        self.conv_1 = nn.Conv2d(in_channels=12, out_channels=64, kernel_size=3, padding=1)
        nn.init.normal_(self.conv_1.weight, mean=0, std=0.001)
        nn.init.zeros_(self.conv_1.bias)

        # Residual blocks
        self.residual_blocks = nn.Sequential(
            *[ResidualBlock(64) for _ in range(10)]
        )

        # Second convolution layer
        self.conv_2 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, padding=1)
        nn.init.normal_(self.conv_2.weight, mean=0, std=0.001)
        nn.init.zeros_(self.conv_2.bias)

        # 出力チャンネル数を変更
        self.conv_3 = nn.Conv2d(in_channels=64, out_channels=(12 * self.scale * self.scale), kernel_size=3, padding=1)
        nn.init.normal_(self.conv_3.weight, mean=0, std=0.001)
        nn.init.zeros_(self.conv_3.bias)

    def forward(self, x):
        orig_height, orig_width = x.size(2), x.size(3)
        x = torch.nn.functional.pad(x, (0, orig_width % 2, 0, orig_height % 2))

        yl, yh = self.xfm(x)
        batch_size, _, height, width = yl.size()
        yh_reshaped = yh[0].view(batch_size, -1, height, width)
        combined = torch.cat((yl, yh_reshaped), dim=1)

        # min_vals = combined.amin(dim=[2, 3], keepdim=True)
        # max_vals = combined.amax(dim=[2, 3], keepdim=True)

        # # チャネルごとに正規化
        # normalized_combined = (combined - min_vals) / (max_vals - min_vals + 1e-8)

        x = self.conv_1(combined)
        residual = x
        x = self.residual_blocks(x)
        x = self.conv_2(x)
        x += residual
        x = self.conv_3(x)
        x = self.pixel_shuffle(x)
        x = self.pixel_shuffle(x)

        # reconstructed_combined = x * (max_vals - min_vals) + min_vals

        yl_reconstructed = x[:, :3, :, :]
        yh_reconstructed = x[:, 3:, :, :].view(batch_size, 3, 3, height * self.scale, width * self.scale)
        reconstructed_image = self.ifm((yl_reconstructed, [yh_reconstructed]))
        reconstructed_image = reconstructed_image[:, :, :orig_height * self.scale, :orig_width * self.scale]

        out = torch.clamp(reconstructed_image, 0.0, 1.0)
        return out

if __name__ == '__main__':
    model = EDWSR()
    total_params = sum(p.numel() for p in model.parameters())
    print(f"モデルの総パラメータ数: {total_params}")