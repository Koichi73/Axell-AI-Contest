import torch
import torch.nn as nn

## Channel Attention (CA) Layer
class CALayer(nn.Module):
    def __init__(self, channel, reduction=16):
        super(CALayer, self).__init__()
        # global average pooling: feature --> point
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        # feature channel downscale and upscale --> channel weight
        self.conv_du = nn.Sequential(
                nn.Conv2d(channel, channel // reduction, 1, padding=0, bias=True),
                nn.ReLU(inplace=True),
                nn.Conv2d(channel // reduction, channel, 1, padding=0, bias=True),
                nn.Sigmoid()
        )

    def forward(self, x):
        y = self.avg_pool(x)
        y = self.conv_du(y)
        return x * y
    
## Residual Channel Attention Block (RCAB)
class RCAB(nn.Module):
    def __init__(self, num_feat, reduction):
        super(RCAB, self).__init__()
        self.conv1 = nn.Conv2d(num_feat, num_feat, kernel_size=3, padding=1, bias=True)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(num_feat, num_feat, kernel_size=3, padding=1, bias=True)
        self.ca = CALayer(num_feat, reduction)

    def forward(self, x):
        res = self.conv1(x)
        res = self.relu(res)
        res = self.conv2(res)
        res = self.ca(res)
        return x + res
    
# Residual Group (with several Residual Blocks)
class ResidualGroup(nn.Module):
    def __init__(self, num_feat, reduction, num_blocks):
        super(ResidualGroup, self).__init__()
        layers = []
        for _ in range(num_blocks):
            layers.append(RCAB(num_feat, reduction))
        self.residual_group = nn.Sequential(*layers)
        self.conv = nn.Conv2d(num_feat, num_feat, kernel_size=3, padding=1, bias=True)

    def forward(self, x):
        res = self.residual_group(x)
        res = self.conv(res)
        return x + res
    
# RCAN Model
class RCAN(nn.Module):
    def __init__(self, num_in_ch=3, num_out_ch=3, num_feat=32, num_groups=2, num_blocks=3, reduction=16):
        super(RCAN, self).__init__()
        # First convolution layer
        self.conv_in = nn.Conv2d(num_in_ch, num_feat, kernel_size=3, padding=1, bias=True)

        # Residual Groups
        self.residual_groups = nn.Sequential(
            *[ResidualGroup(num_feat, reduction, num_blocks) for _ in range(num_groups)]
        )

        # Last convolution layer
        self.conv_out = nn.Conv2d(num_feat, num_feat, kernel_size=3, padding=1, bias=True)

        # Upsampling layer
        self.upsampler = nn.Sequential(
            nn.Conv2d(num_feat, num_feat * 4, kernel_size=3, padding=1, bias=True),
            nn.PixelShuffle(2),
            nn.Conv2d(num_feat, num_out_ch, kernel_size=3, padding=1, bias=True)
        )

    def forward(self, x):
        x = self.conv_in(x)
        res = self.residual_groups(x)
        res = self.conv_out(res)
        x = x + res
        x = self.upsampler(x)
        return x

if __name__ == "__main__":
    # モデルの初期化
    # num_in_ch = 3  # 入力チャネル数（例：RGB画像なら3）
    # num_out_ch = 3  # 出力チャネル数
    # num_feat = 32  # 特徴チャネル数
    # num_groups = 2  # Residual Groupの数
    # num_blocks = 3  # 各Residual Group内のResidual Blockの数
    # reduction = 16  # Channel Attentionの縮小率
    # model = RCAN(num_in_ch, num_out_ch, num_feat, num_groups, num_blocks, reduction)
    model = RCAN()
    total_params = sum(p.numel() for p in model.parameters())
    print(f"モデルの総パラメータ数: {total_params}")