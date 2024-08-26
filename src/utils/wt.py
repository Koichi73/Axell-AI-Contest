import torch
import torch.nn as nn
import torch.nn.functional as F

class DWTLayer(nn.Module):
    def __init__(self, wavelet='haar'):
        super(DWTLayer, self).__init__()
        # Haarウェーブレットフィルタの定義
        self.filter_low = nn.Parameter(torch.tensor([[0.5, 0.5], [0.5, 0.5]]).unsqueeze(0).unsqueeze(0), requires_grad=False)
        self.filter_high = nn.Parameter(torch.tensor([[-0.5, -0.5], [0.5, 0.5]]).unsqueeze(0).unsqueeze(0), requires_grad=False)

    def forward(self, x):
        # 各チャネルにフィルタを適用するため、適切に繰り返す
        cA = F.conv2d(x, self.filter_low.repeat(x.size(1), 1, 1, 1), stride=2, groups=x.size(1))
        cH = F.conv2d(x, self.filter_high.repeat(x.size(1), 1, 1, 1), stride=2, groups=x.size(1))
        cV = F.conv2d(x, self.filter_low.transpose(2, 3).repeat(x.size(1), 1, 1, 1), stride=2, groups=x.size(1))
        cD = F.conv2d(x, self.filter_high.transpose(2, 3).repeat(x.size(1), 1, 1, 1), stride=2, groups=x.size(1))

        # 4つの成分をチャンネル方向に結合
        out = torch.cat([cA, cH, cV, cD], dim=1)

        return out

class WITLayer(nn.Module):
    def __init__(self, wavelet='haar'):
        super(WITLayer, self).__init__()
        # 逆変換用のフィルタ
        self.filter_low = nn.Parameter(torch.tensor([[0.5, 0.5], [0.5, 0.5]]).unsqueeze(0).unsqueeze(0), requires_grad=False)
        self.filter_high = nn.Parameter(torch.tensor([[-0.5, -0.5], [0.5, 0.5]]).unsqueeze(0).unsqueeze(0), requires_grad=False)

    def forward(self, x):
        batch_size, channels, height, width = x.size()
        channels_per_group = channels // 4

        # 各成分を分割
        cA = x[:, 0:channels_per_group, :, :]
        cH = x[:, channels_per_group:2*channels_per_group, :, :]
        cV = x[:, 2*channels_per_group:3*channels_per_group, :, :]
        cD = x[:, 3*channels_per_group:, :, :]

        # 各成分を逆変換して元の解像度に戻す
        up_cA = F.conv_transpose2d(cA, self.filter_low.repeat(channels_per_group, 1, 1, 1), stride=2, groups=channels_per_group)
        up_cH = F.conv_transpose2d(cH, self.filter_high.repeat(channels_per_group, 1, 1, 1), stride=2, groups=channels_per_group)
        up_cV = F.conv_transpose2d(cV, self.filter_low.transpose(2, 3).repeat(channels_per_group, 1, 1, 1), stride=2, groups=channels_per_group)
        up_cD = F.conv_transpose2d(cD, self.filter_high.transpose(2, 3).repeat(channels_per_group, 1, 1, 1), stride=2, groups=channels_per_group)

        # 各成分を足し合わせて再構成
        restored = up_cA + up_cH + up_cV + up_cD

        return restored

if __name__ == "__main__":
    # テスト用のダミーデータ
    x = torch.randn(1, 3, 1500, 2248)
    dwt = DWTLayer()
    wit = WITLayer()

    # DWT変換
    y = dwt(x)
    print(y.size())  # torch.Size([1, 48, 16, 16])

    # WIT変換
    z = wit(y)
    print(z.size())  # torch.Size