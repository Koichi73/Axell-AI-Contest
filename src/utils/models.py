# 4倍拡大サンプルモデル(ESPCN)の構造定義
# 参考 https://github.com/Nhat-Thanh/ESPCN-Pytorch
# モデルへの入力はN, C, H, Wの4次元入力で、チャンネルはR, G, Bの順、画素値は0~1に正規化されている想定となります。  
# また、出力も同様のフォーマットで、縦横の解像度(H, W)が4倍となります。

import torch
from torch import nn, tensor, clip
from transformers import Swin2SRForImageSuperResolution

class ESPCN4x(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.scale = 4
        self.conv_1 = nn.Conv2d(in_channels=1, out_channels=64, kernel_size=5, padding=2)
        nn.init.normal_(self.conv_1.weight, mean=0, std=0.001)
        nn.init.zeros_(self.conv_1.bias)

        self.act = nn.ReLU()

        self.conv_2 = nn.Conv2d(in_channels=64, out_channels=32, kernel_size=3, padding=1)
        nn.init.normal_(self.conv_2.weight, mean=0, std=0.001)
        nn.init.zeros_(self.conv_2.bias)
        
        self.conv_3 = nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, padding=1)
        nn.init.normal_(self.conv_3.weight, mean=0, std=0.001)
        nn.init.zeros_(self.conv_3.bias)

        self.conv_4 = nn.Conv2d(in_channels=32, out_channels=(1 * self.scale * self.scale), kernel_size=3, padding=1)
        nn.init.normal_(self.conv_4.weight, mean=0, std=0.001)
        nn.init.zeros_(self.conv_4.bias)

        self.pixel_shuffle = nn.PixelShuffle(self.scale//2)

    def forward(self, X_in: tensor) -> tensor:
        # 入力画像をデータ拡張する（左右反転、上下反転）
        X_flip_lr = torch.flip(X_in, [3])  # 左右反転
        X_flip_ud = torch.flip(X_in, [2])  # 上下反転
        X_flip_lr_ud = torch.flip(X_in, [2, 3])  # 左右上下反転

        # 4枚それぞれに対して推論を行う
        X_orig = self._forward_once(X_in)
        X_lr = self._forward_once(X_flip_lr)
        X_ud = self._forward_once(X_flip_ud)
        X_lr_ud = self._forward_once(X_flip_lr_ud)

        # 元の向きに戻す
        X_lr = torch.flip(X_lr, [3])
        X_ud = torch.flip(X_ud, [2])
        X_lr_ud = torch.flip(X_lr_ud, [2, 3])

        # 4枚の結果を合成する
        X_out = (X_orig + X_lr + X_ud + X_lr_ud) / 4.0
        return X_out

    def _forward_once(self, X_in: tensor) -> tensor:
        X = X_in.reshape(-1, 1, X_in.shape[-2], X_in.shape[-1])
        X = self.act(self.conv_1(X))
        X = self.act(self.conv_2(X))
        X = self.act(self.conv_3(X))
        X = self.conv_4(X)
        X = self.pixel_shuffle(X)
        X = self.pixel_shuffle(X)
        X = X.reshape(-1, 3, X.shape[-2], X.shape[-1])
        X_out = clip(X, 0.0, 1.0)
        return X_out

class Swin2SR(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.model = Swin2SRForImageSuperResolution.from_pretrained("caidas/swin2SR-classical-sr-x4-64")
    
    def forward(self, x):
        return self.model(x)
