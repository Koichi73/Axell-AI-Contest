import torch
import torch.nn as nn
import pywt
import numpy as np

class DWTLayer(nn.Module):
    def __init__(self, wavelet='haar'):
        super(DWTLayer, self).__init__()
        self.wavelet = wavelet
    
    def forward(self, x):
        # PyTorchのテンソルをNumPy配列に変換
        x_np = x.detach().cpu().numpy()
        
        # 各チャネルごとにDWTを適用し、12成分を取得
        coeffs = []
        for i in range(x_np.shape[1]):  # チャンネル数 (R, G, B)
            cA, (cH, cV, cD) = pywt.dwt2(x_np[:, i, :, :], self.wavelet)
            coeffs.append(cA)
            coeffs.append(cH)
            coeffs.append(cV)
            coeffs.append(cD)
        
        # 12成分をまとめたテンソルを作成
        coeffs_tensor = torch.tensor(np.stack(coeffs, axis=1)).to(x.device)
        
        return coeffs_tensor

class WITLayer(nn.Module):
    def __init__(self, wavelet='haar'):
        super(WITLayer, self).__init__()
        self.wavelet = wavelet
    
    def forward(self, coeffs_tensor):
        # TensorをNumPy配列に変換
        coeffs_np = coeffs_tensor.detach().cpu().numpy()

        # 12チャンネルのテンソルをRGBごとの成分に分割
        batch_size, _, height, width = coeffs_tensor.shape
        restored_images = []
        
        for i in range(batch_size):
            channels = []
            for j in range(3):  # RGBの3チャンネル
                # 各チャンネルに対応する4つの成分を取得
                cA = coeffs_np[i, j * 4, :, :]
                cH = coeffs_np[i, j * 4 + 1, :, :]
                cV = coeffs_np[i, j * 4 + 2, :, :]
                cD = coeffs_np[i, j * 4 + 3, :, :]
                
                # 逆ウェーブレット変換を適用
                restored_channel = pywt.idwt2((cA, (cH, cV, cD)), self.wavelet)
                channels.append(restored_channel)
            
            # RGBチャンネルを結合して1つの画像に
            restored_image = np.stack(channels, axis=0)
            restored_images.append(restored_image)
        
        # NumPy配列をTensorに変換
        restored_images_tensor = torch.tensor(np.stack(restored_images)).to(coeffs_tensor.device)
        
        return restored_images_tensor