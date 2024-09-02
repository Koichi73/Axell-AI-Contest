import torch
from torch import nn
from pytorch_wavelets import DWTForward, DWTInverse
from models.common import create_conv_layer, ResidualBlock

class EDWSR(nn.Module):
    """Enhanced Deep Wavelet Super-Resolution (EDWSR) model.

    Args:
        num_residual_blocks (int, optional): Number of residual blocks.
        scale_factor (int, optional): Upscaling factor.
    
    References:
        「ニューラルネットワークを用いた単一画像超解像のウェーブレット変換による高速化」
        https://ipsj.ixsq.nii.ac.jp/ej/?action=repository_uri&item_id=212269&file_id=1&file_no=1
        *Open the link to start downloading the PDF.
    """
    def __init__(self, num_residual_blocks=10, scale_factor=4, wave='db1'):
        super(EDWSR, self).__init__()
        self.num_residual_blocks = num_residual_blocks
        self.scale = scale_factor
        self.wave = wave
        self.xfm = DWTForward(J=1, wave=self.wave)
        self.ifm = DWTInverse(wave=self.wave)

        self.conv_1 = create_conv_layer(in_channels=12, out_channels=64)
        self.residual_blocks = nn.Sequential(
            *[ResidualBlock(64) for _ in range(self.num_residual_blocks)]
        )
        self.conv_2 = create_conv_layer(in_channels=64, out_channels=64)
        self.conv_3 = create_conv_layer(in_channels=64, out_channels=(12 * self.scale * self.scale))
        self.pixel_shuffle = nn.PixelShuffle(self.scale // 2)

    def forward(self, x):
        """
        Notes:
            - If the size is odd, the output size is shifted by the Wavelet transform,
              so pad and crop the image to make it compatible.
            - Normalization can be done by uncommenting.
        """
        # Pad the input tensor to make it compatible with the DWT
        orig_height, orig_width = x.size(2), x.size(3)
        x = torch.nn.functional.pad(x, (0, orig_width % 2, 0, orig_height % 2))

        # Apply the DWT to the input tensor
        yl, yh = self.xfm(x)
        batch_size, _, height, width = yl.size()
        yh_reshaped = yh[0].view(batch_size, -1, height, width)
        combined = torch.cat((yl, yh_reshaped), dim=1) # shape: (batch_size, 12, height, width)

        # Normalize the tensor
        # min_vals = combined.amin(dim=[2, 3], keepdim=True)
        # max_vals = combined.amax(dim=[2, 3], keepdim=True)
        # normalized_combined = (combined - min_vals) / (max_vals - min_vals + 1e-8)

        # Pass the tensor through the network
        x = self.conv_1(combined)
        residual = x
        x = self.residual_blocks(x)
        x = self.conv_2(x)
        x += residual
        x = self.conv_3(x)
        x = self.pixel_shuffle(x)
        x = self.pixel_shuffle(x)

        # Denormalize the tensor
        # reconstructed_combined = x * (max_vals - min_vals) + min_vals

        # Reconstruct and crop the image
        yl_reconstructed = x[:, :3, :, :]
        yh_reconstructed = x[:, 3:, :, :].view(batch_size, 3, 3, height * self.scale, width * self.scale)
        reconstructed_image = self.ifm((yl_reconstructed, [yh_reconstructed]))
        reconstructed_image = reconstructed_image[:, :, :orig_height * self.scale, :orig_width * self.scale]
        out = torch.clamp(reconstructed_image, 0.0, 1.0)
        return out

if __name__ == "__main__":
    model = EDWSR()
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total number of parameters in the model: {total_params}")
