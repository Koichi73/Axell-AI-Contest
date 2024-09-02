from torch import nn

def create_conv_layer(in_channels, out_channels, kernel_size=3, padding=1, mean=0, std=0.001):
    """Create and initialize a convolutional layer.

    Args:
        in_channels (int): Number of input channels.
        out_channels (int): Number of output channels.
        kernel_size (int, optional): Size of the kernel.
        padding (int, optional): Amount of padding.
        mean (float, optional): Mean for the normal distribution.
        std (float, optional): Standard deviation for the normal distribution.
    
    Returns:
        nn.Conv2d: Initialized convolutional layer.
    """
    conv_layer = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, padding=padding)
    nn.init.normal_(conv_layer.weight, mean=mean, std=std)
    nn.init.zeros_(conv_layer.bias)
    return conv_layer

class ResidualBlock(nn.Module):
    """Residual block for EDSR model.

    Args:
        num_channels (int): Number of input and output channels for the convolutional layers.
    """
    def __init__(self, num_channels):
        super(ResidualBlock, self).__init__()
        self.conv1 = create_conv_layer(num_channels, num_channels)
        self.act = nn.ReLU()
        self.conv2 = create_conv_layer(num_channels, num_channels)

    def forward(self, x):
        residual = x
        out = self.act(self.conv1(x))
        out = self.conv2(out)
        out += residual
        return out
