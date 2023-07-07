import torch
import torch.nn as nn
import torch.nn.functional as F

class Conv3dSame(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, dilation=1):
        super(Conv3dSame, self).__init__()

        padding = self._calculate_padding(kernel_size, dilation)
        self.conv = nn.Conv3d(in_channels, out_channels, kernel_size, stride, padding, dilation)

    def forward(self, x):
        return self.conv(x)

    @staticmethod
    def _calculate_padding(kernel_size, dilation):
        # Calculate padding to maintain spatial dimensions
        padding = [((k - 1) * d + 1) // 2 for k, d in zip(kernel_size, dilation)]
        return padding