import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import pandas as pd
from einops import rearrange, reduce, repeat


# generate a resnet 19 for 1d data

class ResBottleneckBlock(nn.Module):
    '''BottleNeck Block for ResNet
    
    An implementation of ResNet bottleneck block which is used in 1d ResNet. This module contains three conv layers, the middle one is main conv layer with dilation rate. The rest of the layers are 1x1 conv layers. The input and output channels are the same.
    
    Args:
        in_channels: int. input channels
        hidden_channels: int. hidden channels
        kernel_size: int. kernel size of the middle conv layer
        dilation: int. dilation rate of the middle conv layer
        bias: bool. whether to use bias in conv layers
        
    Examples:
        >>> block = ResBottleneckBlock(64, 128, 3, 1, True)
        >>> tensor1 = torch.randn(1, 64, 100)
        >>> tensor2 = block(tensor1)
        >>> tensor2.shape
        torch.Size([1, 64, 100])
        
    '''
    def __init__(self, in_channels, hidden_channels, kernel_size, dilation=1, bias=True      
    ):
        super(ResBottleneckBlock, self).__init__()
        
        padding = (kernel_size - 1) // 2 * dilation
        
        self.conv1 = nn.Conv1d(in_channels, hidden_channels, kernel_size=1, stride=1, padding=0, bias=bias)
        self.conv2 = nn.Conv1d(hidden_channels, hidden_channels, kernel_size=kernel_size, dilation=dilation, padding=padding, bias=bias)
        self.conv3 = nn.Conv1d(hidden_channels, in_channels, kernel_size=1, stride=1, padding=0, bias=bias)
        
        self.activate = nn.LeakyReLU(0.2, inplace=True)
        
        self.bn1 = nn.BatchNorm1d(hidden_channels)
        self.bn2 = nn.BatchNorm1d(hidden_channels)
        self.bn3 = nn.BatchNorm1d(in_channels) 
        
    def forward(self, x):
        identity = x
        out = self.activate(self.conv1(x))
        out = self.bn1(out)
        out = self.activate(self.conv2(out))
        out = self.bn2(out)
        out = self.activate(self.conv3(out))
        out = self.bn3(out)
        
        out += identity
        out = F.relu(out)
        return out

class Downsample(nn.Module):
    '''downsample module for ResNet
    
    An implement of the downsample module for ResNet. This module contains a maxpool layer and a conv layer. And the outputs of the two layers are concatenated. The channels would be doubled after this module and the length of the sequence would be almost halved( and can be controlled by the stride).
    
    Args:
        in_channels: int. input channels
        stride: int, default to be 2. stride of the maxpool layer
    
    Examples:
        >>> ds = Downsample(64)
        >>> tensor1 = torch.randn(1, 64, 100)
        >>> tensor2 = ds(tensor1)
        >>> tensor2.shape
        torch.Size([1, 128, 50])
    '''
    def __init__(self, in_channels, stride=2) -> None:
        super().__init__()
        kernel_size = stride * 2 - 1
        padding = (kernel_size - 1) // 2
        self.maxpool = nn.MaxPool1d(kernel_size=kernel_size, stride=stride, padding=padding)
        self.convpool = nn.Conv1d(in_channels, in_channels, kernel_size=kernel_size, stride=stride, padding=padding)
        
    def forward(self, x):
        return torch.cat([self.maxpool(x), self.convpool(x)], dim=1)
        
        
class ResNet19_1d(nn.Module):
    def __init__(self, in_channels, hidden_channels=[64, 128, 256, 512, 1024], *args, **kwargs):
        super(ResNet19_1d, self).__init__()
        
        self.proj = nn.Conv1d(in_channels, hidden_channels[0], kernel_size=1, stride=1, padding=0, bias=False)
        
        hidden_rate = 2.0
           
        self.block1 = nn.Sequential(
            ResBottleneckBlock(hidden_channels[0], int(hidden_rate * hidden_channels[0]), kernel_size=7, dilation=3, bias=True),
            ResBottleneckBlock(hidden_channels[0], int(hidden_rate * hidden_channels[0]), kernel_size=7, dilation=3, bias=True),
            ResBottleneckBlock(hidden_channels[0], int(hidden_rate * hidden_channels[0]), kernel_size=3, dilation=1, bias=True),
        )
        
        self.ds1 = Downsample(hidden_channels[0])
        
        self.block2 = nn.Sequential(
            ResBottleneckBlock(hidden_channels[1], int(hidden_rate * hidden_channels[1]), kernel_size=3, dilation=1, bias=True),
            ResBottleneckBlock(hidden_channels[1], int(hidden_rate * hidden_channels[1]), kernel_size=3, dilation=1, bias=True),
            ResBottleneckBlock(hidden_channels[1], int(hidden_rate * hidden_channels[1]), kernel_size=3, dilation=1, bias=True),)
        
        self.ds2 = Downsample(hidden_channels[1])
        
        self.block3 = nn.Sequential(
            ResBottleneckBlock(hidden_channels[2], int(hidden_rate * hidden_channels[2]), kernel_size=3, dilation=1, bias=True),
            ResBottleneckBlock(hidden_channels[2], int(hidden_rate * hidden_channels[2]), kernel_size=3, dilation=1, bias=True),
            ResBottleneckBlock(hidden_channels[2], int(hidden_rate * hidden_channels[2]), kernel_size=3, dilation=1, bias=True),
            ResBottleneckBlock(hidden_channels[2], int(hidden_rate * hidden_channels[2]), kernel_size=3, dilation=1, bias=True),)
        
        self.ds3 = Downsample(hidden_channels[2])
        
        self.block4 = nn.Sequential(
            ResBottleneckBlock(hidden_channels[3], int(hidden_rate * hidden_channels[3]), kernel_size=3, dilation=1, bias=True),
            ResBottleneckBlock(hidden_channels[3], int(hidden_rate * hidden_channels[3]), kernel_size=3, dilation=1, bias=True),
            ResBottleneckBlock(hidden_channels[3], int(hidden_rate * hidden_channels[3]), kernel_size=3, dilation=1, bias=True),
        )
        

    def forward(self, x):
        x = self.proj(x)
        x = self.ds1(self.block1(x))
        x = self.ds2(self.block2(x))
        x = self.ds3(self.block3(x))
        x = self.block4(x)

        return x 
    
class ResNet_1d(nn.Module):
    '''
    An resnet classifier for 1d sequence. The input sequence should be in the shape of (batch_size, in_channels, seq_len). And the output would be in the shape of (batch_size, nclass).
    
    Args:
        in_channels: int. input channels
        nclass: int. number of classes
        
    Examples:
        >>> resnet = ResNet_1d(1, 10)
        >>> tensor1 = torch.randn(1, 1, 100)
        >>> tensor2 = resnet(tensor1)
        >>> tensor2.shape
        torch.Size([1, 10])
    '''
    def __init__(self, in_channels, nclass, ):
        super(ResNet_1d, self).__init__()
        self.feature_ext = ResNet19_1d(in_channels, )
        
        self.gap = nn.AdaptiveAvgPool1d(1)
        
        self.mlp = nn.Sequential(
            nn.LazyLinear(512),
            # nn.BatchNorm1d(512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.LazyLinear(256),
            # nn.BatchNorm1d(256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.LazyLinear(nclass))


    def forward(self, x):
        x = self.feature_ext(x)
        x = self.gap(x).squeeze(-1)
        return self.mlp(x)