import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import pandas as pd
from einops import rearrange, reduce, repeat
import math


 
################# qkv Projection ####################
class LinearProjection(nn.Module):
    def __init__(self, dim, heads = 8, dim_head = 64, dropout = 0, bias = True) -> None:
        super().__init__()
        inner_dim = dim_head * heads
        self.heads = heads 
        self.to_q = nn.Linear(dim, inner_dim, bias=bias)
        self.to_kv = nn.Linear(dim, inner_dim * 2, bias=bias)
        self.dim = dim
        self.inner_dim = inner_dim
        
    def forward(self, x, attn_kv=None):
        '''
        Args:
          x: Batch x Length x Dim
        
        Output:
          q: Batch x heads x length x head_dim
          k: Batch x heads x length_kv x head_dim
          v: Batch x heads x length_kv x head_dim
        
        '''
        B_, N, C = x.shape
        if attn_kv is not None:
            attn_kv = attn_kv.unsqueeze(0).repeat(B_,1,1)
        else:
            attn_kv = x
        N_kv = attn_kv.size(1)
        q = self.to_q(x).reshape(B_, N, 1, self.heads, self.inner_dim // self.heads).permute(2, 0, 3, 1, 4)
        kv = self.to_kv(attn_kv).reshape(B_, N_kv, 2, self.heads, self.inner_dim // self.heads).permute(2, 0, 3, 1, 4)
        q = q[0]
        k,v = kv[0], kv[1]
        return q, k, v


################Attention#############
class MSAttention(nn.Module):
    def __init__(self, dim, num_heads, qkv_bias=True, qk_scale=None, attn_drop=0., proj_drop=0., *args, **kwargs) -> None:
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        head_dim = dim // num_heads 
        self.scale = qk_scale or head_dim ** -0.5
        self.qkv_proj = LinearProjection(dim, num_heads, dim//num_heads, bias=qkv_bias)
        
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)
        
        self.softmax = nn.Softmax(dim=-1)
    
    def forward(self, x, attn_kv=None, mask=None):
        B_, N, C = x.shape
        q, k, v = self.qkv_proj(x, attn_kv)
        q = q * self.scale 
        attn = (q @ k.transpose(-2, -1))
        
        if mask is not None:
            # nW = mask.shape[0]
            # mask = repeat(mask, 'nW m n -> nW m (n d)', d =ratio)
            pass 
        else:
            pass
    
        attn =self.attn_drop(attn)
        
        # v: batch x heads x length x head_dim
        # attn: batch x heads x length x length
        x = attn @ v
        # x: batch x heads x length x head_dim
        x = rearrange(x, 'b h l hd -> b l (h hd)')
        # x: batch x length x channels 
        x = self.proj(x)
        x = self.proj_drop(x)
        # x: batch * length * dims 
        return x 
        



###############MLP####################
def odd_floor(a):
    return int(a/2) * 2 + 1
class ECA_layer(nn.Module):
    def __init__(self, channels, gamma=2, b=1):
        super(ECA_layer, self).__init__()
        kernel_size = odd_floor(math.log(channels,2)/gamma + b/gamma)
        # print(kernel_size)
        self.gap = nn.AdaptiveAvgPool1d(1)
        self.conv = nn.Conv1d(1, 1, kernel_size=kernel_size, padding=(kernel_size-1)//2)
        self.act_layer = nn.Sigmoid()
        
    def forward(self, x):
        B, C, L = x.shape 
        attn = self.gap(x).transpose(1, -1)
        attn = self.conv(attn)
        attn = self.act_layer(attn).squeeze(1)
        # return size: batch * channels 
        return attn
class eca_layer_1d(nn.Module):
    def __init__(self, channels, k_size=3) -> None:
        super(eca_layer_1d, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool1d(1)
        self.conv = nn.Conv1d(1, 1, kernel_size=k_size, padding=(k_size - 1)//2, bias=False)
        self.sigmoid = nn.Sigmoid()
        self.channels = channels
        self.k_size = k_size
        
    def forward(self, x):
        y = self.avg_pool(x.transpose(-1, -2))
        y = self.conv(y.transpose(-1, -2))
        y = self.sigmoid(y)
        return x * y.expand_as(x)
class UniLinearProjection(nn.Module):
    def __init__(self, length, hidden_dim, dropout=0., bias=True) -> None:
        super(UniLinearProjection, self).__init__()
        self.length = length
        self.to_q = nn.Linear(length, hidden_dim, bias=bias)
        self.to_kv = nn.Linear(length, 2 * hidden_dim, bias=bias)
        
    def forward(self, x, attn_kv=None):
        B, L, C = x.shape 
        # x size: batch * length * channels
        x = rearrange(x, 'b l c -> b c l')
        if attn_kv:        
            attn_kv = attn_kv.unsqueeze(0).repeat(B, 1, 1)
        else:    
            attn_kv = x            
        q = self.to_q(x)
        kv = self.to_kv(attn_kv).reshape(B, C, 2, -1).permute(2, 0, 1, -1)
        k,v = kv[0], kv[1]
        # q,k,v size: batch * channels * hidden_dim
        return q, k, v
    
class UniConvProjction(nn.Module):
    def __init__(self, dim ) -> None:
        super().__init__()
        
        
    
class ChannelSelfAttention(nn.Module):
    def __init__(self, length, hidden_dim, token_projection='linear', qkv_bias=True, 
                 qk_scale=None, attn_drop=0., proj_drop=0.) -> None:
        super(ChannelSelfAttention, self).__init__()

        self.scale = qk_scale or hidden_dim ** -0.5
        
        if token_projection == 'linear':
            self.qkv_proj = UniLinearProjection(length, hidden_dim, dropout=proj_drop, bias=qkv_bias)
            
        elif token_projection == 'conv':
            # self.qkv_proj = 
            pass
        
        self.attn_drop = attn_drop
        # self.proj = nn.Linear(length, length)
        self.proj_drop = nn.Dropout(proj_drop)
        self.gap = nn.AdaptiveAvgPool1d(1)
        # self.softmax = nn.Softmax(dim=-1)
        
    def forward(self, x, attn_kv=None):
        B, L, C = x.shape 
        q, k, v = self.qkv_proj(x, attn_kv)
        # qkv size: batch * channels * hidden_dim
        q = q * self.scale 
        # attn size: batch * channels * channels 
        attn = (q @ k.transpose(-2, -1))
        # x size: batch * channels * hidden_size 
        y = attn @ v
        y = self.gap(y).transpose(-1, -2)
        return x * y
    
# class CaFF(nn.Module):
#     def __init__(self, length=32, hidden_dim=128, act_layer=nn.GELU, drop=0., use_gap=True) -> None:
#         super(CaFF, self).__init__()
#         self.dim = length 
#         self.hidden_dim = hidden_dim
#         self.use_gap = use_gap
        
#         if use_gap:
#             self.gap = nn.AdaptiveAvgPool1d(1)
#             self.csa = ChannelSelfAttention(1, hidden_dim) 
#         else:
#             self.csa = ChannelSelfAttention(length, hidden_dim) 
#         self.act_layer = act_layer() or nn.Identity()
#         self.dropout = nn.Dropout1d(drop)
        
#     def forward(self, x):
#         B, L, C = x.shape
#         if self.use_gap:
#             x = rearrange(x, 'b l c -> b c l')
#             x = self.gap(x)
#             x = rearrange(x, 'b c l -> b l c')
        
        
        
#         return 


        

class LeFF(nn.Module):
    def __init__(self, dim=32, hidden_dim=128, act_layer=nn.GELU, drop=0., use_eca=False) -> None:
        super().__init__()
        self.dim = dim
        self.hidden_dim = hidden_dim
        self.eca = eca_layer_1d(dim) if use_eca else nn.Identity()
        
        self.linear1 = nn.Sequential(nn.Linear(dim, hidden_dim), act_layer())
        self.dwconv = nn.Sequential(nn.Conv1d(hidden_dim, hidden_dim, groups=hidden_dim, kernel_size=3, stride=1, padding=1), act_layer())
        self.linear2 = nn.Sequential(nn.Linear(hidden_dim, dim))
    
    def forward(self, x):
        bs, hw, c = x.size()
        
        x = self.linear1(x)
        
        x = rearrange(x, 'b n c -> b c n')
        x = self.dwconv(x)
        x = rearrange(x, 'b c n -> b n c')
        
        x = self.linear2(x)
        
        x = self.eca(x)
        return x

class CAFF(nn.Module):
    def __init__(self, dim=32, hidden_dim=128, act_layer=nn.GELU, drop=0.,) -> None:
        super(CAFF, self).__init__()
        self.to_q = nn.Sequential(nn.Conv1d())
        


class FastLeFF(nn.Module):
    def __init__(self, dim=32, hidden_dim=128, act_layer=nn.GELU, drop=0.) -> None:
        super().__init__()
        self.dim = dim
        self.hidden_dim = hidden_dim
        
        self.linear1 = nn.Sequential(nn.Linear(dim ,hidden_dim), act_layer())
        self.dwconv = nn.Sequential(nn.Conv1d(hidden_dim, hidden_dim, kernel_size=3, padding=1, groups=hidden_dim, stride=1), act_layer())
        self.linear2 = nn.Sequential(nn.Linear(hidden_dim, dim))
        
        
    def forward(self, x):
        b, n, c = x.size()
        x = self.linear1(x)
        
        x = rearrange(x, 'b n c -> b c n')
        x = self.dwconv(x)
        x = rearrange(x, 'b c n -> b n c')
        
        x = self.linear2(x)
        return x




    
class MLP(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.) -> None:
        super().__init__()
        hidden_features = hidden_features or in_features
        out_features = out_features or in_features
        
        self.in_features = in_features
        self.hidden_features = hidden_features
        self.out_features = out_features
        
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)
        
    def forward(self, x):
        x = self.act(self.fc1(x))
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x 
#################Pos Encoder################
class AbsPositionalEncoding(nn.Module):
    """位置编码"""
    def __init__(self, num_hiddens, dropout=0., max_len=1000):
        super(AbsPositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(dropout)
        # 创建一个足够长的P
        self.P = torch.zeros((1, max_len, num_hiddens))
        X = torch.arange(max_len, dtype=torch.float32).reshape(
            -1, 1) / torch.pow(10000, torch.arange(
            0, num_hiddens, 2, dtype=torch.float32) / num_hiddens)
        self.P[:, :, 0::2] = torch.sin(X)
        self.P[:, :, 1::2] = torch.cos(X)

    def forward(self, X):
        X = X + self.P[:, :X.shape[1], :].to(X.device)
        return self.dropout(X)

    
class TransformerBlock(nn.Module):
    def __init__(self, dim, num_heads,  
                 mlp_ratio=4., qkv_bias=True, qk_scale=None, drop=0., attn_drop=0., drop_path=0., 
                 act_layer=nn.GELU, norm_layer=nn.LayerNorm, token_projection='linear', token_mlp='leff', 
                 modulator=False, cross_modulation=False) -> None:
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads 
        self.mlp_ratio = mlp_ratio
        self.token_mlp = token_mlp
        
        
        
        self.attn = MSAttention(dim, num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop)
        
        self.norm1 = norm_layer(dim)
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        if token_mlp in ['ffn', 'mlp']:
            self.mlp = MLP(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)
        elif token_mlp == 'leff':
            self.mlp = LeFF(dim, mlp_hidden_dim, act_layer=act_layer, drop=drop)
        elif token_mlp == 'fastleff':
            self.mlp = FastLeFF(dim, mlp_hidden_dim, act_layer=act_layer, drop=drop)
        elif token_mlp == 'csa':
            # self.mlp = ChannelSelfAttention() 
            pass 
        else:
            raise Exception('FFN Error!')
        self.abs_pos_enc = AbsPositionalEncoding(dim)
        
        
        
    def forward(self, x, mask=None):
        B, L, C = x.shape
        
        x = self.abs_pos_enc(x * math.sqrt(self.dim)) 
        x = self.norm1(x)
        msa_in = x
        
        attn = self.attn(msa_in)
        ffn_in = attn + x 
        
        ffn_in = self.norm2(ffn_in)
        ffn_out = self.mlp(ffn_in)
        
        return ffn_out + ffn_in


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
    def __init__(self, in_channels, hidden_channels, kernel_size, dilation=1, bias=True):
        '''
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
        '''
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
        super().__init__()
        kernel_size = stride * 2 - 1
        padding = (kernel_size - 1) // 2
        self.maxpool = nn.MaxPool1d(kernel_size=kernel_size, stride=stride, padding=padding)
        self.convpool = nn.Conv1d(in_channels, in_channels, kernel_size=kernel_size, stride=stride, padding=padding)    

    def forward(self, x):
        return torch.cat([self.maxpool(x), self.convpool(x)], dim=1)


class Convformer(nn.Module):
    def __init__(self, dim, nclass, hidden_dims=[32, 64, 128, 256], num_heads=8, using_posemd=True) -> None:
        super().__init__()
        
        assert len(hidden_dims) >= 4, 'hidden_dims must be a list with 4 elements'
        self.using_posemd = using_posemd
        
        self.init_conv = nn.Conv1d(dim, hidden_dims[0], kernel_size=3, padding=1, stride=2)
        


        
        self.block2 = nn.Sequential( 
            ResBottleneckBlock(hidden_dims[0], hidden_dims[0] * 2, kernel_size=7, dilation=3),
            ResBottleneckBlock(hidden_dims[0], hidden_dims[0] * 2, kernel_size=7, dilation=3),
            ResBottleneckBlock(hidden_dims[0], hidden_dims[0] * 2, kernel_size=7, dilation=3))
        
        self.ds1 = Downsample(hidden_dims[0])
        
        self.block3 = nn.Sequential( 
            ResBottleneckBlock(hidden_dims[1], hidden_dims[1] * 2, kernel_size=3, dilation=2),
            ResBottleneckBlock(hidden_dims[1], hidden_dims[1] * 2, kernel_size=3, dilation=2),
            ResBottleneckBlock(hidden_dims[1], hidden_dims[1] * 2, kernel_size=3, dilation=1))
            
        
        self.ds2 = Downsample(hidden_dims[1])
        
        self.block4 = nn.Sequential( 
            ResBottleneckBlock(hidden_dims[2], hidden_dims[2] * 2, kernel_size=3, dilation=1),
            ResBottleneckBlock(hidden_dims[2], hidden_dims[2] * 2, kernel_size=3, dilation=1),
            ResBottleneckBlock(hidden_dims[2], hidden_dims[2] * 2, kernel_size=3, dilation=1),)
         
        self.ds3 = Downsample(hidden_dims[2])
        
        
        if using_posemd:
            self.posemd = AbsPositionalEncoding(hidden_dims[3])        
        self.block5 = nn.Sequential(
            MSAttention(hidden_dims[3], num_heads=8, qkv_bias=True, qk_scale=None, attn_drop=0., proj_drop=0.),
            # MSAttention(hidden_dims[0], num_heads=8, qkv_bias=True, qk_scale=None, attn_drop=0., proj_drop=0.),
            # MSAttention(hidden_dims[0], num_heads=8, qkv_bias=True, qk_scale=None, attn_drop=0., proj_drop=0.),
        )
        
        self.gap = nn.AdaptiveAvgPool1d(1)
        
        self.mlp = nn.Sequential(
            nn.Linear(hidden_dims[3], 256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(256, nclass))
        
    def forward(self, x):
        B, C, L = x.shape
        
        x = self.init_conv(x)
        
        x = self.block2(x)
        x = self.ds1(x)
        
        x = self.block3(x)
        x = self.ds2(x)
        
        x = self.block4(x)
        x = self.ds3(x)
        
        x = rearrange(x, 'b c l -> b l c')
        if self.using_posemd:
            x = self.posemd(x)
        x = self.block5(x)
        x = rearrange(x, 'b l c -> b c l')
        
        x = self.gap(x).squeeze(-1)
        return self.mlp(x)    
        
        