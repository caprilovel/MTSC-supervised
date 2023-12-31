import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import pandas as pd
from einops import rearrange, reduce, repeat
import math




#################    ALiBi    ################
class AttnwithLinearBiases(nn.Module):
    """AttnwithLinearBiases adds linear biases to attention scores.

    Attnention scores are calculated as the dot product of the query and key
    """
    def __init__(self, heads, length) -> None:
        super().__init__()
        range_matrix = torch.arange(length)
        bias_matrix = repeat(range_matrix, 'i -> j i', j=length) - repeat(range_matrix, 'i -> i j', j=length)
        bias_matrix = torch.abs(bias_matrix)
        m = calculate_m(heads)
        m = m.unsqueeze(-1).unsqueeze(-1)
        bias_matrix = m * repeat(bias_matrix, 'i j -> h i j', h=heads)
        self.bias_matrix = nn.Parameter(bias_matrix, requires_grad=False)
        
    def forward(self, attn_score):
        
        return attn_score - self.bias_matrix.unsqueeze(0)
        


def calculate_m(heads):
    """calculate_m calculates the slope of the linear bias

    Args:
        heads (int): the number of heads
    """
    
    m = [ 2 ** -(8*(i+1)/heads) for i in range(heads)]
    return torch.tensor(m)

def ALiBi(attn_score, heads, length):
    range_matrix = torch.arange(length)
    bias_matrix = repeat(range_matrix, 'i -> j i', j=length) - repeat(range_matrix, 'i -> i j', j=length)
    bias_matrix = torch.abs(bias_matrix)
    m = calculate_m(heads)
    m = m.unsqueeze(-1).unsqueeze(-1)
    bias_matrix = m * repeat(bias_matrix, 'i j -> h i j', h=heads)
    bias_matrix = bias_matrix.to(attn_score.device)
    
    return attn_score - bias_matrix.unsqueeze(0)


################# qkv Projection ####################
class LinearProjection(nn.Module):
    """LinearProjection for qkv projection

    Generate qkv vectors by linear transformation 
    
    Args:
        dim (int): the dimension of the input, the raw dim which is not divided by the n_heads
        heads (int): the number of heads
        dim_head (int): the dimension of one head  
    
    Examples:
        >>> x = torch.randn(2, 512, 10) # Batch x Length x Dim
        >>> lp = LinearProjection(10, 8, 64)
        >>> print(*[i.shape for i in lp(x)], sep='\n') # batch x heads x length x head_dim
        torch.Size([2, 8, 512, 64])
        torch.Size([2, 8, 512, 64])
        torch.Size([2, 8, 512, 64])
        
    """
    def __init__(self, dim, heads = 8, dim_head = 64, dropout = 0, bias = True, using_innerdim=True) -> None:
        super().__init__()
        if using_innerdim:
            inner_dim = dim_head * heads
        else: 
            inner_dim = dim 
            assert dim % heads == 0, f"dim {dim} must be divisible by n_heads {heads}"
        self.heads = heads 
        self.to_q = nn.Linear(dim, inner_dim, bias=bias)
        self.to_kv = nn.Linear(dim, inner_dim * 2, bias=bias)
        self.dim = dim
        self.inner_dim = inner_dim
        
    def forward(self, x, attn_kv=None):
        '''
        Args:
          x: Batch x Length x Dim
          attn_kv: attention matrix for key and value, Batch x Length_kv x Dim
        
        Output:
          q: Batch x heads x length x head_dim
          k: Batch x heads x length_kv x head_dim
          v: Batch x heads x length_kv x head_dim
        
        Examples:
            >>> x = torch.randn(2, 512, 10)
            >>> lp = LinearProjection(10, 8, 64)
            >>> lp(x).shape
            torch.Size([2, 512, 512, 64])
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
    

class SepConv1d(nn.Module):
    '''SepConv1d
    Using a depthwise convolution followed by a pointwise convolution to generate the desired output channels. More lightweight than a normal convolution.
    
    Args:
        in_channels (int): the number of input channels
        out_channels (int): the number of output channels
        kernel_size (int): the size of the kernel in the depthwise convolution
        stride (int): the stride of the depthwise convolution, default 1
        padding (int): the padding of the depthwise convolution, default 0
        dilation (int): the dilation of the depthwise convolution, default 1
        act_layer (nn.Module): the activation layer, default nn.ReLU
        
    Examples:
        >>> x = torch.randn(2, 32, 200)
        >>> sepconv = SepConv1d(32, 64, 3)
        >>> sepconv(x).shape
        torch.Size([2, 64, 198])
        
    '''
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, act_layer=nn.ReLU) -> None:
        super(SepConv1d, self).__init__()
        
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride 
        
        self.depthwise = nn.Conv1d(in_channels, in_channels, kernel_size=kernel_size, stride=stride, padding=padding, dilation=dilation, groups=in_channels)
        self.pointwise = nn.Conv1d(in_channels, out_channels, kernel_size=1)
        self.act_layer = act_layer() or nn.Identity()
        
    def forward(self, x):
        x = self.depthwise(x)
        x = self.act_layer(x)
        x = self.pointwise(x)
        return x
        
    
class ConvProjection(nn.Module):
    ''' ConvProjection for qkv projection
    
    the input tensor will be projected to q, k, v by a sepconv layer which contains a depthwise convolution and a pointwise convolution. Since multi-head is employed, the output channels of the sepconv layer should be divided by the number of heads. For such reason, the output channels of the sepconv layer should be divisible by the number of heads.
    
    In the function follered, we always use c indicates the channels of the input, n indicates the length of the input and b indicates the batchsize of the input.
    
    Args:
        dim (int): the dimension of the input
        heads (int): the number of heads,  default to be 8
        dim_head (int): the dimension of one head, multiplying heads and dim_head will give you the hidden size(inner dim). default to be 64
        kernel_size (int): the size of the kernel in the depthwise convolution, default to be 3
        q_stride (int): the stride of the depthwise convolution for q, default to be 1
        k_stride (int): the stride of the depthwise convolution for k, default to be 1
        v_stride (int): the stride of the depthwise convolution for v, default to be 1
        dropout (float): the dropout rate, default to be 0
        
    Examples:
        >>> x = torch.randn(2, 512, 10) # batch x length x dim
        >>> cp = ConvProjection(10, 8, 64)
        >>> print(*[i.shape for i in cp(x)], sep='\n')
        torch.Size([2, 8, 512, 64])
        torch.Size([2, 8, 512, 64])
        torch.Size([2, 8, 512, 64])
         
    '''
    def __init__(self, dim, heads=8, dim_head=64, kernel_size=3, q_stride=1, k_stride=1, v_stride=1, dropout=0., last_stage=False, bias=True) -> None:
        
        super().__init__()
        
        inner_dim = heads * dim_head
        self.inner_dim = inner_dim
        self.heads = heads 
        pad = (kernel_size - q_stride)//2
        
        self.to_q = SepConv1d(dim, inner_dim, kernel_size, q_stride, pad)
        self.to_k = SepConv1d(dim, inner_dim, kernel_size, k_stride, pad)
        self.to_v = SepConv1d(dim, inner_dim, kernel_size, v_stride, pad)
        
    def forward(self, x, attn_kv=None):
        b, n, c, h = *x.shape, self.heads
        attn_kv = attn_kv or x
        x = rearrange(x, 'b n c -> b c n')
        attn_kv = rearrange(attn_kv, 'b n c -> b c n')
        
        q = self.to_q(x)
        # batch x inner_dim x length
        q = rearrange(q, 'b (h d) n -> b h n d', h=h)
        
        
        k = self.to_k(attn_kv)
        # batch x inner_dim x length
        k = rearrange(k, 'b (h d) n -> b h n d', h=h)
        
        v = self.to_v(attn_kv)
        # batch x inner_dim x length
        v = rearrange(v, 'b (h d) n -> b h n d', h=h)
        
        return q,k,v

################Attention#############
class MSAttention(nn.Module):
    '''Multihead Self Attention Module 
    
    An implement of Multihead Self Attention Module, which is used in Transformer.
    
    Args:
        dim: the dimension of the input, the raw dim which is not divided by the n_heads, int
        num_heads: the number of heads, int
        token_projection: the method of projection, str, default: 'linear'
        qkv_bias: whether to add bias to qkv projection, bool, default: True
        qk_scale: the scale of qk, float, default: None
        attn_drop: the dropout rate of attention, float, default: 0.
        proj_drop : the dropout rate of projection, float, default: 0.
        using_innerdim: if is true, then the inner dim is specific by the num_heads and head_dim, else the inner dim is same as the input dim, and the input dim should be divisible by the num_heads. default: True
    
    Examples:
        >>> x = torch.randn(2, 512, 256)
        >>> msa = MSAttention(256, 8, token_projection='linear')
        >>> print(msa(x).shape)

    '''
    def __init__(self, dim, num_heads, token_projection='linear', qkv_bias=True, qk_scale=None, attn_drop=0., proj_drop=0., using_innerdim=True, pos_emd="alibi") -> None:
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        head_dim = dim // num_heads 
        self.scale = qk_scale or head_dim ** -0.5
        
        if token_projection == 'conv':
            self.qkv_proj = ConvProjection(dim, num_heads, dim//num_heads, bias=qkv_bias)
        elif token_projection == 'linear':
            self.qkv_proj = LinearProjection(dim, num_heads, dim//num_heads, bias=qkv_bias, using_innerdim=using_innerdim)
        else:
            raise Exception("Projection error!")
        
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)
        
        self.softmax = nn.Softmax(dim=-1)
        
        # self.pe = AttnwithLinearBiases(num_heads, head_dim)
    
    def forward(self, x, attn_kv=None, mask=None):
        B_, N, C = x.shape
        q, k, v = self.qkv_proj(x, attn_kv)
        q = q * self.scale 
        attn = (q @ k.transpose(-2, -1))
        # attn = self.pe(attn)
        attn = ALiBi(attn, self.num_heads, N)
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

##################Transformer Block#############      
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
        
        
        
        self.attn = MSAttention(dim, num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale, 
                                attn_drop=attn_drop, proj_drop=drop, 
                                token_projection=token_projection)
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
        # self.abs_pos_enc = AbsPositionalEncoding(dim)
        
        
        
    def forward(self, x, mask=None):
        B, L, C = x.shape
        
        # x = self.abs_pos_enc(x * math.sqrt(self.dim)) 
        x = self.norm1(x)
        msa_in = x
        
        attn = self.attn(msa_in)
        ffn_in = attn + x 
        
        
        return ffn_in
######################Conv Layer#########################
class DownSample(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(DownSample, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        
        self.conv = nn.Sequential(
            nn.Conv1d(in_channels, out_channels, kernel_size=4, stride=2, padding=1),
        )
    def forward(self, x):
        B, L, C = x.shape
        x = rearrange(x, 'b l c -> b c l')
        x = self.conv(x)
        x = rearrange(x, 'b c l -> b l c')        
        return x

class UpSample(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(UpSample, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        
        self.conv = nn.Sequential(
            nn.ConvTranspose1d(in_channels, out_channels, kernel_size=2, stride=2),
        )
    def forward(self, x):
        B, L, C = x.shape
        x = rearrange(x, 'b l c -> b c l')
        x = self.conv(x)
        x = rearrange(x, 'b c l -> b l c')        
        return x
    
class InputProj(nn.Module):
    def __init__(self, in_channels=2, out_channels=32, kernel_size=3, stride=1, norm_layer=None, act_layer=nn.LeakyReLU):
        super(InputProj, self).__init__()
        act_layer = act_layer or nn.Identity
        norm_layer = norm_layer or nn.Identity
        self.proj = nn.Sequential(
            nn.Conv1d(in_channels, out_channels, kernel_size, stride=stride, padding=kernel_size//2),
            act_layer(inplace=True)
        )
        self.norm = norm_layer(out_channels)
        
    def forward(self, x):
        B, C, L = x.shape
        x = self.proj(x)
        x = rearrange(x, 'b c l -> b l c')
        x = self.norm(x)
        return x
    
class OutputProj(nn.Module):
    def __init__(self, in_channels=32, out_channels=2, kernel_size=3, stride=1, norm_layer=None, act_layer=None):
        super(OutputProj, self).__init__()
        act_layer = act_layer or nn.Identity
        norm_layer = norm_layer or nn.Identity
        self.proj = nn.Sequential(
            nn.Conv1d(in_channels, out_channels, kernel_size, stride=stride, padding=kernel_size//2),
            act_layer(inplace=True)
        )
        self.norm = norm_layer(out_channels)
        
    def forward(self, x):
        B, L, C = x.shape
        x = rearrange(x, 'b l c -> b c l')
        x = self.proj(x)
        x = self.norm(x)
        return x
######################Network#########################   

class AlibiNet(nn.Module):
    def __init__(self, transblock_num, input_dims, nclass, hidden_dim=64, n_heads=8, fc_dim=128, ):
        super(AlibiNet, self).__init__()
        
        self.convandnorm = nn.Sequential(
            *[
                nn.Conv1d(in_channels=input_dims, out_channels=hidden_dim, kernel_size=3, stride=1, padding=1),
                nn.LeakyReLU(),
                nn.BatchNorm1d(hidden_dim),
            ]
        )
        
        self.transblocks = nn.Sequential(
            *[TransformerBlock(
            dim=hidden_dim, num_heads=n_heads   
                ) for _ in range(transblock_num)]
        )
        
        self.gap = nn.AdaptiveAvgPool1d(1)
        
        self.fc = nn.Sequential(
            nn.LazyLinear(fc_dim),
            nn.LeakyReLU(),
            nn.LazyLinear(nclass),
            nn.Sigmoid()
        )
        
    def forward(self, x):
        b, c, l =x.shape
        x = self.convandnorm(x)
        x = rearrange(x, 'b c l -> b l c')
        x = self.transblocks(x)
        x = self.gap(x).squeeze(-1)
        x = self.fc(x)        
        return x

 
        
        


######################Transformer########################
class Transformer(nn.Module):
    def __init__(self, token_projection='linear', token_mlp='leff', skip_connect=True) -> None:
        super(Transformer, self).__init__()
        
        self.inputproj = InputProj()
        
        self.trans_enc1 = TransformerBlock(32, 8, token_projection=token_projection, token_mlp=token_mlp)
        self.downsample1 = DownSample(32, 64)

        self.trans_enc2 = TransformerBlock(64, 8,token_projection=token_projection, token_mlp=token_mlp)
        self.downsample2 = DownSample(64, 128)
        
        self.trans_enc3 = TransformerBlock(128, 8,token_projection=token_projection, token_mlp=token_mlp)
        self.downsample3 = DownSample(128, 256)
        
        self.trans_enc4 = TransformerBlock(256, 8,token_projection=token_projection, token_mlp=token_mlp)
        self.downsample4 = DownSample(256, 512)
        
        self.trans_dec1 = TransformerBlock(512, 8,token_projection=token_projection, token_mlp=token_mlp)
        self.upsample1 = UpSample(512, 256)
        
        self.trans_dec2 = TransformerBlock(256, 8,token_projection=token_projection, token_mlp=token_mlp)
        self.upsample2 = UpSample(256, 128)
        
        self.trans_dec3 = TransformerBlock(128, 8,token_projection=token_projection, token_mlp=token_mlp)
        self.upsample3 = UpSample(128, 64)
        
        self.trans_dec4 = TransformerBlock(64, 8,token_projection=token_projection, token_mlp=token_mlp)
        self.upsample4 = UpSample(64, 32)
        
        self.outproj = OutputProj()
        self.skip_connect = skip_connect
        
    def forward(self, x):
        x_proj = self.inputproj(x)
        
        x1 = self.downsample1(self.trans_enc1(x_proj))
        
        x2 = self.downsample2(self.trans_enc2(x1))
        
        x3 = self.downsample3(self.trans_enc3(x2))

        x4 = self.downsample4(self.trans_enc4(x3))

        y1 = self.upsample1(self.trans_dec1(x4))
        if self.skip_connect:
            
            y2 = self.upsample2(self.trans_dec2(x3 + y1))
            
            y3 = self.upsample3(self.trans_dec3(x2 + y2))
            
            y4 = self.upsample4(self.trans_dec4(x1 + y3))
            
            y_proj = self.outproj(x_proj + y4)
        else:
            y2 = self.upsample2(self.trans_dec2(y1))
            
            y3 = self.upsample3(self.trans_dec3(y2))
            
            y4 = self.upsample4(self.trans_dec4(y3))
            
            y_proj = self.outproj(y4)
        
        return y_proj