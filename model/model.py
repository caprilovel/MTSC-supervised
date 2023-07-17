import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import pandas as pd
from einops import rearrange, reduce, repeat

class linear_trans(nn.Module):
    def __init__(self, in_dims, out_dims, conv_size=1, ):
        super(linear_trans, self).__init__()
        self.conv_layer = nn.Conv1d(in_dims, out_dims, kernel_size=conv_size)
        
        
        
    def forward(self, x):
        return

class mhsa(nn.Module):
    def __init__(self, heads, dims, hidden_dims, ):
        super(mhsa, self).__init__()
        self.heads = heads
        self.dims = dims
        self.hidden_dims = hidden_dims
        
        
        
        
    def forward(self, x):
        
        return
    
    
class EmbedAttn(nn.Module):
    def __init__(self, embed_dim, hidden_dim, act_func=nn.LeakyReLU):
        """__init__ embedding attention

        Embedding attention, add attention to a specific embedding among multiple embeddings.

        Args:
            embed_dim (integer): the dim of the embed. 
            hidden_dim (integer): the dim of the hidden state.
            
        """
        super(EmbedAttn, self).__init__()
        self.linear = nn.Linear(embed_dim, hidden_dim)
        self.act_layer = act_func()
        self.gap = nn.AdaptiveAvgPool1d(1)
        
    def forward(self, x):
        y = self.linear(x)
        y = self.act_layer(y)
        y = self.gap(x).squeeze(-1)
        a = F.softmax(y, dim=1)    
        return x * a.unsqueeze(-1) 

class EChannelAttn(nn.Module):
    def __init__(self, channels, k_size=3):
        super(EChannelAttn, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool1d(1)
        self.conv = nn.Conv1d(1, 1, kernel_size=k_size, padding=(k_size-1)//2, bias=False)
        self.sigmoid = nn.Sigmoid()
        self.channels = channels 
        self.k_size = k_size 
        
        
    def forward(self, x):
        y = self.avg_pool(x.transpose(-1, -2))
        y = self.conv(y.transpose(-1, -2))
        y = self.sigmoid(y)
        return x * y.expand_as(x)

class ScaleLayer(nn.Module):
    def __init__(self, in_channels, out_channels, lstm_hidden_size, lstm_layers, kernel_size=[3,3], stride=[1,1], padding=[1,1], lstm_bidirectional=True, use_attn=False, act_func=nn.LeakyReLU):
        """__init__ init function of the pyramid scale layer 

        Init function of the pyramid scale layer.

        Args:
            in_channels (interger): channels/features/dimensions of the input tensor 
            out_channels (interger): channels/features/dimensions of the expect output tensor
            kernel_size (array, intergers): the array of the convolution kernel size   
            stride (array, intergers): the array of the convolution stride size   
            padding (array, intergers): the array of the convolution padding size   
            lstm_hidden_size (interger): the number of hidden features of the lstm hidden state.
            lstm_layers (interger): number of recurrent layers.
            lstm_bidirectional (bool, optional): if True, becomes a bidirectional LSTM. Defaults to True.
            use_attn (bool, optional): if True, An ECA module will be used first. Defaults to False.
            act_func (_type_, optional): the class of the used activation function. Defaults to nn.LeakyReLU.
        """
        super(ScaleLayer, self).__init__()
        self.use_attn = use_attn
        
        lstm_dict = {
            "input_size":in_channels,
            "hidden_size":lstm_hidden_size,
            "num_layers":lstm_layers,
            "batch_first":True,
            "bidirectional":lstm_bidirectional,
        }
        self.lstm = nn.LSTM(**lstm_dict)

        
        self.conv1 = nn.Conv1d(in_channels, in_channels, kernel_size=kernel_size[0], padding=padding[0], stride=stride[0])
        self.batch_norm_conv1 = nn.BatchNorm1d(in_channels)
        
        self.conv2 = nn.Conv1d(in_channels, in_channels, kernel_size=kernel_size[1], padding=padding[1], stride=stride[1])
        self.batch_norm_conv2 = nn.BatchNorm1d(in_channels)
        
        self.downconv3 = nn.Conv1d(in_channels, out_channels, kernel_size=3, stride=2, padding=1)
        self.batch_norm_conv3 = nn.BatchNorm1d(out_channels)
        
        self.act_layer = act_func()
        
        if use_attn:
            self.channelattn = EChannelAttn(in_channels)
        
            
    
        
    def forward(self, x):
        """forward forward function. 

        forward function of the pyramid scale layer.

        Args:
            x (tensor): the input tensor, should be provided as (batch, feature, seq)

        Returns:
            lstmy: the lstm output 
            cnny: the convolutional layer output
        """
        N = x.size(0)
        if self.use_attn:
            x = self.channelattn(x)
        
        lstmx = rearrange(x, "b c l -> b l c")
        lstmy, _ = self.lstm(lstmx)
        
        x1 = self.conv1(x)
        x1 = self.batch_norm_conv1(x1)
        x1 = self.act_layer(x1)
        
        x2 = self.conv2(x1)
        x2 = self.batch_norm_conv2(x2)
        x2 = self.act_layer(x2)
        
        x2 += x
        
        x3 = self.downconv3(x2)
        x3 = self.batch_norm_conv3(x3)
        convy = self.act_layer(x3)
        
        lstmy = reduce(lstmy, 'b l c -> b c', "mean")
            
        return convy, lstmy
    
    
class FusionModule(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=1, padding=0, stride=1):
        super(FusionModule, self).__init__()
        self.conv1 = nn.Conv1d(in_channels=in_channels, out_channels=in_channels, kernel_size=kernel_size, padding=padding, stride=stride)
        self.batch_norm_layer1 = nn.LazyBatchNorm1d()
        self.act_layer = nn.LeakyReLU()
        
        self.conv2 = nn.Conv1d(in_channels*2, out_channels, kernel_size=1)
        self.batch_norm_layer2 = nn.LazyBatchNorm1d() 
        
        
    def forward(self, x):
        """forward _summary_

        _extended_summary_

        Args:
            x (torch.tensor): input tensor, shape should be [batch, channels, length]

        Returns:
            _type_: _description_
        """
        y = self.conv1(x)
        y = self.batch_norm_layer1(y)
        y = self.act_layer(y)
        y = torch.cat([x, y], dim=1)
        y = self.conv2(y)
        y = self.batch_norm_layer2(y)
        y = self.act_layer(y)
        return  y


class MMTSC(nn.Module):
    def __init__(self, nclass, in_channels, len_ts, dropout=0.5, embed_dim=256,  pyramid_layer_num=4, fc_layers=[500, 300], use_lstm=True, use_attn=True, use_proto=False):
        """__init__ the MMTSC network

        _extended_summary_

        Args:
            nclass (_type_): _description_
            in_channels (_type_): _description_
            len_ts (_type_): _description_
            dropout (float, optional): _description_. Defaults to 0.5.
            embed_dim (int, optional): _description_. Defaults to 256.
            pyramid_layer_num (int, optional): _description_. Defaults to 4.
            fc_layers (list, optional): _description_. Defaults to [500, 300].
            use_lstm (bool, optional): _description_. Defaults to True.
            use_attn (bool, optional): _description_. Defaults to True.
            use_proto (bool, optional): _description_. Defaults to False.
        """
        super(MMTSC, self).__init__()
        self.nclass = nclass 
        self.dropout = dropout
        self.in_channels = in_channels
        self.len_ts = len_ts
        self.use_proto = use_proto
        self.use_attn = use_attn
        self.use_lstm = use_lstm

        self.fusion_module = FusionModule(in_channels=in_channels, out_channels=32)
        
        channels =  [32 * 2 ** i for i in range(pyramid_layer_num + 1)]
               
        self.pyramid_layer_list = nn.ModuleList()
        for pl in range(pyramid_layer_num):
            self.pyramid_layer_list.add_module(f"pyramid_layer{pl}", ScaleLayer(in_channels=channels[pl], out_channels=channels[pl+1], lstm_hidden_size=embed_dim, lstm_layers=1))
        
                
        self.embed_attn = EmbedAttn(embed_dim=embed_dim*2, hidden_dim=embed_dim*2)
        
        
        # todo 2 can be replaced by the bidirection
        fc_layers = [(pyramid_layer_num +1) * 2 * embed_dim  ] + fc_layers
        if not use_proto:
            fc_layers.append(nclass)  

        self.mlp = nn.Sequential()
        
        for i in range(len(fc_layers)-1):
            self.mlp.add_module(f"linear{i}", nn.Linear(fc_layers[i], fc_layers[i+1]))
            self.mlp.add_module(f"batch{i}", nn.LazyBatchNorm1d())
            self.mlp.add_module(f"act{i}", nn.LeakyReLU())
            

        if self.use_proto:
            proto_dim = 128
            self.proto_linear1 = nn.Linear(fc_layers[-1], proto_dim * nclass)
            self.attn_act = nn.Tanh()
            self.proto_linear2 = nn.Linear(proto_dim, nclass)
            
        
    def forward(self, x):
        
        fuse_x = self.fusion_module(x)
        
        lstm_scale = []
        plx = fuse_x
        for pyramid_layer in self.pyramid_layer_list:
            plx, lstmy = pyramid_layer(plx)
            lstm_scale.append(lstmy)
        
        plx = reduce(plx, "b c l -> b c", "mean")
        embed_x = rearrange([plx] + lstm_scale, "n b c -> b n c") 
        embed_x = self.embed_attn(embed_x)
        
        embed_x = rearrange(embed_x, "b c l -> b (c l)")
        
        return self.mlp(embed_x) 
        
        
        
        
        
        
        
        
        return