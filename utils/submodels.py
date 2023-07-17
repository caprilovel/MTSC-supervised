from turtle import forward
import torch 
import torch.nn as nn 
import torch.nn.functional as F 
import numpy as np


#-----------------------------------------------------------#

#   TemporalAttentionModule, 复现论文"A New Attention Mechanism to Classify Multivariate Time Series"模型

#-----------------------------------------------------------#


class TemporalAttention(nn.Module):
    def __init__(self, length, in_channels, qk_channels, v_channels, attention_type='multipy'):
        super().__init__()
        self.length = length
        self.key_conv = nn.Conv1d(in_channels, qk_channels, 1)
        self.query_conv = nn.Conv1d(in_channels, qk_channels, 1)
        self.value_conv = nn.Conv1d(in_channels, v_channels, 1)

        self.out_conv = nn.Conv1d(v_channels, in_channels, 1)
        self.attention_type = attention_type


    def forward(self, x, mask=False):
        #  data shape
        #  key: batch x length x qk_channels
        #  query: batch x qk_channels x length
        #  value: batch x length x v_channels

        key = torch.transpose(self.key_conv(x), 1, 2)
        query = self.query_conv(x)
        value = torch.transpose(self.value_conv(x), 1, 2)
        # attn_map: batch x key x query
        if self.attention_type is 'multipy':
            attn_map =  torch.bmm(key, query)
        attn_mask = torch.triu(torch.ones(self.length, self.length), diagonal=0)
        if mask:
            attn_map = attn_map * attn_mask
        attn_score = F.softmax(attn_map, dim=2)
        attn_value = torch.bmm(attn_score, value) 
        attn_value = torch.transpose(attn_value, 1, 2)
        return self.out_conv(attn_value)


import torch.nn as nn
import torch.nn.functional as F
import torch
import numpy as np
import os
import random
from utils.utils import *


class MultiScaleConv1d(nn.Module):
    def __init__(self, in_channels, out_channels, input_scales = [3,5,7,11,13]):
        super().__init__()
        self.conv_array = []
        for i in input_scales:
            self.conv_array.append(nn.Conv1d(in_channels, out_channels,kernel_size=i, padding=(i-1)//2))
    def forward(self, x):
        y = []
        for conv in self.conv_array:
            y.append(conv(x))
        y_sum = torch.cat(y, dim=1)
        return y_sum    


#-----------------------------------------------------------#

#   TemporalAttentionModule, 复现论文"A New Attention Mechanism to Classify Multivariate Time Series"模型

#-----------------------------------------------------------#


class TemporalAttention(nn.Module):
    def __init__(self, length, in_channels, qk_channels, v_channels):
        super().__init__()
        self.length = length
        self.key_conv = nn.Conv1d(in_channels, qk_channels, 1)
        self.query_conv = nn.Conv1d(in_channels, qk_channels, 1)
        self.value_conv = nn.Conv1d(in_channels, v_channels, 1)

        self.out_conv = nn.Conv1d(v_channels, in_channels, 1)



    def forward(self, x, mask=False):
        #  data shape
        #  key: batch x length x qk_channels
        #  query: batch x qk_channels x length
        #  value: batch x length x v_channels

        key = torch.transpose(self.key_conv(x), 1, 2)
        query = self.query_conv(x)
        value = torch.transpose(self.value_conv(x), 1, 2)
        # attn_map: batch x key x query
        attn_map =  torch.bmm(key, query)
        # 构建上三角矩阵
        attn_mask = torch.triu(torch.ones(self.length, self.length), diagonal=0)
        if mask:
            attn_map = attn_map * attn_mask
        attn_score = F.softmax(attn_map, dim=2)
        attn_value = torch.bmm(attn_score, value) 
        attn_value = torch.transpose(attn_value, 1, 2)
        return self.out_conv(attn_value)

class PyramidLayer(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=[3,3,3], hidden_size=256, use_rnn=True, use_SE=True):
        super().__init__()
        paddings = (np.array(kernel_size)-1)//2
        
        self.use_rnn = use_rnn


        self.res1 = nn.Conv1d(in_channels, in_channels, kernel_size=kernel_size[0], padding=paddings[0])
        self.res2 = nn.Conv1d(in_channels, in_channels, kernel_size=kernel_size[1], padding=paddings[1])
        
        if use_rnn:
            self.lstm = nn.LSTM(input_size=in_channels, hidden_size=hidden_size, batch_first=True)

        if use_SE:
            pass
        self.downsampleLayer = nn.Sequential()
        self.downsampleLayer.add_module('downcnn', nn.Conv1d(in_channels, out_channels, kernel_size=3, stride=2, padding=1))
        self.downsampleLayer.add_module('batchnorm', nn.BatchNorm1d(out_channels))
        self.downsampleLayer.add_module('relu', nn.LeakyReLU())
        


    def forward(self, x):
        N = x.size(0)
        resx = self.res1(x)
        resy = self.res2(resx)
        resy += x
        if self.use_rnn:
            lstmx = x.permute(0, 2, 1)
            lstmy, _ = self.lstm(lstmx)
            return self.downsampleLayer(resy), lstmy.mean(1).view(N, -1) 
        else:
            return self.downsampleLayer(resy)




class LinearAttn(nn.Module):
    def __init__(self, input_size, hidden_size, batch_first=False):
        super().__init__()
        self.linear = nn.Linear(input_size, hidden_size)
        self.avgpool = nn.AdaptiveAvgPool1d(1)
        self.batch_first = batch_first
    def forward(self, x):
        # batch first:x size: BxLxD (Batch, Length,  Dimension)
        # batch first False: LxBxD
        if self.batch_first:
            x_trans = x
        else:
            x_trans = x.permute([1,0,2])        
        y = self.linear(x_trans)
        y = F.relu(y)
        y = self.avgpool(y)
        y = y.squeeze(2)
        att_score = F.softmax(y, dim=1)
        att_score = att_score.unsqueeze(2)
        att_score = att_score.expand_as(x_trans)
        att_score = att_score.permute([0,2,1])
        if not self.batch_first:
            att_score = att_score.permute([1,0,2])
        return att_score
        
class LSTMAtt(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers = 1, bias=True, bidirectional=False, batch_first=False):
        super().__init__()
        self.lstm = nn.LSTM(input_size=input_size, hidden_size=hidden_size, num_layers=num_layers, bias=bias, batch_first=batch_first, bidirectional=bidirectional)
        #
        self.attn_module = LinearAttn(hidden_size, hidden_size, batch_first=batch_first)
        
    def forward(self, x):
        y,(h_0,c_0) = self.lstm(x)
        attn_score = self.attn_module(y)
        
        return y*attn_score 


class OmniScale(nn.Module):
    def __init__(self, in_channels, out_channels, input_scales = [3,5,7,11,13]):
        super().__init__()
        self.conv_array = []
        for i in input_scales:
            conv = nn.Sequential()
            conv.add_module('padding{}'.format(i), nn.ConstantPad1d((i//2, (i-1)//2), 0))
            conv.add_module('conv_kernel{}'.format(i), nn.Conv1d(in_channels, out_channels, kernel_size=i))
            self.conv_array.append(conv) 
    def forward(self, x):
        y = []
        for conv in self.conv_array:
            y.append(conv(x))
        y_sum = torch.cat(y, dim=1)
        return y_sum    
        

class SE_Block(nn.Module):
    def __init__(self, ch_in, reduction=3):
        super(SE_Block, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Sequential(
            nn.Linear(ch_in, ch_in // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(ch_in // reduction, ch_in, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _= x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1)
        return x * y.expand_as(x)

class ResBlock(nn.Module):
    def __init__(self, input_channels, num_channels, strides=[1,1], kernel_size=[3,3], use_con1x1=False):
        super().__init__()
        # Model 
        self.conv1 = nn.Conv1d(input_channels, num_channels, kernel_size=kernel_size[0], padding=1, stride=strides[0])
        self.conv2 = nn.Conv1d(num_channels, num_channels, kernel_size=kernel_size[1], padding=1, stride=strides[1])
        self.bn1 = nn.BatchNorm1d(num_channels)
        self.bn2 = nn.BatchNorm1d(num_channels)
        
        if use_con1x1:
            self.conv3 = nn.Conv1d(input_channels, num_channels, kernel_size=1, stride=strides)
        else:
            self.conv3 = None
        self.relu = nn.ReLU(inplace=True)
        self.se_block = SE_Block(input_channels)
        
        
    def forward(self, X):
        
        Y = F.relu(self.bn1(self.conv1(X)))
        Y = self.bn2(self.conv2(Y))
        if self.conv3:
            X = self.conv3(X)
        # Y = self.se_block(Y)
        Y += X
        return F.relu(Y)
    
class BOPLayer(nn.Module):
    def __init__(self, input, filters = [256, 128, 64]):
        super().__init__()
        self.fc1 = nn.Linear(in_features=input, out_features=filters[0])
        self.relu = nn.ReLU(inplace=True)
        self.fc2 = nn.Linear(in_features=filters[0], out_features= filters[1])
        self.fc3 = nn.Linear(filters[1], filters[2])
        
    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        return x
         
        
    
    
class TransBlcok(nn.Module):
    def __init__(self, len_ts, lstm_dim, input_channels, output_channels,\
        bidirectional=True, use_lstm=True):
        super(TransBlcok, self).__init__()
        # Model Parameters
        self.ts_length = len_ts
        self.lstm_dim = lstm_dim
        self.bidirectional = bidirectional
        self.use_lstm = use_lstm
        self.input_channels = input_channels
        self.output_channels = output_channels
        # Model Frame
        if self.use_lstm:
            self.rnn = nn.LSTM(self.ts_length, self.lstm_dim, bidirectional=self.bidirectional)
        self.residual = ResBlock(input_channels, input_channels)
        
    def forward(self, x):
        N = x.size(0)
        if self.use_lstm:
            x_lstm = self.rnn(x)[0]
            x_lstm = x_lstm.mean(1)
            x_lstm = x_lstm.view(N, -1)
            output_lstm = x_lstm
        output_residual = self.residual(x)
        return output_lstm, output_residual

class DownBlock(nn.Module):
    def __init__(self, input_channels, output_channels):
        super().__init__()
        self.cnn = nn.Conv1d(input_channels, output_channels, stride=2, kernel_size=3, padding=0)
        self.bn = nn.BatchNorm1d(output_channels)
    def forward(self, X):
        Y = self.cnn(X)
        Y = self.bn(Y)
        return Y
    
class DiffCNN(nn.Module):
    def __init__(self, input_channels, output_channels):
        super(DiffCNN, self).__init__()
        self.cnn1 = nn.Conv1d(in_channels=input_channels,out_channels=output_channels,kernel_size=7,padding=3)
        self.cnn2 = nn.Conv1d(in_channels=output_channels, out_channels=output_channels, kernel_size=5, padding=2)
        self.cnn3 = nn.Conv1d(in_channels=output_channels ,out_channels=output_channels, kernel_size=3, padding=1)

    def forward(self,x):
        x = self.cnn1(x)
        x = self.cnn2(x)
        x = self.cnn3(x)
        return x  
    
class TransNet(nn.Module):

    def __init__(self, nfeat, len_ts, nclass, dropout, filters, kernels, dilation, layers, weasel_dim, use_att=True, use_lstm=False, use_transformer=True):
        super(TransNet, self).__init__()
        # Model Parameters
        self.nclass = nclass
        self.dropout = dropout
        self.use_lstm = use_lstm
        self.use_transformer = use_transformer 
        self.bidiractional = True
        self.channel = nfeat
        self.ts_length = len_ts
        paddings = [0, 0, 0]
        lstm_dims = [256, 256, 256]
        dilations = [1, 1, 1]
        self.filters = [128, 256, 512]
        self.bn = nn.BatchNorm1d(self.channel)
        self.ts_length2 = output_conv_size(self.ts_length, kernel_size=kernels[0], stride=2, padding=paddings[0], dilation=dilations[0])
        self.ts_length3 = output_conv_size(self.ts_length2, kernel_size=kernels[1], stride=2, padding=paddings[1], dilation=dilations[1])
        self.channel_interplay = nn.Conv1d(self.channel, self.channel, kernel_size=1)
        self.bopLayer = BOPLayer(weasel_dim, filters=[1024,512,512])
        
        self.diff = DiffCNN(self.channel-1, 256)
        self.mh = 0        
        if self.ts_length % 5 == 0:
            self.mh = 5
        elif self.ts_length % 4 == 0:
            self.mh = 4
        elif self.ts_length % 3 == 0:
            self.mh = 3
        elif self.ts_length % 2 ==0:
            self.mh = 2
        
        self.transformerlayer = nn.TransformerEncoderLayer(d_model=self.ts_length, nhead=self.mh, dim_feedforward=256)
        self.transformerblock = nn.TransformerEncoder(self.transformerlayer, 2)
        
        # Model Frame
        if True:
            self.cnn = nn.Conv1d(self.channel * 2, self.filters[0], kernel_size=7, padding=3)
            # Layer 1
            self.trans1 = TransBlcok(self.ts_length, lstm_dim=lstm_dims[0], input_channels=self.filters[0], \
                output_channels=self.filters[0])
            self.db1 = DownBlock(self.filters[0], self.filters[1])
            
            # Layer2
            self.trans2 = TransBlcok(self.ts_length2, lstm_dim=lstm_dims[1], input_channels=self.filters[1],\
                output_channels=self.filters[1])
            self.db2 = DownBlock(self.filters[1], self.filters[2])

            # Layer3 
            self.trans3 = TransBlcok(self.ts_length3, lstm_dim=lstm_dims[2], input_channels=self.filters[2],\
                output_channels=self.filters[2])
            
            
 
            
            # FC Layers
            fc_input = 0
            
            for i in range(3):
                if True:
                    lstm_dim = 2 * lstm_dims[i]
                    fc_input += lstm_dim     
            
            fc_input += self.filters[2]
            fc_input += 256
            fc_input += 512


        # Representation mapping function
        layers = [fc_input] + layers
        print("Layers", layers)
        self.mapping = nn.Sequential()
        for i in range(len(layers) - 2):
            self.mapping.add_module("fc_" + str(i), nn.Linear(layers[i], layers[i + 1]))
            self.mapping.add_module("bn_" + str(i), nn.BatchNorm1d(layers[i + 1]))
            self.mapping.add_module("relu_" + str(i), nn.LeakyReLU())

        # add last layer
        self.mapping.add_module("fc_" + str(len(layers) - 2), nn.Linear(layers[-2], layers[-1]))
        if len(layers) == 2:  # if only one layer, add batch normalization
            self.mapping.add_module("bn_" + str(len(layers) - 2), nn.BatchNorm1d(layers[-1]))

        # Attention
        att_dim, semi_att_dim = 128, 128
        self.use_att = use_att
        if self.use_att:
            self.att_models = nn.ModuleList()
            for _ in range(nclass):

                att_model = nn.Sequential(
                    nn.Linear(layers[-1], att_dim),
                    nn.Tanh(),
                    nn.Linear(att_dim, 1)
                )
                self.att_models.append(att_model)

        
    def forward(self, input, xweasel):
        x, labels, idx_train, idx_val, idx_test = input  # x is N * L, where L is the time-series feature dimension

        
        y = self.bn(x)
        z = y[:,1:,:] - y[:,:-1,:]
        
        z = self.diff(z)
        z = z.mean(2)
        # x = self.transformerblock(x)
        
        interplay = self.channel_interplay(x)
        x = torch.cat([x, interplay], dim=1)
        x_weasel = self.bopLayer(xweasel)
        
        if True:
            N = x.size(0)
            
            x = self.cnn(x)
            # Layer1
            lstm_x1, residual_x = self.trans1(x)
            residual_x = self.db1(residual_x)
            
            lstm_x2, residual_x = self.trans2(residual_x)
            residual_x = self.db2(residual_x)
            
            lstm_x3, residual_x = self.trans3(residual_x)
            
            residual_x = residual_x.mean(2)
            residual_x = residual_x.view(N, -1)
        # linear mapping to low-dimensional space
        x = torch.cat([lstm_x1, lstm_x2, lstm_x3, residual_x], dim=1)
        x = torch.cat([x, z], dim=1)
        x = torch.cat([x, x_weasel], dim=1)
        x = self.mapping(x)
        
        # generate the class protocal with dimension C * D (nclass * dim)
        proto_list = []
        for i in range(self.nclass):
            idx = (labels[idx_train].squeeze() == i).nonzero().squeeze(1)
            if self.use_att:
                A = self.att_models[i](x[idx_train][idx])  # N_k * 1
                A = torch.transpose(A, 1, 0)  # 1 * N_k
                A = F.softmax(A, dim=1)  # softmax over N_k

                class_repr = torch.mm(A, x[idx_train][idx]) # 1 * L
                class_repr = torch.transpose(class_repr, 1, 0)  # L * 1
            else:  # if do not use attention, simply use the mean of training samples with the same labels.
                class_repr = x[idx_train][idx].mean(0)  # L * 1
            proto_list.append(class_repr.view(1, -1))
        x_proto = torch.cat(proto_list, dim=0)

        # prototype distance
        proto_dists = euclidean_dist(x_proto, x_proto)
        proto_dists = torch.exp(-0.5*proto_dists)
        num_proto_pairs = int(self.nclass * (self.nclass - 1) / 2)
        proto_dist = torch.sum(proto_dists) / num_proto_pairs

        dists = euclidean_dist(x, x_proto)

        dump_embedding(x_proto, x, labels)
        return torch.exp(-0.5*dists), proto_dist


