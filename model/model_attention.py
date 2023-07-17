import torch
import torch.nn as nn
import torch.nn.functional as F
from utils import euclidean_dist, normalize, output_conv_size, dump_embedding
import numpy as np
import math



class TapNet(nn.Module):

    def __init__(self, nfeat, len_ts, nclass, dropout, filters, kernels, dilation, layers, use_rp, rp_params,
                 use_att=True, use_metric=False, use_lstm=False, use_cnn=True, use_mhsa=False, lstm_dim=128):
        super(TapNet, self).__init__()
        self.nclass = nclass
        self.dropout = dropout
        self.use_metric = use_metric
        self.use_lstm = use_lstm
        self.use_cnn = use_cnn
        self.use_multiheadselfatt = use_mhsa
        # parameters for random projection
        self.use_rp = use_rp
        self.rp_group, self.rp_dim = rp_params
        self.use_transformer = True
        self.bidiraction = True
        if True:
            # LSTM
            
            self.channel = nfeat
            self.ts_length = len_ts
            if self.use_lstm:
                self.lstm_dim = lstm_dim
                self.lstm = nn.LSTM(self.ts_length, self.lstm_dim, bidirectional=self.bidiraction)
            if self.use_transformer:
                self.transformer_layer = nn.TransformerEncoderLayer(d_model=self.ts_length, nhead=5)
                self.transformer_encoder = nn.TransformerEncoder(self.transformer_layer, num_layers=2)


            paddings = [0, 0, 0]
            if self.use_rp:
                self.conv_1_models = nn.ModuleList()
                self.idx = []
                for i in range(self.rp_group):
                    self.conv_1_models.append(nn.Conv1d(self.rp_dim, filters[0], kernel_size=kernels[0], dilation=dilation, stride=1, padding=paddings[0]))
                    self.idx.append(np.random.permutation(nfeat)[0: self.rp_dim])
            else:
                self.conv_1 = nn.Conv1d(self.channel, filters[0], kernel_size=kernels[0], dilation=dilation, stride=1, padding=paddings[0])

            self.conv_bn_1 = nn.BatchNorm1d(filters[0])

            self.conv_2 = nn.Conv1d(filters[0], filters[1], kernel_size=kernels[1], stride=1, padding=paddings[1])

            self.conv_bn_2 = nn.BatchNorm1d(filters[1])

            self.conv_3 = nn.Conv1d(filters[1], filters[2], kernel_size=kernels[2], stride=1, padding=paddings[2])

            self.conv_bn_3 = nn.BatchNorm1d(filters[2])

            # self.transformer = nn.TransformerEncoder(self.ts_length,nhead = 4,dim_forward=256)
            # compute the size of input for fully connected layers
            fc_input = 0
            if self.bidiraction:
                self.lstm_dim = 2 * self.lstm_dim
            if self.use_cnn:
                conv_size = len_ts
                for i in range(len(filters)):
                    conv_size = output_conv_size(conv_size, kernels[i], 1, paddings[i])
                fc_input += conv_size 
                #* filters[-1]
            if self.use_lstm:
                fc_input += conv_size * self.lstm_dim
            
            if self.use_rp:
                if self.use_lstm:
                    fc_input = self.rp_group * filters[2] + self.lstm_dim
                else:
                    fc_input = self.rp_group * filters[2]
            # use transformer
            # fc_input += self.ts_length

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

        
    def forward(self, input):
        x, labels, idx_train, idx_val, idx_test = input  # x is N * L, where L is the time-series feature dimension

        if True:
            N = x.size(0)

            if self.use_transformer:
                x_transformer = self.transformer_encoder(x)
                # x_transformer = x_transformer.mean(1)
                # x_transformer = x_transformer.view(N, -1)

            # LSTM
            if self.use_lstm:
                x_lstm = self.lstm(x)[0]
                x_lstm = x_lstm.mean(1)
                x_lstm = x_lstm.view(N, -1)

            
            if self.use_cnn:
                # Convolutional Network
                # input ts: # N * C * L
                if self.use_rp:
                    for i in range(len(self.conv_1_models)):
                        #x_conv = x
                        x_conv = self.conv_1_models[i](x[:, self.idx[i], :])
                        x_conv = self.conv_bn_1(x_conv)
                        x_conv = F.leaky_relu(x_conv)

                        x_conv = self.conv_2(x_conv)
                        x_conv = self.conv_bn_2(x_conv)
                        x_conv = F.leaky_relu(x_conv)

                        x_conv = self.conv_3(x_conv)
                        x_conv = self.conv_bn_3(x_conv)
                        x_conv = F.leaky_relu(x_conv)

                        x_conv = torch.mean(x_conv, 2)

                        if i == 0:
                            x_conv_sum = x_conv
                        else:
                            x_conv_sum = torch.cat([x_conv_sum, x_conv], dim=1)

                    x_conv = x_conv_sum
                else:
                    x_conv = x
                    x_conv = self.conv_1(x_conv)  # N * C * L
                    x_conv = self.conv_bn_1(x_conv)
                    x_conv = F.leaky_relu(x_conv)

                    x_conv = self.conv_2(x_conv)
                    x_conv = self.conv_bn_2(x_conv)
                    x_conv = F.leaky_relu(x_conv)

                    x_conv = self.conv_3(x_conv)
                    x_conv = self.conv_bn_3(x_conv)
                    x_conv = F.leaky_relu(x_conv)

                    x_conv = x_conv.view(N, -1)

            if self.use_lstm and self.use_cnn:
                x = torch.cat([x_conv, x_lstm], dim=1)
            elif self.use_lstm:
                x = x_lstm
            elif self.use_cnn:
                x = x_conv
            #
            # if self.use_transformer:
            #     x = torch.cat([x, x_transformer], dim=1)
        # linear mapping to low-dimensional space
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


