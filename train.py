import os,sys
sys.path.append("../")
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import pandas as pd
from einops import rearrange, reduce, repeat
from utils.dataset import UEADataset , MyOnehot
from torch.utils.data import DataLoader

def acc(tensor_pred, tensor_true):
    pred = torch.argmax(tensor_pred, dim=1)
    true = torch.argmax(tensor_true, dim=1)
    return torch.sum(pred==true).item()/len(pred) 

def train_in_dataset(model_name, dataset_name, batch_size=32, lr=1e-4, model_args=None, pre_train_path=None, ):
    train_dataset = UEADataset(dataset_name)
    test_dataset = UEADataset(dataset_name, train=False)
    datashape = train_dataset.data[0].shape
    labels = np.unique(train_dataset.label)
    onehot = MyOnehot(labels)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True)
    
    
    print("-------------Dataset---------------")
    print("Dataset Name: {}".format(dataset_name))
    print("Datashape: {}".format(datashape))
    print("nclass: {}".format(len(labels)))
    print("label names: {}".format(labels))
    print("train data num: {}".format(train_dataset.__len__()))
    print("test data num: {}".format(test_dataset.__len__()))
            
    if model_name == "convtransformer":
        from model.convformer import Convformer
        model = Convformer(dim=datashape[0], nclass=len(labels))
    elif model_name == "swintransformer":
        from model.SwinTranformer1d import SwinTransformer1D
        model = SwinTransformer1D(
            num_classes=len(labels),
            in_chans=datashape[0],
            embed_dim=128,
            depths=[2, 2, 6, 2],
            num_heads=[4, 8, 16, 32],
            window_size=7,
            mlp_ratio=4.,
            qkv_bias=True,
            qk_scale=None,
            drop_rate=0.,
            attn_drop_rate=0.,
            drop_path_rate=0.2,
            norm_layer=nn.LayerNorm,
            patch_norm=False,
            use_checkpoint=False)
    elif model_name == "resnet1d":
        from model.resnetmodel import ResNet_1d
        model = ResNet_1d(in_channels=datashape[0], nclass=len(labels))
    elif model_name == "alibimodel":
        from model.alibimodel import AlibiNet
        model = AlibiNet(transblock_num=5, input_dims=datashape[0], nclass=len(labels))
    print("model name:{}".format(model.__class__.__name__))
    model = model.cuda()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    

if __name__ == "__main__":
    pass