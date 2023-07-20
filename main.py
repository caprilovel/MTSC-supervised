#--------------------------------------#

#   显卡设置

#--------------------------------------#
# 显卡配置应该在导入torch之前，否则失效
import os
print("-------------GPUs Distribution---------------")
from global_utils.find_gpus import find_gpus
os.environ['CUDA_VISIBLE_DEVICES'] = find_gpus()
import warnings
warnings.filterwarnings('ignore')

#--------------------------------------#

#   命令行参数设置

#--------------------------------------#
from utils.Args import TorchArgs 
args = TorchArgs()


args.add_argument(
    '--model_index', type=int, default=0, help="model index"
)
args.add_argument(
    '--dataset_index', type=int, default=0, help="dataset index"
)
args.add_argument(
    '--patch_size', type=int, default=4, help="patch size"
)

dict_args = vars(args.parse_args())
#--------------------------------------#

#   日志设置

#--------------------------------------#
from global_utils.log_utils import get_time_str,mkdir,Logger
import sys

models_name = ["convtransformer", "swintransformer", "resnet1d", "alibimodel"]
model_name = models_name[dict_args["model_index"]]
timestamp = str(model_name) + get_time_str() 
mkdir('./log/{}'.format(timestamp))

log_path = './log/{}/log.log'.format(timestamp)
using_log = True
if using_log:
    sys.stdout = logger = Logger(log_path)
print("-------------Generate Log File---------------")
print('log file dir:./log/')
print('log: {}'.format(log_path))
for k,v in dict_args.items():
    print("{}: {}".format(k,v))



    
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import pandas as pd
from einops import rearrange, reduce, repeat


#--------------------------------------#

#   数据设置

#--------------------------------------#
from utils.dataset import UEADataset, load_data, MyOnehot
from global_utils.format_utils import yaml_convert_config
from torch.utils.data import DataLoader
dataset_path = './data/raw/' 
dataset_names = yaml_convert_config('./data/dataset.yml')
el_dataset = dataset_names['equallength']
dataset_name = el_dataset[dict_args['dataset_index']]

train_dataset = UEADataset(dataset_name)
test_dataset = UEADataset(dataset_name, train=False)

datashape = train_dataset.data[0].shape
labels = np.unique(train_dataset.label)
one_hot = MyOnehot(labels)
batch_size = dict_args['batch_size']
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True)

print("-------------Dataset---------------")
print("Dataset Name: {}".format(dataset_name))
print("Datashape: {}".format(datashape))
print("nclass: {}".format(len(labels)))
print("label names: {}".format(labels))
print("train data num: {}".format(train_dataset.__len__()))
print("test data num: {}".format(test_dataset.__len__()))

#--------------------------------------#

#   模型设置

#--------------------------------------#
if model_name == "convtransformer":
    from model.convformer import Convformer
    model = Convformer(dim=datashape[0], nclass=len(labels))
elif model_name == "swintransformer":
    from model.SwinTranformer1d import SwinTransformer1D
    model = SwinTransformer1D(
        num_classes=len(labels),
        patch_size=dict_args['patch_size'],
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
optimizer = optim.Adam(model.parameters(), lr=dict_args['lr'])


model_data_path = './model_data/{}/'.format(model.__class__.__name__ + '_' + timestamp)
mkdir(model_data_path)



from tqdm import tqdm
def acc(tensor_pred, tensor_true):
    pred = torch.argmax(tensor_pred, dim=1)
    true = torch.argmax(tensor_true, dim=1)
    return torch.sum(pred==true).item()/len(pred)
def train():
    train_acc = []
    test_acc = []
    train_loss_list = []
    test_loss_list = []
    max_train_acc = 0.0
    max_test_acc = 0.0
    for epoch in range(dict_args['epochs']):
        model.train()
        with tqdm(total=(train_dataset.__len__()-1)//batch_size+1,bar_format='{desc} {n_fmt:>4s}/{total_fmt:<4s} {percentage:3.0f}%|{bar}| {postfix}',) as t:
            epoch_loss_list = []
            true_label = []
            pred_label = []
            for _, (data, label) in enumerate(train_loader):
                label = one_hot.transform(label)
                data = torch.FloatTensor(data).cuda()
                label = torch.FloatTensor(label).cuda()
                optimizer.zero_grad()
                out = model(data)
                loss = F.cross_entropy(out, label.argmax(dim=1))
                epoch_loss_list.append(loss.item())
                loss.backward()
                optimizer.step()
                true_label.append(label.cpu())
                pred_label.append(out.cpu())
                
    
            
                t.update(1)
                t.set_description_str(f"\33[36m【Train Epoch {epoch + 1:03d}】")
                t.set_postfix_str(f"epoch_train_loss={epoch_loss_list[-1] / batch_size:.4f} ")
            # print("train acc = ",acc(torch.cat(pred_label, dim=0), torch.cat(true_label,dim=0)))
            train_acc.append(acc(torch.cat(pred_label, dim=0), torch.cat(true_label,dim=0)))
        if epoch % 10 == 0:
            torch.save(model.state_dict(), os.path.join(model_data_path, 'model_{}.pth'.format(epoch)))
        
        model.eval()
        with torch.no_grad():
            with tqdm(total=(test_dataset.__len__()-1)//batch_size+1,bar_format='{desc} {n_fmt:>4s}/{total_fmt:<4s} {percentage:3.0f}%|{bar}| {postfix}',) as t:
                epoch_loss_list = []
                true_label = []
                pred_label = []
                for _, (data, label) in enumerate(test_loader):
                    label = one_hot.transform(label)
                    data = torch.FloatTensor(data).cuda()
                    label = torch.FloatTensor(label).cuda()
                    
                    out = model(data)
                    
                    loss = F.cross_entropy(out, label.argmax(dim=1))
                    epoch_loss_list.append(loss.item())
                    true_label.append(label.cpu())
                    pred_label.append(out.cpu())
                    
        
                
                    t.update(1)
                    t.set_description_str(f"\33[36m【Test Epoch {epoch + 1:03d}】")
                    t.set_postfix_str(f"epoch_test_loss={epoch_loss_list[-1] / batch_size:.4f}")
                # print("test acc = ",acc(torch.cat(pred_label, dim=0), torch.cat(true_label,dim=0)))
                test_acc.append(acc(torch.cat(pred_label, dim=0), torch.cat(true_label,dim=0)))
                max_train_acc = max(max_train_acc, train_acc[-1])
                max_test_acc = max(max_test_acc, test_acc[-1])
                
    print("max train acc = ",max_train_acc)
    print("max test acc = ",max_test_acc)
    
    def plot_and_save(data, file_path):
        import matplotlib.pyplot as plt
        plt.plot(data)
        plt.savefig(file_path)
    # stamp = get_time_str()
    plot_path = f'./log/{timestamp}/plot/'
    mkdir(f'./log/{timestamp}/plot/')
    plot_and_save(train_acc, plot_path + dataset_name + 'train_acc.png')
    plot_and_save(test_acc, plot_path+ dataset_name + 'test_acc.png')
train()


