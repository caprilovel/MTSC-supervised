from __future__ import division
import numpy as np
import scipy.sparse as sp
import sklearn
import sklearn.metrics
import torch
import pandas as pd
import random
import os
import numpy as np
import random
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import os
import torch 
from sklearn.metrics import accuracy_score, f1_score
import sys
import time 
import re
#-----------------------------------------------------------#

#   label_select,用于样本不均衡的数据集，每一次的输入样本为每一类相同数目的样本

#-----------------------------------------------------------#


def label_select(labels, sampling):
    '''
    this function is used for selecting the same amount train samples in each class

    input:
    labels: List, the label list
    samling: the size of the samples in each class
    
    output:
     the list of initialize train dataset index
    
    '''
    if type(labels) is list:
        labels = np.array(labels)
    classes = np.unique(labels)
    # n_class = len(classes)
    class_dict = {}
    sample_list = []   
    for i in classes:
        class_dict[i] = [j for j,x in enumerate(labels) if x==i]
        np.random.shuffle(class_dict[i])
        sample_list.append(class_dict[i][0:sampling])
    return np.concatenate(np.array(sample_list))

# def predict (output):
#     '''
#     输入为batch * 预测向量,返回batch * 预测结果
#     '''
#     return output.max(1)[1].cpu().numpy()

def save_confusion_matrix(cm, path, title=None, labels_name=None,   cmap=plt.cm.Blues):

    
    plt.rc('font',family='Times New Roman',size='8')   # 设置字体样式、大小
    # plt.colorbar()
    # 按行进行归一化
    cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    str_cm = cm.astype(np.str).tolist()
    for row in str_cm:
        print('\t'.join(row))
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            if int(cm[i, j]*100 + 0.5) == 0:
                cm[i, j]=0

    fig, ax = plt.subplots()
    im = ax.imshow(cm, interpolation='nearest', cmap=cmap)
    # ax.figure.colorbar(im, ax=ax) # 侧边的颜色条带
    
    ax.set(xticks=np.arange(cm.shape[1]),
           yticks=np.arange(cm.shape[0]),
           xticklabels=labels_name, yticklabels=labels_name,
           title=title,
           ylabel='Actual',
           xlabel='Predicted')

    # 通过绘制格网，模拟每个单元格的边框
    ax.set_xticks(np.arange(cm.shape[1]+1)-.5, minor=True)
    ax.set_yticks(np.arange(cm.shape[0]+1)-.5, minor=True)
    ax.grid(which="minor", color="gray", linestyle='-', linewidth=0.2)
    ax.tick_params(which="minor", bottom=False, left=False)

    # 将x轴上的lables旋转45度
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
             rotation_mode="anchor")

    # 标注百分比信息
    fmt = 'd'
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            if int(cm[i, j]*100 + 0.5) > 0:
                ax.text(j, i, format(int(cm[i, j]*100 + 0.5) , fmt) + '%',
                        ha="center", va="center",
                        color="white"  if cm[i, j] > thresh else "black")
    fig.tight_layout()
    plt.savefig(path + 'cm.jpg', dpi=300)


#-----------------------------------------------------------#

#   random_seed，用于torch训练随机种子

#-----------------------------------------------------------#


def random_seed(seed=2020):
    """random_seed Setting a random seed for the whole random functions.

    The random seed setting for the functions above:

    Args:
        seed (int, optional): _description_. Defaults to 2020.
    """
    # determine the random seed
    random.seed(seed)
    # hash 
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True

def accuracy(output, labels):
    pred = output.max(1)[1].cpu().numpy()
    labels = labels.cpu().numpy()
    return accuracy_score(labels, pred)


def mkdir(path):
    folder = os.path.exists(path)
    if not folder:
        os.makedirs(path)


def euclidean_dist(x, y):
    # x: N x D
    # y: M x D
    n = x.size(0)
    m = y.size(0)
    d = x.size(1)
    assert d == y.size(1)

    x = x.unsqueeze(1).expand(n, m, d)
    y = y.unsqueeze(0).expand(n, m, d)

    return torch.pow(x - y, 2).sum(2)

#-----------------------------------------------------------#

#   将时间（秒）转换为 时间（天，小时，分钟，秒）

#   用法示例：day,hour,minute,second = second2time(30804.7)

#   print("time: {:02d}d{:02d}h{:02d}m{:.2f}s".format(day, hour, minute, second))

#   用于显示训练时间

#-----------------------------------------------------------#

def second2time(second):
    intsecond = int(second)
    day = int(second) // (24 * 60 * 60)
    intsecond -= day * (24 * 60 * 60)
    hour = intsecond // (60 * 60)
    intsecond -= hour * (60 * 60)
    minute = intsecond // 60
    intsecond -= 60 * minute
    return (day, hour, minute, second - int(second) + intsecond)


#-----------------------------------------------------------#

#   使用方法：在开始加入此行代码sys.stdout = Logger('log.log')

#-----------------------------------------------------------#


class Logger(object):
    def __init__(self, logFile='./Default.log'):
        self.terminal = sys.stdout
        self.log = open(logFile, 'a')
    
    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)
    
    def flush(self):
        pass

def string2boolean(s):
    tmp_s = s.strip().lower()
    if tmp_s not in {'false', 'true'}:
        return s
    else:
        return s == 'true'    

def boolean_string(s):
    if s not in {'False', 'True'}:
        raise ValueError('Not a valid boolean string')
    return s == 'True'

def encode_onehot(labels):
    classes = set(labels)
    classes_dict = {c: np.identity(len(classes))[i, :] for i, c in
                    enumerate(classes)}
    labels_onehot = np.array(list(map(classes_dict.get, labels)),
                             dtype=np.int32)
    return labels_onehot


def loaddata(filename):
    df = pd.read_csv(filename, header=None, delimiter=",")
    a = np.array(df.as_matrix())
    return a




def normalize(mx):
    """Row-normalize sparse matrix"""
    row_sums = mx.sum(axis=1)
    mx = mx.astype('float32')
    row_sums_inverse = 1 / row_sums
    f = mx.multiply(row_sums_inverse)
    return sp.csr_matrix(f).astype('float32')


# def accuracy(output, labels):
#     preds = output.max(1)[1].cpu().numpy()
#     labels = labels.cpu().numpy()
#     accuracy_score = (sklearn.metrics.accuracy_score(labels, preds))

#     return accuracy_score

# def f1score(output, labels):
#     preds = output.max(1)[1].cpu().numpy()
#     labels = labels.cpu().numpy()
#     f1score = (sklearn.metrics.f1_score(labels, preds))

#     return f1score

# def sconfusion_matrix(output, labels):
#     preds = output.max(1)[1].cpu().numpy()
#     labels = labels.cpu().numpy()
#     f1score = (sklearn.metrics.confusion_matrix(labels, preds))

#     return f1score
def euclidean_dist(x, y):
    # x: N x D
    # y: M x D
    n = x.size(0)
    m = y.size(0)
    d = x.size(1)
    assert d == y.size(1)

    x = x.unsqueeze(1).expand(n, m, d)
    y = y.unsqueeze(0).expand(n, m, d)

    return torch.pow(x - y, 2).sum(2)


def output_conv_size(in_size, kernel_size, stride, padding=0):

    output = int((in_size - kernel_size + 2 * padding) / stride) + 1

    return output

def dump_embedding(proto_embed, sample_embed, labels, dump_file='./plot/embeddings.txt'):
    proto_embed = proto_embed.cpu().detach().numpy()
    sample_embed = sample_embed.cpu().detach().numpy()
    embed = np.concatenate((proto_embed, sample_embed), axis=0)

    nclass = proto_embed.shape[0]
    labels = np.concatenate((np.asarray([i for i in range(nclass)]),
                             labels.squeeze().cpu().detach().numpy()), axis=0)

    with open(dump_file, 'w') as f:
        for i in range(len(embed)):
            label = str(labels[i])
            line = label + "," + ",".join(["%.4f" % j for j in embed[i].tolist()])
            f.write(line + '\n')







def array_fulfill(init_array, length):
    '''fulfill an array
 
    '''
    init_array = np.array(init_array)
    if len(init_array) < length:
        ful_array = np.tile(init_array, length//len(init_array))
        tmp = init_array[0:length%(len(init_array))]
        ful_array = np.concatenate((ful_array, tmp), axis=0)
    else:
        ful_array = init_array
    return ful_array

 