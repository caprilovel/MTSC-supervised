import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
from einops import rearrange, reduce, repeat
import matplotlib.pyplot as plt


def accuracy(output, labels):
    """accuracy calculate the accuracy

    Used for the calculation of the accuracy

    Args:
        output (torch.Tensor): the prediction vector, size: batch * num_class.
        labels (torch.Tensor): the True label of the input, size: batch * 1.

    Returns:
        numpy.float64: the accuracy of this batch.
    """
    preds = output.max(1)[1].cpu().numpy()
    labels = labels.cpu().numpy()
    acc_score = (accuracy_score(labels, preds))

    return acc_score

def f1score(output, labels):
    """f1score calculate the f1 score

    Calculate the f1 score for the raw output.

    Args:
        output (torch.Tensor): the prediction vector, size: batch * num_class.
        labels (torch.Tensor): the True label of the input, size: batch * 1.

    Returns:
        numpy.float64: the f1 score of this batch.
    """
    preds = output.max(1)[1].cpu().numpy()
    labels = labels.cpu().numpy()
    f1score = (f1_score(labels, preds))

    return f1score

def sconfusion_matrix(output, labels):
    """sconfusion_matrix _summary_

    Calculate the confuse matrix for the raw output.

    Args:
        output (torch.Tensor): the prediction vector, size: batch * num_class.
        labels (torch.Tensor): the True label of the input, size: batch * 1.

    Returns:
        numpy.ndarray: the confuse matrix of this batch.
    """
    preds = output.max(1)[1].cpu().numpy()
    labels = labels.cpu().numpy()
    confuse_matrix = (confusion_matrix(labels, preds))

    return confuse_matrix

def predict(output):
    """predict A simple implemrntation of prediction vetor to label.

    Used for the convertion from prediction vector to label. Only can be used for tensor-form data
    
    Args:
        output (torch.Tensor): prediction vector, size: batch * num_class

    Returns:
        numpy.ndarray: label vector, size: batch * 1 
    """
    return output.max(-1)[1].cpu().numpy()


def save_cm_matrix(cm, path, classes, title, cmap=plt.cm.Blues, color_bar=False):
    """save_cm_matrix plot and save the confusion matrix graph

    plot and save the confusion matrix.

    Args:
        cm (np.ndarray): confusion matrix in array form. 
        path (str): the path to save the confusion matrix image.
        classes (List): the list of the names of each class.
        title (str): the title of the graph 
        cmap (_type_, optional): the color mode of the displayed image. Defaults to plt.cm.Blues.
        color_bar (bool, optional): whether to use the color bar . Defaults to False.
    """
    import matplotlib.pyplot as plt
    from sklearn.metrics import confusion_matrix
    plt.rc('font', family='Times New Roman', size='8')
    
    cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis] # 归一化
    str_cm = cm.astype(np.str).tolist()
    
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            if int(cm[i, j]*100 + 0.5) == 0:
                cm[i, j] = 0
                
    fig, ax = plt.subplots()
    im = ax.imshow(cm, interpolation='nearest', cmap=cmap)
    ax.figure.colorbar(im, ax=ax)
    
    ax.set(xticks=np.arange(cm.shape[1]),
           yticks=np.arange(cm.shape[0]),
           xticklabels=classes, yticklabels=classes,
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
    plt.savefig(path, dpi=300)
    
    plt.show()
    
    
    # plt.colorbar()
    