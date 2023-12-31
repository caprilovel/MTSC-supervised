import os 
import numpy as np
import pandas as pds 
import math
import random
from datetime import datetime
import pickle
from scipy.io.arff import loadarff
from sklearn.preprocessing import StandardScaler, MinMaxScaler

def load_UEA_arff(dataset):
    ''' read uea data in 
    '''
    train_data = loadarff(f'datasets/UEA/{dataset}/{dataset}_TRAIN.arff')[0]
    test_data = loadarff(f'datasets/UEA/{dataset}/{dataset}_TEST.arff')[0]
    
    def extract_data(data):
        res_data = []
        res_labels = []
        for t_data, t_label in data:
            t_data = np.array([d.tolist() for d in t_data])
            t_label = t_label.decode("utf-8")
            res_data.append(t_data)
            res_labels.append(t_label)
            
        return np.array(res_data).swapaxes(1, 2), np.array(res_labels)
    
    train_X, train_y = extract_data(train_data)
    test_X, test_y = extract_data(test_data)
    
    scalar = StandardScaler()
    scalar.fit(train_X.reshape(-1, train_X.shape[-1]))
    train_X = scalar.transform(train_X.reshape(-1, train_X.shape[-1])).reshape(train_X.shape)
    test_X = scalar.transform(test_X.reshape(-1, test_X.shape[-1])).reshape(test_X.shape)
    
    
    labels = np.unique(train_y)
    transform = {k : i for i,k in enumerate(labels)}
    train_y = np.vectorize(transform.get)(train_y)
    test_y = np.vectorize(transform.get)(test_y)
    return train_X, train_y, test_X, test_y    
    