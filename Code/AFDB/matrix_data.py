# coding=gbk

import random
import numpy as np
from keras.utils import np_utils

def shuffle_set (data, label):

    train_row = list(range(len(label)))
    random.shuffle(train_row)
    Data = data[train_row]
    Label = label[train_row]
    return Data, Label

data = np.load('data.npy')
data = data.reshape(data.shape[0], 125, 10)#重塑

label = np.load('label.npy')
label = np_utils.to_categorical(label,2)#编码

Data, Label = shuffle_set(data,label)

Data = np.save('D:\python_PAF/' + 'datamatrix', Data)
Label = np.save('D:\python_PAF/' + 'labelmatrix', Label)