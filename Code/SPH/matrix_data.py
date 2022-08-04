# coding=gbk
import random
import numpy as np
from keras.utils import np_utils

# 打乱数据顺序
def shuffle_set(data, label):
    train_row = list(range(len(label)))
    random.shuffle(train_row)
    Data = data[train_row]
    Label = label[train_row]
    return Data, Label

# 读取数据
# data=np.load('Data.npy')
# data=np.load('Data1.npy')
# data=np.load('Data2.npy')
# data=np.load('Data3.npy')
data=np.load('Data4.npy')
data=data.reshape(data.shape[0], 125, 10)#矩阵重塑

# label=np.load('Label.npy')
# label=np.load('Label1.npy')
# label=np.load('Label2.npy')
# label=np.load('Label3.npy')
label=np.load('Label4.npy')
label=np_utils.to_categorical(label,2)#编码

Data,Label=shuffle_set(data,label)

Data = np.save('D:\our_data/' + 'Datamatrix4', Data)
Label = np.save('D:\our_data/' + 'Labelmatrix4', Label)