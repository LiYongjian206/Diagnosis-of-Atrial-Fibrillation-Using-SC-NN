# coding=gbk
import random
import numpy as np
from keras.utils import np_utils
def shuffle_set(data, label):
    train_row = list(range(len(label)))
    random.shuffle(train_row)
    Data = data[train_row]
    Label = label[train_row]
    return Data, Label

data=np.load('Data.npy')
print(data[0])
data=data.reshape(data.shape[0], 125, 10)#重塑

label=np.load('Label.npy')
print(label.shape)
label=np_utils.to_categorical(label,2)#编码

Data,Label=shuffle_set(data,label)

Data = np.save('D:\python_easy/' + 'Datamatrix', Data)
Label = np.save('D:\python_easy/' + 'Labelmatrix', Label)