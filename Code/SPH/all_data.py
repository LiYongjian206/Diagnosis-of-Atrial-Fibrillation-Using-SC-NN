import numpy as np
from keras.layers import Concatenate

# 数据导入
data = np.load('D:\our_data\Datamatrix.npy')
label = np.load('D:\our_data\Labelmatrix.npy')
data = np.array(data)

data1 = np.load('D:\our_data\Datamatrix1.npy')
label1 = np.load('D:\our_data\Labelmatrix1.npy')
data1 = np.array(data1)

data2 = np.load('D:\our_data\Datamatrix2.npy')
label2 = np.load('D:\our_data\Labelmatrix2.npy')
data2 = np.array(data2)

data3 = np.load('D:\our_data\Datamatrix3.npy')
label3 = np.load('D:\our_data\Labelmatrix3.npy')
data3 = np.array(data3)

data4 = np.load('D:\our_data\Datamatrix4.npy')
label4 = np.load('D:\our_data\Labelmatrix4.npy')
data4 = np.array(data4)

# 数据合并
Data = Concatenate(axis=0)([data,data1,data2,data3,data4])
Label = Concatenate(axis=0)([label,label1,label2,label3,label4])
# 数据保存
D = np.save('D:\our_data/' + 'Ddata_matrix', Data)
L = np.save('D:\our_data/' + 'Dlabel_matrix', Label)