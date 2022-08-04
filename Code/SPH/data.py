# encoding:utf-8
import csv
import pywt
import scipy.io
import numpy as np
from scipy.signal import resample

# --------------------小波去噪-----------------
def WTfilt_1d(sig):
    """
    对信号进行小波变换滤波
    :param sig: 输入信号，1-d array
    :return: 小波滤波后的信号，1-d array
    """
    coeffs = pywt.wavedec(sig, 'db6', level=9)
    coeffs[-1] = np.zeros(len(coeffs[-1]))
    coeffs[-2] = np.zeros(len(coeffs[-2]))
    coeffs[0] = np.zeros(len(coeffs[0]))
    sig_filt = pywt.waverec(coeffs, 'db6')
    return sig_filt

N = []
AF = []
# 数据读取
data = scipy.io.loadmat('afdata.mat')
data = data['ans'][:, 1]
data = WTfilt_1d(data)
path = 'label.csv'

# data = scipy.io.loadmat('afdata1.mat')
# data = data['ans'][:, 1]
# data = WTfilt_1d(data)
# path = 'label1.csv'
#
# data = scipy.io.loadmat('afdata2.mat')
# data = data['ans'][:, 1]
# data = WTfilt_1d(data)
# path = 'label2.csv'
# data = scipy.io.loadmat('afdata3.mat')
# data = data['ans'][:, 1]
# data = WTfilt_1d(data)
# path = 'label3.csv'
#
# data = scipy.io.loadmat('afdata4.mat')
# data = data['ans'][:, 1]
# data = WTfilt_1d(data)
# path = 'label4.csv'

f = 250
t = 5

with open(path, 'r') as csvfile:
    reader = csv.reader(csvfile)
    start = [row[1] for row in reader]

with open(path, 'r') as csvfile:
    reader = csv.reader(csvfile)
    label = [row[2] for row in reader]

# 数据分割
i = 1
while i<98621:
    print(i)
    j=i
    if label[j] =='N' :
        Start = int(int(start[j])/5)#5ms一个采样点
        Seg = data[Start:Start+1000]
        seg = resample(Seg, f*t, axis=0)  # 重采样
        N.append(seg)


    if label[j] == 'Af' :
        Start = int(int(start[j])/5)
        Seg = data[Start:Start+1000]
        seg = resample(Seg, f*t, axis=0)  # 重采样
        AF.append(seg)


    while int(start[i])<int(start[j])+5000 :#5000ms=5s
        i=i+1

def Z_ScoreNormalization(x,mu,sigma):
    x = (x - mu) / sigma;
    return x;

N = np.array(N)
N = Z_ScoreNormalization(N,np.average(N),np.std(N))

AF = np.array(AF)
AF = Z_ScoreNormalization(AF,np.average(AF),np.std(AF))

label_N = np.zeros(N.shape[0])
label_AF = np.ones(AF.shape[0])

Data = np.concatenate((N,  AF), axis=0)
Label = np.concatenate((label_N,  label_AF), axis=0)
# 数据保存
Data = np.save('D:\our_data/' + 'Data', Data)
Label = np.save('D:\our_data/' + 'Label', Label)

# Data = np.save('D:\our_data/' + 'Data1', Data)
# Label = np.save('D:\our_data/' + 'Label1', Label)
#
# Data = np.save('D:\our_data/' + 'Data2', Data)
# Label = np.save('D:\our_data/' + 'Label2', Label)
#
# Data = np.save('D:\our_data/' + 'Data3', Data)
# Label = np.save('D:\our_data/' + 'Label3', Label)
#
# Data = np.save('D:\our_data/' + 'Data4', Data)
# Label = np.save('D:\our_data/' + 'Label4', Label)

