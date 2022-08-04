# encoding:utf-8
import csv
import wfdb
import pywt
import scipy.io
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from scipy.signal import resample
from wfdb import processing

def Z_ScoreNormalization(x,mu,sigma):
    x = (x - mu) / sigma;
    return x;

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

for i in range(1,6878):
    with open('REFERENCE.csv','r') as csvfile:
        reader = csv.reader(csvfile)
        name = [row[0] for row in reader]
    with open('REFERENCE.csv','r') as csvfile:
        reader = csv.reader(csvfile)
        label1 = [row[1] for row in reader]
    with open('REFERENCE.csv','r') as csvfile:
        reader = csv.reader(csvfile)
        label2 = [row[2] for row in reader]

    name = name[i]
    label1 = label1[i]
    label2 = label2[i]

    if i < 2001:
       data = scipy.io.loadmat('TrainingSet1/'  + str(name)  + '.mat') # 读取mat文件

    if i < 4471 and i > 2000:
        data = scipy.io.loadmat('TrainingSet2/' + str(name) + '.mat')  # 读取mat文件

    if i < 6878 and i > 4470:
        data = scipy.io.loadmat('TrainingSet3/' + str(name) + '.mat')  # 读取mat文件

    data = data['ECG']['data'][0, 0]

    data = WTfilt_1d(data[0,:])

    if 2500<=data.shape[1]<5000:
        seg = data[0, 0:2500]
        seg = WTfilt_1d(seg)
        seg = resample(seg, 1250, axis=0)#重采样

        if label1 == '1' or label2 == '1':
            N.append(seg)

        if label1 == '2' or label2 == '2':
            AF.append(seg)

    if 5000 <= data.shape[1] < 7500 :
        seg = data[0, 0:2500]
        seg = WTfilt_1d(seg)
        seg = resample(seg, 1250, axis=0)  # 重采样
        seg1 = data[0, 2500:5000]
        seg1 = WTfilt_1d(seg1)
        seg1 = resample(seg1, 1250, axis=0)  # 重采样

        if label1 == '1' or label2 == '1':
            N.append(seg)
            N.append(seg1)

        if label1 == '2' or label2 == '2':
            AF.append(seg)
            AF.append(seg1)

    if 7500 <= data.shape[1] < 10000 :
        seg = data[0, 0:2500]
        seg = WTfilt_1d(seg)
        seg = resample(seg, 1250, axis=0)  # 重采样
        seg1 = data[0, 2500:5000]
        seg1 = WTfilt_1d(seg1)
        seg1 = resample(seg1, 1250, axis=0)  # 重采样
        seg2 = data[0, 5000:7500]
        seg2 = WTfilt_1d(seg2)
        seg2 = resample(seg2, 1250, axis=0)  # 重采样

        if label1 == '1' or label2 == '1':
            N.append(seg)
            N.append(seg1)
            N.append(seg2)

        if label1 == '2' or label2 == '2':
            AF.append(seg)
            AF.append(seg1)
            AF.append(seg2)

    if 10000 <= data.shape[1] < 12500 :
        seg = data[0, 0:2500]
        seg = WTfilt_1d(seg)
        seg = resample(seg, 1250, axis=0)  # 重采样
        seg1 = data[0, 2500:5000]
        seg1 = WTfilt_1d(seg1)
        seg1 = resample(seg1, 1250, axis=0)  # 重采样
        seg2 = data[0, 5000:7500]
        seg2 = WTfilt_1d(seg2)
        seg2 = resample(seg2, 1250, axis=0)  # 重采样
        seg3 = data[0, 7500:10000]
        seg3 = WTfilt_1d(seg3)
        seg3 = resample(seg3, 1250, axis=0)  # 重采样

        if label1 == '1' or label2 == '1':
            N.append(seg)
            N.append(seg1)
            N.append(seg2)
            N.append(seg3)

        if label1 == '2' or label2 == '2':
            AF.append(seg)
            AF.append(seg1)
            AF.append(seg2)
            AF.append(seg3)

    if 12500 <= data.shape[1] < 15000 :
        seg = data[0, 0:2500]
        seg = WTfilt_1d(seg)
        seg = resample(seg, 1250, axis=0)  # 重采样
        seg1 = data[0, 2500:5000]
        seg1 = WTfilt_1d(seg1)
        seg1 = resample(seg1, 1250, axis=0)  # 重采样
        seg2 = data[0, 5000:7500]
        seg2 = WTfilt_1d(seg2)
        seg2 = resample(seg2, 1250, axis=0)  # 重采样
        seg3 = data[0, 7500:10000]
        seg3 = WTfilt_1d(seg3)
        seg3 = resample(seg3, 1250, axis=0)  # 重采样
        seg4 = data[0, 10000:12500]
        seg4 = WTfilt_1d(seg4)
        seg4 = resample(seg4, 1250, axis=0)  # 重采样

        if label1 == '1' or label2 == '1':
            N.append(seg)
            N.append(seg1)
            N.append(seg2)
            N.append(seg3)
            N.append(seg4)

        if label1 == '2' or label2 == '2':
            AF.append(seg)
            AF.append(seg1)
            AF.append(seg2)
            AF.append(seg3)
            AF.append(seg4)

    if 15000 <= data.shape[1]  :
        seg = data[0, 0:2500]
        seg = WTfilt_1d(seg)
        seg = resample(seg, 1250, axis=0)  # 重采样
        seg1 = data[0, 2500:5000]
        seg1 = WTfilt_1d(seg1)
        seg1 = resample(seg1, 1250, axis=0)  # 重采样
        seg2 = data[0, 5000:7500]
        seg2 = WTfilt_1d(seg2)
        seg2 = resample(seg2, 1250, axis=0)  # 重采样
        seg3 = data[0, 7500:10000]
        seg3 = WTfilt_1d(seg3)
        seg3 = resample(seg3, 1250, axis=0)  # 重采样
        seg4 = data[0, 10000:12500]
        seg4 = WTfilt_1d(seg4)
        seg4 = resample(seg4, 1250, axis=0)  # 重采样
        seg5 = data[0, 12500:15000]
        seg5 = WTfilt_1d(seg5)
        seg5 = resample(seg5, 1250, axis=0)  # 重采样

        if label1 == '1' or label2 == '1':
            N.append(seg)
            N.append(seg1)
            N.append(seg2)
            N.append(seg3)
            N.append(seg4)
            N.append(seg5)


        if label1 == '2' or label2 == '2':
            AF.append(seg)
            AF.append(seg1)
            AF.append(seg2)
            AF.append(seg3)
            AF.append(seg4)
            AF.append(seg5)

N = np.array(N)
N = Z_ScoreNormalization(N,np.average(N),np.std(N))

AF = np.array(AF)
AF = Z_ScoreNormalization(AF,np.average(AF),np.std(AF))

label_N = np.zeros(N.shape[0])
label_AF = np.ones(AF.shape[0])

Data = np.concatenate((N,  AF), axis=0)
Label = np.concatenate((label_N,  label_AF), axis=0)

Data = np.save('D:\python_easy/' + 'Data', Data)
Label = np.save('D:\python_easy/' + 'Label', Label)