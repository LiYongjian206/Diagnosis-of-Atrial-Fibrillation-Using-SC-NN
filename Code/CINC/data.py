import numpy as np
import wfdb
import pywt
from wfdb import processing

AF = []
N = []

path = 'D:/python_PAF/training2017/'
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

for i in range(8528):

    with open("answers.txt", "r") as f:  # 打开文件
        data = f.read()  # 读取文件
        name = data[9*i:9*i+6]
        label = data[7+9*i]

        Record = wfdb.rdsamp(path + name )[0][:, 0]
        record_ori,_ = wfdb.processing.resample_sig(x=Record, fs=300, fs_target=250)
        record = WTfilt_1d(record_ori)  # 小波去噪


        if len(record) < 2500 and len(record)>=1250:
            Seg  = record[0:1250]
            if label == 'N' :
                N.append(Seg)
            if label == 'A':
                AF.append(Seg)

        if len(record) < 3750 and len(record)>=2500:
            Seg = record[0:1250]
            Seg1 = record[1250:2500]
            if label == 'N'  :
                N.append(Seg)
            if label == 'A':
                AF.append(Seg)
                AF.append(Seg1)


        if len(record) < 5000 and len(record)>=3750:

            Seg = record[0:1250]
            Seg1 = record[1250:2500]
            Seg2 = record[2500:3750]
            if label == 'N'  :
                N.append(Seg)
            if label == 'A':
                AF.append(Seg)
                AF.append(Seg1)
                AF.append(Seg2)


        if len(record) < 6250 and len(record)>=5000:
            Seg = record[0:1250]
            Seg1 = record[1250:2500]
            Seg2 = record[2500:3750]
            Seg3 = record[3750:5000]
            if label == 'N'  :
                N.append(Seg)
            if label == 'A':
                AF.append(Seg)
                AF.append(Seg1)
                AF.append(Seg2)
                AF.append(Seg3)


        if len(record) < 7500 and len(record)>=6250:
                Seg = record[0:1250]
                Seg1 = record[1250:2500]
                Seg2 = record[2500:3750]
                Seg3 = record[3750:5000]
                Seg4 = record[5000:6250]
                if label == 'N'  :
                    N.append(Seg)
                if label == 'A':
                    AF.append(Seg)
                    AF.append(Seg1)
                    AF.append(Seg2)
                    AF.append(Seg3)
                    AF.append(Seg4)


        if len(record)<8750 and len(record)>=7500:
        # if  len(record) >= 7500*2:
                    Seg = record[0:1250]
                    Seg1 = record[1250:2500]
                    Seg2 = record[2500:3750]
                    Seg3 = record[3750:5000]
                    Seg4 = record[5000:6250]
                    Seg5 = record[6250:7500]
                    if label == 'N'  :
                        N.append(Seg)
                    if label == 'A':
                        AF.append(Seg)
                        AF.append(Seg1)
                        AF.append(Seg2)
                        AF.append(Seg3)
                        AF.append(Seg4)
                        AF.append(Seg5)


        if len(record) < 10000 and len(record)>=8750:
                    Seg = record[0:1250]
                    Seg1 = record[1250:2500]
                    Seg2 = record[2500:3750]
                    Seg3 = record[3750:5000]
                    Seg4 = record[5000:6250]
                    Seg5 = record[6250:7500]
                    Seg6 = record[7500:8750]
                    if label == 'N'  :
                        N.append(Seg)
                    if label == 'A':
                        AF.append(Seg)
                        AF.append(Seg1)
                        AF.append(Seg2)
                        AF.append(Seg3)
                        AF.append(Seg4)
                        AF.append(Seg5)
                        AF.append(Seg6)

        if len(record) < 11250 and len(record)>=10000:
                    Seg = record[0:1250]
                    Seg1 = record[1250:2500]
                    Seg2 = record[2500:3750]
                    Seg3 = record[3750:5000]
                    Seg4 = record[5000:6250]
                    Seg5 = record[6250:7500]
                    Seg6 = record[7500:8750]
                    Seg7 = record[8750:10000]
                    if label == 'N' :
                        N.append(Seg)
                    if label == 'A':
                        AF.append(Seg)
                        AF.append(Seg1)
                        AF.append(Seg2)
                        AF.append(Seg3)
                        AF.append(Seg4)
                        AF.append(Seg5)
                        AF.append(Seg6)
                        AF.append(Seg7)


        if len(record) < 12500 and len(record)>=11250:
                    Seg = record[0:1250]
                    Seg1 = record[1250:2500]
                    Seg2 = record[2500:3750]
                    Seg3 = record[3750:5000]
                    Seg4 = record[5000:6250]
                    Seg5 = record[6250:7500]
                    Seg6 = record[7500:8750]
                    Seg7 = record[8750:10000]
                    Seg8 = record[10000:11250]
                    if label == 'N' :
                        N.append(Seg)
                    if label == 'A':
                        AF.append(Seg)
                        AF.append(Seg1)
                        AF.append(Seg2)
                        AF.append(Seg3)
                        AF.append(Seg4)
                        AF.append(Seg5)
                        AF.append(Seg6)
                        AF.append(Seg7)
                        AF.append(Seg8)


        if len(record) < 13750 and len(record)>=12500:
                    Seg = record[0:1250]
                    Seg1 = record[1250:2500]
                    Seg2 = record[2500:3750]
                    Seg3 = record[3750:5000]
                    Seg4 = record[5000:6250]
                    Seg5 = record[6250:7500]
                    Seg6 = record[7500:8750]
                    Seg7 = record[8750:10000]
                    Seg8 = record[10000:11250]
                    Seg9 = record[11250:12500]
                    if label == 'N'  :
                        N.append(Seg)
                    if label == 'A':
                        AF.append(Seg)
                        AF.append(Seg1)
                        AF.append(Seg2)
                        AF.append(Seg3)
                        AF.append(Seg4)
                        AF.append(Seg5)
                        AF.append(Seg6)
                        AF.append(Seg7)
                        AF.append(Seg8)
                        AF.append(Seg9)


        if len(record) < 15000 and len(record)>=13750:
                    Seg = record[0:1250]
                    Seg1 = record[1250:2500]
                    Seg2 = record[2500:3750]
                    Seg3 = record[3750:5000]
                    Seg4 = record[5000:6250]
                    Seg5 = record[6250:7500]
                    Seg6 = record[7500:8750]
                    Seg7 = record[8750:10000]
                    Seg8 = record[10000:11250]
                    Seg9 = record[11250:12500]
                    Seg10 = record[12500:13750]
                    if label == 'N':
                        N.append(Seg)
                    if label == 'A':
                        AF.append(Seg)
                        AF.append(Seg1)
                        AF.append(Seg2)
                        AF.append(Seg3)
                        AF.append(Seg4)
                        AF.append(Seg5)
                        AF.append(Seg6)
                        AF.append(Seg7)
                        AF.append(Seg8)
                        AF.append(Seg9)
                        AF.append(Seg10)


        if len(record) >=15000:
                    Seg = record[0:1250]
                    Seg1 = record[1250:2500]
                    Seg2 = record[2500:3750]
                    Seg3 = record[3750:5000]
                    Seg4 = record[5000:6250]
                    Seg5 = record[6250:7500]
                    Seg6 = record[7500:8750]
                    Seg7 = record[8750:10000]
                    Seg8 = record[10000:11250]
                    Seg9 = record[11250:12500]
                    Seg10 = record[12500:13750]
                    Seg11 = record[13750:15000]
                    if label == 'N'  :
                        N.append(Seg)
                    if label == 'A':
                        AF.append(Seg)
                        AF.append(Seg1)
                        AF.append(Seg2)
                        AF.append(Seg3)
                        AF.append(Seg4)
                        AF.append(Seg5)
                        AF.append(Seg6)
                        AF.append(Seg7)
                        AF.append(Seg8)
                        AF.append(Seg9)
                        AF.append(Seg10)
                        AF.append(Seg11)


def Z_ScoreNormalization(x,mu,sigma):
    x = (x - mu) / sigma;
    return x;

N_segement = np.array(N)
N_segement = Z_ScoreNormalization(N_segement,np.average(N_segement),np.std(N_segement))

AF_segement = np.array(AF)
AF_segement = Z_ScoreNormalization(AF_segement,np.average(AF_segement),np.std(AF_segement))

label_N = np.zeros(N_segement.shape[0])
label_AF = np.ones(AF_segement.shape[0])

Data = np.concatenate((N_segement,  AF_segement), axis=0)
Label = np.concatenate((label_N,  label_AF), axis=0)

Data = np.save('D:\pythonProject/' + 'data', Data)
Label = np.save('D:\pythonProject/' + 'label', Label)