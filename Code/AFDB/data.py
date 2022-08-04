import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import wfdb
import glob
import pywt
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

def Z_ScoreNormalization(x,mu,sigma):
    x = (x - mu) / sigma;
    return x;

# -------------------------心拍截取-------------------

def heartbeat(file):

    AF = [];
    N  = [];

    # --------去掉指定的两个导联的头文件---------
    De_file = [panth[:-1] + '\\00735.hea', panth[:-1] + '\\03665.hea']
    file = list(set(file).difference(set(De_file)))
    for f in range(len(file)):
        annotation = wfdb.rdann(panth + file[f][-9:-4], 'atr')
        record_name = annotation.record_name  # 读取记录名称
        Record = wfdb.rdsamp(panth + record_name)[0][:, 0]  # 一般只取一个导联
        record = WTfilt_1d(Record)  # 小波去噪
        label = annotation.aux_note # 房颤标签列表
        label_index = annotation.sample  # 标签索引列表

        for j in range(len(label_index)):

            if  label[j] == 'N\x00' or label[j] == 'J\x00' :
                i = 0
                if j != (len(label_index)-1):

                    while label_index[j] + 250*(i+1) < label_index[j+1]:
                       Seg = record[label_index[j]+(250*i):label_index[j] + 250*(i+1)]
                       Seg = Z_ScoreNormalization(Seg, np.average(Seg), np.std(Seg))
                       N.append(Seg)
                       i = i + 1

                if j == (len(label_index)-1):

                    while label_index[j] + 250*(i+1) < len(record):
                       Seg = record[label_index[j]+(250*i):label_index[j] + 250*(i+1)]
                       Seg = Z_ScoreNormalization(Seg, np.average(Seg), np.std(Seg))
                       N.append(Seg)
                       i = i + 1

            if label[j] == 'AF\x00' or label[j] == 'AFL\x00':
                i = 0
                if j != (len(label_index)-1):
                    while label_index[j] + 250 * (i + 1) < label_index[j + 1]:
                        Seg = record[label_index[j] + (250 * i):label_index[j] + 250 * (i + 1)]
                        Seg = Z_ScoreNormalization(Seg, np.average(Seg), np.std(Seg))
                        AF.append(Seg)
                        i = i + 1

                if j  == (len(label_index)-1):
                    while label_index[j] + 250*(i+1) < len(record):
                       Seg = record[label_index[j]+(250*i):label_index[j] +250*(i+1)]
                       Seg = Z_ScoreNormalization(Seg, np.average(Seg), np.std(Seg))
                       AF.append(Seg)
                       i = i + 1

    N_segement = np.array(N)
    N_segement = N_segement[39000:]

    AF_segement = np.array(AF)

    label_N = np.zeros(N_segement.shape[0])
    label_AF = np.ones(AF_segement.shape[0])

    Data = np.concatenate((N_segement,  AF_segement), axis=0)
    Label = np.concatenate((label_N,  label_AF), axis=0)

    return Data, Label


# -----------------------心拍截取和保存---------------------
panth = 'D:\python_PAF/AF/'
file = glob.glob(panth + '*.hea')
Data, Label = heartbeat(file)

Data = np.save('D:\python_PAF/' + 'data', Data)
Label = np.save('D:\python_PAF/' + 'label', Label)








