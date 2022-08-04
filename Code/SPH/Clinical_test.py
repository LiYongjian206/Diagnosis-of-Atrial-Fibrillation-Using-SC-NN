import keras
import numpy as np
from keras.utils import np_utils
from sklearn.metrics import confusion_matrix  # 混淆矩阵
from sklearn import metrics  # 模型评估

# 加载数据集
data = np.load('D:\our_data\maxdata.npy')#our data
label = np.load('D:\our_data\maxlabel.npy')
data = np.array(data)

# 加载训练好的模型
model = keras.models.load_model('D:/python_PAF/model_matrix.hdf5')#MIT
# model = keras.models.load_model('D:/pythonProject/my_model.hdf5')#挑战赛
# model = keras.models.load_model('D:/python_easy/model_matrix.hdf5')#CPSC

loss,acc = model.evaluate(data, label, batch_size=64, verbose=2)
F1=[]
Con_Matr=[]
y_pred = model.predict(data)
y_test = np.argmax(label, axis=1)
y_pred = np.argmax(y_pred, axis=1)
f1 = metrics.f1_score(y_test, y_pred, average='macro')
F1.append(f1)
con_matr = confusion_matrix(y_test, y_pred)
Con_Matr.append(con_matr)
print(Con_Matr)
print(F1)