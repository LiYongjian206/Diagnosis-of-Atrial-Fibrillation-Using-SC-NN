from sklearn.metrics import confusion_matrix  # 混淆矩阵
from sklearn.model_selection import train_test_split  # 划分数据集

import numpy as np
import matplotlib.pyplot as plt

from sklearn import metrics  # 模型评估
from keras import  Input, layers
from keras.callbacks import LearningRateScheduler
from keras.callbacks import ModelCheckpoint
from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPooling2D, BatchNormalization, LeakyReLU, \
    GlobalAveragePooling2D, Activation, GlobalMaxPooling2D, Reshape, \
    Concatenate, Add, Lambda
from keras import backend as K
from keras.models import Model
from keras.utils.vis_utils import plot_model

# 学习率更新以及调整
def scheduler(epoch):
    if epoch == 0:
        lr = K.get_value(model.optimizer.lr)  # keras默认0.001，CPSC小学习率
        K.set_value(model.optimizer.lr, lr*10)
        print("lr changed to {}".format(lr))
    if epoch != 0:
        lr = K.get_value(model.optimizer.lr)
        K.set_value(model.optimizer.lr, lr / (1 + 0.0001 * epoch))
        print("lr changed to {}".format(lr))
    return K.get_value(model.optimizer.lr)


# 定义空矩阵
F1 = []
Con_Matr = []

# 数据导入
data = np.load('D:\python_easy\Datamatrix.npy')
label = np.load('D:\python_easy\Labelmatrix.npy')
data = np.array(data)
X_train, X_test, y_train, y_test = train_test_split(data, label, test_size=0.2, random_state=32)


def self_attention1(inputs):

    # 先对卷积出来的keras张量进行lanbda封装
    x = Conv2D(256, (1, 1), strides=(1, 1))(inputs)
    bn = BatchNormalization(momentum=0.99, epsilon=0.001)(x)

    x = GlobalAveragePooling2D()(bn)
    x = Lambda(lambda x: x)(x)

    y = GlobalMaxPooling2D()(bn)
    y = Lambda(lambda y: y)(y)

    z = bn

    add = Add()([x, y])

    out = layers.Multiply()([z, add])  # 逐元素相乘
    out = Activation('softmax')(out)
    return out  # 给通道加权重

def self_attention2(inputs):

    # 先对卷积出来的keras张量进行lanbda封装
    x = Conv2D(128, (1, 1), strides=(1, 1))(inputs)
    bn = BatchNormalization(momentum=0.99, epsilon=0.001)(x)

    x = GlobalAveragePooling2D()(bn)
    x = Lambda(lambda x: x)(x)

    y = GlobalMaxPooling2D()(bn)
    y = Lambda(lambda y: y)(y)

    z = bn

    out1 = layers.Multiply()([z, x])  # 逐元素相乘
    out2 = layers.Multiply()([z, y])  # 逐元素相乘
    out = Concatenate(axis=3)([out1,out2])
    out = Activation('softmax')(out)
    return out  # 给通道加权重

def conv1(input_shape):
    conv1 = Conv2D(64, (1, 3), strides=(1, 1), padding='same')(input_shape)
    bn1 = BatchNormalization(momentum=0.99, epsilon=0.001)(conv1)
    action1 = LeakyReLU()(bn1)

    conv2 = Conv2D(64, (3, 1), strides=(1, 1), padding='same')(action1)
    bn2 = BatchNormalization(momentum=0.99, epsilon=0.001)(conv2)
    action2 = LeakyReLU()(bn2)
    out = action2
    return out

def conv2(input_shape):
    conv3 = Conv2D(128, (1, 5), strides=(1, 1), padding='same')(input_shape)
    bn3 = BatchNormalization(momentum=0.99, epsilon=0.001)(conv3)
    action3 = LeakyReLU()(bn3)

    conv4 = Conv2D(128, (5, 1), strides=(1, 1), padding='same')(action3)
    bn4 = BatchNormalization(momentum=0.99, epsilon=0.001)(conv4)
    action4 = LeakyReLU()(bn4)
    out = action4
    return out

def conv3(input_shape):
    conv5 = Conv2D(256, (1, 7), strides=(1, 1), padding='same')(input_shape)
    bn5 = BatchNormalization(momentum=0.99, epsilon=0.001)(conv5)
    action5 = LeakyReLU()(bn5)

    conv6 = Conv2D(256, (7, 1), strides=(1, 1), padding='same')(action5)
    bn6 = BatchNormalization(momentum=0.99, epsilon=0.001)(conv6)
    action6 = LeakyReLU()(bn6)
    out = action6
    return out

def models(input_shape):
    conv_1 = conv1(input_shape)
    pool1 = MaxPooling2D((2, 2), strides=2)(conv_1)

    conv_3 = conv2(pool1)
    pool3 = MaxPooling2D((2, 2), strides=2)(conv_3)

    conv_5 = conv3(pool3)
    pool5 = MaxPooling2D((2, 2), strides=2)(conv_5 )

    # 注意力机制
    # CWM
    self_att1 = self_attention1(pool5)

    # SWM
    self_att2 = self_attention2(pool5)

    # IIM
    add = Add()([self_att1, self_att2])
    x1 = Flatten()(add)
    x1 = Dropout(0.5)(x1)

    concat = Concatenate(axis=3)([self_att1,self_att2])
    x2 = Flatten()(concat)
    x2 = Dropout(0.5)(x2)

    x = Concatenate(axis=1)([x1, x2])
    x = Dropout(0.5)(x)

    out = Dense(2, activation='softmax')(x)

    out = Model(inputs=[input_shape], outputs=[out], name="NEWNet")
    return out

inputs = Input(shape=(125, 10, 1))

model = models(inputs)

model.summary()
plot_model(model, to_file='D:/python_easy/model_matrix.png')

model.compile(loss='binary_crossentropy',
              optimizer='Adam', metrics=['accuracy'])
filepath = "D:/python_easy/model_matrix_relu.hdf5"  # 保存模型的路径

checkpoint = ModelCheckpoint(filepath=filepath, verbose=2,
                             monitor='val_accuracy', mode='max')

reduce_lr = LearningRateScheduler(scheduler)  # 学习率的改变
callback_lists = [checkpoint, reduce_lr]

train_history = model.fit(x=X_train,
                          y=y_train, validation_split=0.2,
                          class_weight=None, callbacks=callback_lists,
                          epochs=300, batch_size=64, verbose=2)

loss, accuracy = model.evaluate(X_test, y_test)

Acc = []
Loss = []
Acc.append(accuracy)
Loss.append(loss)

y_pred = model.predict(X_test)
y_test = np.argmax(y_test, axis=1)
y_pred = np.argmax(y_pred, axis=1)
f1 = metrics.f1_score(y_test, y_pred, average='macro')
F1.append(f1)
con_matr = confusion_matrix(y_test, y_pred)
Con_Matr.append(con_matr)
print(Con_Matr)
print(F1)

def show_train_history(train_history, train, validation):
    plt.ylim(0, 1.05)
    plt.plot(train_history.history[train])
    plt.plot(train_history.history[validation])
    plt.title('Train History')
    plt.ylabel(train)
    plt.xlabel('Epoch')
    plt.legend(['train', 'validation'], loc='lower right')
    plt.show()


show_train_history(train_history, 'accuracy', 'val_accuracy')  # 绘制准确率执行曲线
show_train_history(train_history, 'loss', 'val_loss')  # 绘制损失函数执行曲线

from sklearn.metrics import roc_curve#画roc曲线
from sklearn.metrics import auc#auc值计算

y_pred_keras = y_pred
fpr_keras, tpr_keras, thresholds_keras = roc_curve(y_test, y_pred_keras)
auc_keras = auc(fpr_keras, tpr_keras)

plt.figure(1)
plt.plot([0, 1], [0, 1], 'r-.')
plt.plot(fpr_keras, tpr_keras, label='newNet (area = {:.4f})'.format(auc_keras))
plt.xlabel('False positive rate')
plt.ylabel('True positive rate')
plt.title('ROC curve')
plt.legend(loc='lower right')
plt.show()

