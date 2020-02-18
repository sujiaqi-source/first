import numpy as np
import  keras
from keras.models import Sequential
from keras.layers import Dense, Activation,Dropout
import  xlrd
from keras.optimizers import Adam,SGD
from keras.callbacks import TensorBoard
from matplotlib import  pyplot
from sklearn import preprocessing



min_max_scaler = preprocessing.MinMaxScaler()
book = xlrd.open_workbook('data1-12.xlsx')
sheet1 = book.sheets()[0]
# print(sheet1)
x_train=np.zeros((1298,9),dtype=np.float32)
y_train=np.zeros((1298),dtype=np.float32)
x_test=np.zeros((150,9),dtype=np.float32)
y_test=np.zeros((150),dtype=np.float32)
# x_nor1=np.zeros((1298),dtype=np.float32)
# x_nor2=np.zeros((150),dtype=np.float32)


for i in range(154,1452,1):
    data=sheet1.row_values(i)
    x_train[i-154,0]=data[1]
    x_train[i - 154, 1] = data[2]/10000
    x_train[i - 154, 2] = data[3]/1000
    x_train[i-154,3]=data[4]
    x_train[i-154,4]=data[5]*100
    x_train[i-154,5]=data[7]/10
    x_train[i-154,6]=data[8]/100
    x_train[i-154,7]=data[9]/100
    x_train[i -154,8]=data[15]/10
    # x_train[i-154,9]=data[6]
    # x_train[i - 154, 10]=data[14]
    # x_train[i - 154, 8]=data[11]
    y_train[i-154]=data[11]/100
    # print(data[14])
    # print(y_train[i-])


for i in range(4,154,1):
    data=sheet1.row_values(i)
    x_test[i-4,0]=data[1]
    x_test[i -4, 1] = data[2]/10000
    x_test[i -4, 2] = data[3]/1000
    x_test[i-4,3]=data[4]
    x_test[i-4,4]=data[5]*100
    x_test[i-4,5]=data[7]/10
    x_test[i-4,6]=data[8]/100
    x_test[i-4,7]=data[9]/100
    x_test[i -4,8] = data[15]/10
    # x_test[i-4,9]=data[6]
    # x_test[i -4, 10]=data[14]
    # x_test[i -4, 8]=data[11]

    y_test[i -4] = data[11]/100

min_max_scaler = preprocessing.MinMaxScaler()
for i in range(9):
    x_train[:, i] = min_max_scaler.fit_transform(x_train[:, i].reshape(1298, 1)).reshape(1298)
    x_test[:, i] = min_max_scaler.fit_transform(x_test[:, i].reshape(150, 1)).reshape(150)


print(x_train.shape)
print(y_train.shape)
print(x_test.shape)
print(y_test.shape)


model=Sequential()
model.add(Dense(output_dim=100,input_dim=9))
model.add(Activation('relu'))
# model.add(Dropout(0.10))
model.add(Dense(10))
model.add(Activation('relu'))
# model.add(Dropout(0.1))
model.add(Dense(1))
# adam=SGD(0.01)
model.compile(loss='mae',optimizer='adam')
checkpoint=keras.callbacks.ModelCheckpoint('ANNweights.hdf5',
 monitor='val_loss',
verbose=1,
save_best_only=True,
save_weights_only=False,
mode='min',
period=1)
tbCallBack = TensorBoard(log_dir='./logs',  # log 目录
                 histogram_freq=0,  # 按照何等频率（epoch）来计算直方图，0为不计算
#                  batch_size=32,     # 用多大量的数据计算直方图
                 write_graph=True,  # 是否存储网络结构图
                 write_grads=True, # 是否可视化梯度直方图
                 write_images=True,# 是否可视化参数
                 embeddings_freq=0,
                 embeddings_layer_names=None,
                 embeddings_metadata=None)
history = model.fit(x_train, y_train, epochs=3, batch_size=10, validation_data=(x_test, y_test), verbose=2, shuffle=False,callbacks=[tbCallBack,checkpoint])
pred = model.predict(x_test)

a=0
d=0
for i in range(150):
    d = y_test[i] + d
    a=abs(y_test[i]-pred[i])+a

d=d/150
a=a/150
print(1-a/d)
pyplot.plot(history.history['loss'], label='train')
pyplot.plot(history.history['val_loss'], label='test')
pyplot.legend()
pyplot.show()










