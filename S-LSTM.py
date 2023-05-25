import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM
from tensorflow.keras.layers import Dense, Dropout,Activation
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.wrappers.scikit_learn import KerasRegressor
from sklearn.model_selection import GridSearchCV
from tensorflow.keras import optimizers
from sklearn import metrics

f = open('工业000034/工业均线数据.csv')
df = pd.read_csv(f)  # 读入股票数据
start=2
end=10
df = df.iloc[:, start:end]

df_for_training=df[1535:2981]
df_for_testing=df[2981:3027]

scaler = MinMaxScaler(feature_range=(0,1))#缩放至0到1
df_for_training_scaled = scaler.fit_transform(df_for_training)
df_for_testing_scaled=scaler.transform(df_for_testing)

#对训练数据进行拆分
def createXY(dataset,n_past):
    dataX = []
    dataY = []
    for i in range(n_past, len(dataset)):
            dataX.append(dataset[i - n_past:i, 1:dataset.shape[1]])
            dataY.append(dataset[i,0])
    return np.array(dataX),np.array(dataY)

time_step=4
trainX,trainY=createXY(df_for_training_scaled,time_step)
testX,testY=createXY(df_for_testing_scaled,time_step)

sgd = optimizers.SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)#定义一个优化器

def build_model(optimizer):
    grid_model = Sequential()
    grid_model.add(LSTM(64,return_sequences=True,input_shape=(time_step,end-start-1)))
    grid_model.add(LSTM(32))
    grid_model.add(Dropout(0.2))
    grid_model.add(Dense(1))
    grid_model.add(Activation('tanh'))
    grid_model.compile(loss = 'mse',optimizer = optimizer,metrics=[ 'accuracy' ])
    return grid_model

my_model=build_model(sgd)
history=my_model.fit(trainX,trainY,validation_data=(testX, testY),epochs=40,batch_size=4,verbose=2)
# grid_model = KerasRegressor(build_fn=build_model,verbose=2)
#
# parameters = {'batch_size' : [100],
#               'epochs' : [60],
#               'optimizer' : [ 'adam','Adadelta'] }#,'Adadelta'动态生成学习率    'adam','Adadelta'
#
# grid_search  = GridSearchCV(estimator = grid_model,
#                             param_grid = parameters,
#                             cv = 2)            # 网格搜索
#
# grid_search = grid_search.fit(trainX,trainY,validation_data=(testX, testY))
# grid_search.best_params_
# print(grid_search.best_params_)
#
# my_model=grid_search.best_estimator_.model

prediction=my_model.predict(testX)
prediction_copies_array = np.repeat(prediction,end-start, axis=-1)
pred=scaler.inverse_transform(np.reshape(prediction_copies_array,(len(prediction),end-start)))[:,0]
original_copies_array = np.repeat(testY,end-start, axis=-1)
original=scaler.inverse_transform(np.reshape(original_copies_array,(len(testY),end-start)))[:,0]

original1=[]
pred1=[]
temp=(original.max()-original.min())/2
for i in range(len(original)):
    if original.min()<original[i]<original.min()+temp:
        original1.append(0)
    else:original1.append(1)
for i in range(len(pred)):
    if abs((pred[i]-original[i])/original[i])<0.017:
        pred1.append(original1[i])
    elif original1[i]==1:
        pred1.append(0)
    else:pred1.append(1)
print(pred1)
print(original1)

A=metrics.accuracy_score(original1, pred1)
print('准确度：'+str(A))
P=metrics.precision_score(original1, pred1)#, average='weighted'
print('精确率：'+str(P))
R=metrics.recall_score(original1, pred1)#, average='weighted'
print('召回率：'+str(R))
F1=metrics.f1_score(original1, pred1)#, average='weighted'
print('F1值：'+str(F1))

plt.figure()
plt.plot(list(range(len(pred))), pred, color='b', )
plt.plot(list(range(len(original))), original, color='r')
plt.legend(['pred', 'true'])
plt.show()

plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model train vs validation loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train','validation'], loc='upper right')
plt.show()