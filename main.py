import bnlearn as bn
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt, rcParams
from matplotlib.font_manager import FontProperties
from sklearn import metrics
from sklearn.metrics import roc_curve,auc
from sklearn.preprocessing import label_binarize

edges = [('S0', 'S1'),
         ('D0', 'S1'),
         ('S1', 'K'),
         ('S1', 'M'),
         ('K', 'D1'),
         ('M', 'D1'),
         ('D0', 'D1'),
         ]

DAG = bn.make_DAG(edges)
print(DAG['adjmat'])
bn.print_CPD(DAG)
bn.plot(DAG)

df = pd.read_csv('工业000034/K和M.csv')
data=df.iloc[:,1:7]
start=1306
end=1346
train=pd.concat([data.loc[:start],data.loc[end:]])
test=data.loc[start:end]
print(test)

DAG = bn.parameter_learning.fit(DAG, train ,methodtype='maximumlikelihood')

pred0=[]
pred0_pro=[]
pred1=[]
pred1_pro=[]
count=0
for i in range(0,len(test)):
    q0 = bn.inference.fit(DAG, variables=['S1'], evidence={
                                                            'S0': test.loc[i+start,'S0'],
                                                           'K':test.loc[i+start,'K'],
                                                           'M':test.loc[i+start,'M']
                                                           })
    temp=q0.df['p'].idxmax()
    pred0_pro.insert(i,q0.df['p'].tolist())
    pred0.append(temp)
    q1 = bn.inference.fit(DAG, variables=['S1'], evidence={
                                                            'S0': test.loc[i+start,'S0'],
                                                           'K': test.loc[i + start, 'K'],
                                                           'M': test.loc[i + start, 'M'],
                                                           'D0':test.loc[i+start,'D0'],
                                                           'D1':test.loc[i+start,'D1']
                                                           })
    temp1 = q1.df['p'].idxmax()
    pred1_pro.insert(i, q1.df['p'].tolist())
    pred1.append(temp1)

print("未加传动证据：")
A=metrics.accuracy_score(test['S1'], pred0)
print('准确度：'+str(A))
P=metrics.precision_score(test['S1'], pred0, average='weighted')
print('精确率：'+str(P))
R=metrics.recall_score(test['S1'], pred0, average='weighted')
print('召回率：'+str(R))
F1=metrics.f1_score(test['S1'], pred0,average='weighted')
print('F1值：'+str(F1))
print('*****************')
print('加传动证据：')
A=metrics.accuracy_score(test['S1'], pred1)
print('准确度：'+str(A))
P=metrics.precision_score(test['S1'], pred1, average='weighted')
print('精确率：'+str(P))
R=metrics.recall_score(test['S1'], pred1, average='weighted')
print('召回率：'+str(R))
F1=metrics.f1_score(test['S1'], pred1,average='weighted')
print('F1值：'+str(F1))


n_classes = len(np.unique(test['S1']))
y_test = label_binarize(test['S1'], classes=np.arange(n_classes))

# 计算每个类别的 ROC 曲线
fpr0 = dict()
tpr0 = dict()
fpr1 = dict()
tpr1 = dict()
roc_auc0 = dict()
roc_auc1 = dict()
for i in range(n_classes):
    fpr0[i], tpr0[i], _0 = roc_curve(y_test[:, i], np.array(pred0_pro)[:, i])
    fpr1[i], tpr1[i], _1 = roc_curve(y_test[:, i], np.array(pred1_pro)[:, i])
    roc_auc0[i] = auc(fpr0[i], tpr0[i])
    roc_auc1[i] = auc(fpr1[i], tpr1[i])

fpr0["micro"], tpr0["micro"], _0 = roc_curve(y_test.ravel(), np.array(pred0_pro).ravel())
roc_auc0["micro"] = auc(fpr0["micro"], tpr0["micro"])
fpr1["micro"], tpr1["micro"], _1 = roc_curve(y_test.ravel(), np.array(pred1_pro).ravel())
roc_auc1["micro"] = auc(fpr1["micro"], tpr1["micro"])

all_fpr0 = np.unique(np.concatenate([fpr0[i] for i in range(n_classes)]))
all_fpr1 = np.unique(np.concatenate([fpr1[i] for i in range(n_classes)]))

mean_tpr0 = np.zeros_like(all_fpr0)
mean_tpr1 = np.zeros_like(all_fpr1)
for i in range(n_classes):
    mean_tpr0 += np.interp(all_fpr0, fpr0[i], tpr0[i])
    mean_tpr1 += np.interp(all_fpr1, fpr1[i], tpr1[i])

mean_tpr0 /= n_classes
mean_tpr1 /= n_classes

fpr0["macro"] = all_fpr0
tpr0["macro"] = mean_tpr0
fpr1["macro"] = all_fpr1
tpr1["macro"] = mean_tpr1
roc_auc0["macro"] = auc(fpr0["macro"], tpr0["macro"])
roc_auc1["macro"] = auc(fpr1["macro"], tpr1["macro"])

font1 = FontProperties(fname=r"c:\windows\fonts\STZHONGS.TTF")      #华文中宋,size=6
font2 = FontProperties(fname=r"c:\windows\fonts\timesi.ttf")    #Times New Roman
config = {
    "font.family": 'serif',
    "font.size": 10,
    "mathtext.fontset": 'stix',
    "font.serif": ['Times New Roman'],
}
rcParams.update(config)     #设置全文默认字体


plt.figure()
plt.plot(fpr0["macro"], tpr0["macro"],label="MATID(0) (AUC = {0:0.2f})".format(roc_auc0["macro"]),linestyle='--')
plt.plot(fpr1["macro"], tpr1["macro"],label="MATID(1) (AUC = {0:0.2f})".format(roc_auc1["macro"]))
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
x=[0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1]
y=[0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1]
plt.xticks(x)
plt.yticks(y)
plt.xlabel("假正类率",loc='right',fontsize=12,fontproperties=font1)
plt.ylabel("真正类率",loc='top',fontsize=12,fontproperties=font1)
plt.title("下跌趋势ROC曲线",fontsize=12,fontproperties=font1)
plt.legend()
plt.savefig('E:/桌面/3033下跌roc.svg',dpi=800)
plt.show()

plt.figure(figsize=(8,5))#figsize=(20,10)
plt.scatter(list(range(len(pred0))), pred0,marker='v',color='g')  #散点图
plt.scatter(list(range(len(pred1))), pred1,marker='^',color='r')
plt.plot(list(range(len(np.array(test['S1'])))), np.array(test['S1']),':',marker='s',color='black')

x=list(range(0,len(pred0),2))
y=list([0,1,2])
plt.xticks(x)#自定义X轴间隔
plt.yticks(y)
plt.title('下跌趋势下MATID与真实趋势对比图',fontsize=12,fontproperties=font1)  #标题
plt.xlabel('交易日/d',loc='right',fontsize=12,fontproperties=font1)  #X轴标签
plt.ylabel('趋势离散值',loc='top',fontsize=12,fontproperties=font1)  #Y轴标签
plt.legend(['MATID(0)','MATID(1)', '真实趋势'],loc=2,prop=font1)
plt.savefig('E:/桌面/3033下跌.svg',dpi=800)
plt.show()