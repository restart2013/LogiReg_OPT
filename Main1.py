import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import time
import sys
import os

from LogisticReg import *

# 读入数据&预处理
df = pd.read_csv('BreastCancerData.csv',encoding='utf-8')
print(df)
def std_scale(df,column):
    for col in column:
        df[col] = (df[col]-df[col].mean())/df[col].std()
data=df.copy()
data['Diagnosis'] = data.Diagnosis.map(lambda x:1 if x=='M' else 0)
std_scale(data,data.columns[2:])
x = data.drop(['id','Diagnosis'],axis=1).to_numpy()
y = data.Diagnosis.to_numpy()

# 取消打印
class HiddenPrint:
    def __init__(self, activated=True):
        # activated参数表示当前修饰类是否被**
        self.activated = activated
        self.original_stdout = None

    def open(self):
        sys.stdout.close()
        sys.stdout = self.original_stdout

    def close(self):
        self.original_stdout = sys.stdout
        sys.stdout = open(os.devnull, 'w')
        # 这里的os.devnull实际上就是Linux系统中的“/dev/null”
        # /dev/null会使得发送到此目标的所有数据无效化，就像“被删除”一样
        # 这里使用/dev/null对sys.stdout输出流进行重定向

    def __enter__(self):
        if self.activated:
            self.close()

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.activated:
            self.open()

# 画图准备
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False



#############实验部分#################
# 梯度下降
lr_gd = LogisticRegression(10)
lr_gd.initparam('norm')
dic_gd = lr_gd.train(x, y,200,method='graddescent')
# adam
lr_adam = LogisticRegression(10)
lr_adam.initparam('norm')
dic_adam = lr_adam.train(x, y,200,method='adam')
# 牛顿法
lr_nt = LogisticRegression(10)
lr_nt.initparam('norm')
dic_nt = lr_nt.train(x, y,200,method='newton')
# BFGS
lr_bfgs = LogisticRegression(10)
lr_bfgs.initparam('norm')
dic_bfgs = lr_bfgs.train(x, y,200,method='gfbs')
# DFP
lr_dfp = LogisticRegression(10)
lr_dfp.initparam('norm')
dic_dfp = lr_dfp.train(x, y,200,method='dfp')

plt.plot(range(len(dic_gd['loss_line'])),dic_gd['loss_line'], color='#20B2AA')
plt.plot(range(len(dic_adam['loss_line'])),dic_adam['loss_line'], color='#CD5C5C')
plt.plot(range(len(dic_nt['loss_line'])),dic_nt['loss_line'], color='#228B22')
plt.plot(range(len(dic_bfgs['loss_line'])),dic_bfgs['loss_line'], color='#FF7F50')
plt.plot(range(len(dic_dfp['loss_line'])),dic_dfp['loss_line'], color='#7B68EE')
plt.xlim(-5,80)
plt.legend(['Gradient Descent', 'Adam', 'Newton', 'BFGS', 'DFP'])
plt.xlabel('epoch')
plt.ylabel('logLoss')
plt.show()

plt.bar(x=['Gradient Descent', 'Newton', 'BFGS', 'DFP'], height=[dic_gd['loss_line'][-1], dic_nt['loss_line'][-1], dic_bfgs['loss_line'][-1], dic_dfp['loss_line'][-1]],
       color=['#20B2AA','#228B22','#FF7F50','#7B68EE'])
plt.xlabel('算法')
plt.ylabel('平均收敛loss/精度')
for pltx, plty in zip(['Gradient Descent', 'Newton', 'BFGS', 'DFP'],[dic_gd['loss_line'][-1], dic_nt['loss_line'][-1], dic_bfgs['loss_line'][-1], dic_dfp['loss_line'][-1]]):
    plt.text(pltx, plty+0.002, '{:.3f}'.format(plty), ha='center')
plt.ylim(0.1,0.145)
plt.show()

def time_count(method):
    t = 0
    step = 0
    for seed in range(1023,2023):
        lr = LogisticRegression(10)
        lr.initparam('norm',seed=seed)
        t_s = time.time()
        dic = lr.train(x, y, 500, stop_delta={'criterion':'loss','delta':1e-3}, method=method)
        t_e = time.time()
        t += t_e-t_s
        step += len(dic['delta_line'])
    return t, step

print('start')
print('梯度下降ing')
with HiddenPrint():
    t_gd, step_gd = time_count('graddescent')
print('牛顿ing')
with HiddenPrint():
    t_nt, step_nt = time_count('newton')
print('BFGSing')
with HiddenPrint():
    t_bfgs, step_bfgs = time_count('gfbs')
print('DFPing')
with HiddenPrint():
    t_dfp, step_dfp = time_count('dfp')

plt.bar(x=['Gradient Descent', 'Newton', 'BFGS', 'DFP'], height=[step_gd/1000, step_nt/1000, step_bfgs/1000, step_dfp/1000],
       color=['#20B2AA','#228B22','#FF7F50','#7B68EE'])
plt.xlabel('算法')
plt.ylabel('平均迭代步数')
for pltx, plty in zip(['Gradient Descent', 'Newton', 'BFGS', 'DFP'],[step_gd/1000, step_nt/1000, step_bfgs/1000, step_dfp/1000]):
    plt.text(pltx, plty+0.2, '{:.3f}'.format(plty), ha='center')
plt.show()

plt.bar(x=['Gradient Descent', 'Newton', 'BFGS', 'DFP'], height=[t_gd/1000, t_nt/1000, t_bfgs/1000, t_dfp/1000],
       color=['#20B2AA','#228B22','#FF7F50','#7B68EE'])
plt.xlabel('算法')
plt.ylabel('平均求解时长')
for pltx, plty in zip(['Gradient Descent', 'Newton', 'BFGS', 'DFP'],[t_gd/1000, t_nt/1000, t_bfgs/1000, t_dfp/1000]):
    plt.text(pltx, plty+0.01, '{:.3f}'.format(plty), ha='center')
plt.show()

plt.bar(x=['Gradient Descent', 'Newton', 'BFGS', 'DFP'], height=[t_gd/step_gd, t_nt/step_nt, t_bfgs/step_bfgs, t_dfp/step_dfp],
       color=['#20B2AA','#228B22','#FF7F50','#7B68EE'])
plt.xlabel('算法')
plt.ylabel('平均每步迭代时长')
for pltx, plty in zip(['Gradient Descent', 'Newton', 'BFGS', 'DFP'],[t_gd/step_gd, t_nt/step_nt, t_bfgs/step_bfgs, t_dfp/step_dfp]):
    plt.text(pltx, plty+0.001, '{:.3f}'.format(plty), ha='center')
plt.show()

print('梯度下降预测准确率',(abs(lr_gd.predict(x)-y)<0.5).sum()/x.shape[0])
print('牛顿预测准确率',(abs(lr_nt.predict(x)-y)<0.5).sum()/x.shape[0])
print('bfgs预测准确率',(abs(lr_bfgs.predict(x)-y)<0.5).sum()/x.shape[0])
print('dfp预测准确率',(abs(lr_dfp.predict(x)-y)<0.5).sum()/x.shape[0])