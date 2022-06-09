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
# 次梯度
lr_sg = LogisticRegression_L1(10,0.01)
lr_sg.initparam('norm')
dic_sg = lr_sg.train(x, y,200,stop_delta={'criterion':'param','delta':5e-3},method='subgrad')
# 近似点梯度
lr_pg = LogisticRegression_L1(10,0.01)
lr_pg.initparam('norm')
dic_pg = lr_pg.train(x, y,200,stop_delta={'criterion':'param','delta':5e-3},method='proxgraddescent')
# 加速近似点/fista
lr_fista = LogisticRegression_L1(10,0.01)
lr_fista.initparam('norm')
dic_fista = lr_fista.train(x, y,200,stop_delta={'criterion':'param','delta':5e-3},method='fista')

plt.plot(range(len(dic_sg['loss_line'])),dic_sg['loss_line'], color='#5F9EA0')
plt.plot(range(len(dic_pg['loss_line'])),dic_pg['loss_line'], color='#00BFFF')
plt.plot(range(len(dic_fista['loss_line'])),dic_fista['loss_line'], color='#4682B4')
plt.xlim(-5,60)
plt.legend(['次梯度下降', '近似点梯度下降', '加速近似点/Fista'])
plt.xlabel('epoch')
plt.ylabel('logLoss')
plt.title('L1正则,alpha=0.01')
plt.show()

# 次梯度
lr_sg = LogisticRegression_L1(10,0.01)
lr_sg.initparam('norm')
dic_sg = lr_sg.train(x, y,200,method='subgrad')
# 近似点梯度
lr_pg = LogisticRegression_L1(10,0.01)
lr_pg.initparam('norm')
dic_pg = lr_pg.train(x, y,200,method='proxgraddescent')
# 加速近似点/fista
lr_fista = LogisticRegression_L1(10,0.01)
lr_fista.initparam('norm')
dic_fista = lr_fista.train(x, y,200,method='fista')

plt.bar(x=['次梯度下降', '近似点梯度下降', '加速近似点/Fista'], height=[dic_sg['loss_line'][-1], dic_pg['loss_line'][-1], dic_fista['loss_line'][-1]],
       color=['#5F9EA0','#00BFFF','#4682B4'])
plt.xlabel('算法')
plt.ylabel('平均收敛loss/精度')
for pltx, plty in zip(['次梯度下降', '近似点梯度下降', '加速近似点/Fista'],[dic_sg['loss_line'][-1], dic_pg['loss_line'][-1], dic_fista['loss_line'][-1]]):
    plt.text(pltx, plty+0.002, '{:.3f}'.format(plty), ha='center')
plt.ylim(0.1,0.165)
plt.show()

def time_count(method):
    t = 0
    step = 0
    for seed in range(1023,2023):
        lr = LogisticRegression_L1(10,0.01)
        lr.initparam('norm',seed=seed)
        t_s = time.time()
        dic = lr.train(x, y, 500, stop_delta={'criterion':'loss','delta':1e-3}, method=method)
        t_e = time.time()
        t += t_e-t_s
        step += len(dic['delta_line'])
    return t, step

print('start')
print('次梯度ing')
with HiddenPrint():
    t_sg, step_sg = time_count('subgrad')
print('近似点梯度ing')
with HiddenPrint():
    t_pg, step_pg = time_count('proxgraddescent')
print('加速近似点ing')
with HiddenPrint():
    t_fista, step_fista = time_count('fista')
print('finish')

plt.bar(x=['次梯度下降', '近似点梯度下降', '加速近似点/Fista'], height=[step_sg/1000, step_pg/1000, step_fista/1000],
       color=['#5F9EA0','#00BFFF','#4682B4'])
plt.xlabel('算法')
plt.ylabel('平均迭代步数')
for pltx, plty in zip(['次梯度下降', '近似点梯度下降', '加速近似点/Fista'],[step_sg/1000, step_pg/1000, step_fista/1000]):
    plt.text(pltx, plty+0.2, '{:.3f}'.format(plty), ha='center')
plt.show()

plt.bar(x=['次梯度下降', '近似点梯度下降', '加速近似点/Fista'], height=[t_sg/1000, t_pg/1000, t_fista/1000],
       color=['#5F9EA0','#00BFFF','#4682B4'])
plt.xlabel('算法')
plt.ylabel('平均收敛时长')
for pltx, plty in zip(['次梯度下降', '近似点梯度下降', '加速近似点/Fista'],[t_sg/1000, t_pg/1000, t_fista/1000]):
    plt.text(pltx, plty+0.001, '{:.3f}'.format(plty), ha='center')
plt.show()

plt.bar(x=['次梯度下降', '近似点梯度下降', '加速近似点/Fista'], height=[t_sg/step_sg, t_pg/step_pg, t_fista/step_fista],
       color=['#5F9EA0','#00BFFF','#4682B4'])
plt.xlabel('算法')
plt.ylabel('平均每步迭代时长')
for pltx, plty in zip(['次梯度下降', '近似点梯度下降', '加速近似点/Fista'],[t_sg/step_sg, t_pg/step_pg, t_fista/step_fista]):
    plt.text(pltx, plty+0.0005, '{:.3f}'.format(plty), ha='center')
plt.show()