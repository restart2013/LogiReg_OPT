import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# 本文件为例子，展示本作业的编程LogisticRegression如何使用
# 首先导入LogisticRegression
from LogisticReg import *

# 读入数据
df = pd.read_csv('BreastCancerData.csv',encoding='utf-8')
print(df)

# 数据预处理
def std_scale(df,column):
    for col in column:
        df[col] = (df[col]-df[col].mean())/df[col].std()
data=df.copy()
data['Diagnosis'] = data.Diagnosis.map(lambda x:1 if x=='M' else 0)
std_scale(data,data.columns[2:])
x = data.drop(['id','Diagnosis'],axis=1).to_numpy()
y = data.Diagnosis.to_numpy()

# 使用逻辑回归拟合，newton法求解
model = LogisticRegression(10)
model.initparam('norm',seed=2022) # 初始化参数
# 采用wolfe准则自适应步长，停止迭代条件为epoch大于200或者loss下降小于0.0001，打印中间过程
res_dict = model.train(x, y, epoch=200, lr='wolfe', stop_delta={'criterion':'loss','delta':1e-4},
                       method='newton', if_print=True)
# 返回结果为字典格式

# 画图
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False
plt.plot(range(len(res_dict['loss_line'])),res_dict['loss_line'])
plt.xlabel('epoch')
plt.ylabel('loss')
plt.show()

# 模型预测
y_pred = model.predict(x)
# 模型拟合准确率
precission = (np.abs(y_pred-y)<0.5).mean()
print('模型拟合准确率: {}'.format(precission))