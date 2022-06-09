import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.decomposition import PCA

from LogisticReg import LogisticRegression

# 本文件为3d画图，采用PCA降维数据，可视化算法loss轨迹
# 注意运行本文件需要安装sklearn库
# 数据读取&预处理
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

# 运行算法
pca = PCA(n_components=1)
x_pca = pca.fit_transform(x)
# 梯度下降
model_gd = LogisticRegression(1)
model_gd.initparam('norm')
dic_gd = model_gd.train(x_pca,y,200,stop_delta={'criterion':'loss','delta':1e-3},method='graddescent')
# 牛顿
model_nt = LogisticRegression(1)
model_nt.initparam('norm')
dic_nt = model_nt.train(x_pca,y,200,stop_delta={'criterion':'loss','delta':1e-3},method='newton')


# 画图
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False
# 2d
plt.plot([i[0][0] for i in dic_gd['param_line']], [i[1][0] for i in dic_gd['param_line']])
plt.scatter([i[0][0] for i in dic_gd['param_line']], [i[1][0] for i in dic_gd['param_line']])
plt.plot([i[0][0] for i in dic_nt['param_line']], [i[1][0] for i in dic_nt['param_line']],color='red')
plt.scatter([i[0][0] for i in dic_nt['param_line']], [i[1][0] for i in dic_nt['param_line']],color='red')
plt.legend(['梯度下降','牛顿'])
plt.xlim(-2.5,2.5)
plt.ylim(-2.5,2.5)
plt.xlabel('w')
plt.ylabel('b')
plt.grid(True)
plt.show()
# 3d
ax = plt.subplot(projection='3d')
ax.plot([i[0][0] for i in dic_gd['param_line']], [i[1][0] for i in dic_gd['param_line']],[i for i in dic_gd['loss_line']])
ax.scatter([i[0][0] for i in dic_gd['param_line']], [i[1][0] for i in dic_gd['param_line']],[i for i in dic_gd['loss_line']])
ax.plot([i[0][0] for i in dic_nt['param_line']], [i[1][0] for i in dic_nt['param_line']],[i for i in dic_nt['loss_line']],color='red')
ax.scatter([i[0][0] for i in dic_nt['param_line']], [i[1][0] for i in dic_nt['param_line']],[i for i in dic_nt['loss_line']],color='red')
ax.legend(['梯度下降','牛顿'])
ax.set_xlabel('w')
ax.set_ylabel('b')
ax.set_zlabel('loss')
plt.show()

