# LogiReg_OPT
基于python实现多种最优化算法求解逻辑回归和带L1正则项的逻辑回归  
  
某高校某学院某最优化课程课程作业的部分内容  
__本部分代码主要为：__
逻辑回归模型和带L1正则项的逻辑回归模型；各类最优化算法求解实现；以及各算法间效率、精度比较  
__代码实现内容包括：__
梯度下降法，Adam算法，经典牛顿法，BFGS算法，DFP算法，wolfe准则自适应步长；次梯度方法，近似点梯度方法，加速近似点方法(FISTA)  
  
__文件说明__  
LogisticReg.py 为逻辑回归以及各类牛顿类算法的主体部分编程实现  
Example1.py 为逻辑回归演示文件，演示如何调用LogisticReg.py文件并运用牛顿类算法求解逻辑回归问题  
Main1.py 为课程论文逻辑回归实验部分代码  
Example2.py 为L1正则逻辑回归演示文件，演示如何调用LogisticReg.py文件并运用牛顿类算法求解逻辑回归问题  
Main2.py 为课程论文L1正则逻辑回归实验部分代码  
3Dplot.py 为算法loss轨迹的3D可视化，采用PCA降维数据，对比梯度下降和经典牛顿法的参数和loss下降过程，运行此文件需要安装sklearn库  
  
可以改进的地方：没有实现L2正则项情况，但L2和无正则项的情况求解方法是一样的；没有实现小批量方法  
