import numpy as np


# 逻辑回归
class LogisticRegression:
    def __init__(self, dimension):
        self.w = np.zeros(dimension)
        self.b = np.zeros(1)
        self.d = dimension

    def initparam(self, mod='zero', seed=2022):
        # 参数初始化
        if mod == 'zero':
            self.w = np.zeros(self.d)
        elif mod == 'norm':
            if seed:
                np.random.seed(seed)
            self.w = np.random.normal(0, 1 / np.sqrt(self.d), (self.d,))
        else:
            self.w = mod

    def newton_method(self, x, y, lr):
        # 牛顿法
        g_w, g_b = self.gradient(x, y)
        h_w, h_b = self.hessian(x)
        h_w_inv = np.linalg.inv(h_w)
        if lr == 'wolfe':
            wolfe_lr = self.wolfe(x, y, -np.dot(h_w_inv, g_w), -g_b / h_b)
            return wolfe_lr * np.dot(h_w_inv, g_w), wolfe_lr * g_b / h_b
        else:
            self.w -= lr * np.dot(h_w_inv, g_w)
            self.b -= lr * g_b / h_b
            return lr * np.dot(h_w_inv, g_w), lr * g_b / h_b

    def gradient_descent(self, x, y, lr):
        # 梯度下降法
        g_w, g_b = self.gradient(x, y)
        if lr == 'wolfe':
            wolfe_lr = self.wolfe(x, y, -g_w, -g_b)
            return wolfe_lr * g_w, wolfe_lr * g_b
        else:
            self.w -= lr * g_w
            self.b -= lr * g_b
            return lr * g_w, lr * g_b

    def gfbs(self, x, y, lr, g_w_last, g_b_last, G_w, G_b, delta_w, delta_b):
        # GFBS算法
        g_w, g_b = self.gradient(x, y)
        delta_g_w, delta_g_b = g_w - g_w_last, g_b - g_b_last
        G_w = (np.identity(self.d) - np.outer(delta_w, delta_g_w) / np.dot(delta_w, delta_g_w)) @ G_w @ (
                    np.identity(self.d) - np.outer(delta_w, delta_g_w) / np.dot(delta_w, delta_g_w)).T + np.outer(
            delta_w, delta_w) / np.dot(delta_w, delta_g_w)
        G_b = delta_b / delta_g_b
        if lr == 'wolfe':
            wolfe_lr = self.wolfe(x, y, -np.dot(G_w, g_w), -G_b * g_b)
            return g_w, g_b, G_w, G_b, -wolfe_lr * np.dot(G_w, g_w), -wolfe_lr * G_b * g_b
        else:
            self.w -= lr * np.dot(G_w, g_w)
            self.b -= lr * G_b * g_b
            return g_w, g_b, G_w, G_b, -lr * np.dot(G_w, g_w), -lr * G_b * g_b

    def dfp(self, x, y, lr, g_w_last, g_b_last, G_w, G_b, delta_w, delta_b):
        # DFP算法
        g_w, g_b = self.gradient(x, y)
        delta_g_w, delta_g_b = g_w - g_w_last, g_b - g_b_last
        G_w = G_w + np.outer(delta_w, delta_w) / np.dot(delta_w, delta_g_w) - (
                    G_w @ np.outer(delta_g_w, delta_g_w) @ G_w) / (delta_g_w.T @ G_w @ delta_g_w)
        G_b = delta_b / delta_g_b
        if lr == 'wolfe':
            wolfe_lr = self.wolfe(x, y, -np.dot(G_w, g_w), -G_b * g_b)
            return g_w, g_b, G_w, G_b, -wolfe_lr * np.dot(G_w, g_w), -wolfe_lr * G_b * g_b
        else:
            self.w -= lr * np.dot(G_w, g_w)
            self.b -= lr * G_b * g_b
            return g_w, g_b, G_w, G_b, -lr * np.dot(G_w, g_w), -lr * G_b * g_b

    def gradient(self, x, y):
        # 求一阶导
        g_w = np.zeros(self.d)
        g_b = np.zeros(1)
        for x_, y_ in zip(x, y):
            g_w += (1 / (1 + np.exp(-np.dot(self.w, x_) - self.b)) - y_) * x_
            g_b += (1 / (1 + np.exp(-np.dot(self.w, x_) - self.b)) - y_)
        g_w, g_b = g_w / x.shape[0], g_b / x.shape[0]
        return g_w, g_b

    def hessian(self, x):
        # 求hessian矩阵
        h_w = np.zeros((self.d, self.d))
        h_b = np.zeros(1)
        for x_ in x:
            h_w += np.exp(-np.dot(self.w, x_) - self.b) / ((1 + np.exp(-np.dot(self.w, x_) - self.b)) ** 2) * np.outer(
                x_, x_)
            h_b += np.exp(-np.dot(self.w, x_) - self.b) / ((1 + np.exp(-np.dot(self.w, x_) - self.b)) ** 2)
        h_w, h_b = h_w / x.shape[0], h_b / x.shape[0]
        return h_w, h_b

    def wolfe(self, x, y, dwk, dbk, c1=0.3, c2=0.7, lr_min=0, lr_max=1000, max_search_cnt=10000):
        # 区间搜索实现wolfe准则
        lr_min = lr_min
        lr_max = lr_max
        lr = 1
        wk = self.w.copy()
        bk = self.b.copy()
        ypred = self.predict(x)
        loss = self.logloss(y, ypred)
        g_w, g_b = self.gradient(x, y)
        i = 1
        while True:
            self.w += lr * dwk
            self.b += lr * dbk
            ypred_ = self.predict(x)
            loss_ = self.logloss(y, ypred_)
            g_w_, g_b_ = self.gradient(x, y)
            if loss_ <= loss + c1 * lr * (np.dot(g_w, dwk) + np.dot(g_b, dbk)):
                if (np.dot(g_w_, dwk) + np.dot(g_b_, dbk)) >= c2 * (np.dot(g_w, dwk) + np.dot(g_b, dbk)):
                    break
                else:
                    self.w = wk.copy()
                    self.b = bk.copy()
                    lr_min = lr
                    lr = lr_min + (lr_max - lr_min) / 10
                    # 最大search停止
                    if i > max_search_cnt:
                        break
            else:
                self.w = wk.copy()
                self.b = bk.copy()
                lr_max = lr
                lr = lr_min + (lr_max - lr_min) * 9 / 10
            i += 1
        return lr

    def train(self, x, y, epoch, lr=1, stop_delta={'criterion': 'param', 'delta': 1e-2}, method='newton',
              if_print=False):
        param_line = []
        delta_line = []
        loss_line = []
        param_line.append((self.w, self.b))
        ypred = self.predict(x)
        loss = self.logloss(y, ypred)
        loss_line.append(loss)
        for ep in range(epoch):
            # 牛顿法训练
            if method == 'newton':
                delta_w, delta_b = self.newton_method(x, y, lr)
            # 梯度下降训练
            if method == 'graddescent':
                delta_w, delta_b = self.gradient_descent(x, y, lr)
            # gfbs
            if method == 'gfbs':
                if ep == 0:
                    g_w_last, g_b_last = self.gradient(x, y)
                    G_w, G_b = self.hessian(x)
                    G_w, G_b = np.linalg.inv(G_w), 1 / G_b
                    delta_w, delta_b = self.newton_method(x, y, lr)
                    delta_w, delta_b = -delta_w, -delta_b
                else:
                    g_w_last, g_b_last, G_w, G_b, delta_w, delta_b = self.gfbs(x, y, lr, g_w_last, g_b_last, G_w, G_b,
                                                                               delta_w, delta_b)
            # dfp
            if method == 'dfp':
                if ep == 0:
                    g_w_last, g_b_last = self.gradient(x, y)
                    G_w, G_b = self.hessian(x)
                    G_w, G_b = np.linalg.inv(G_w), 1 / G_b
                    delta_w, delta_b = self.newton_method(x, y, lr)
                    delta_w, delta_b = -delta_w, -delta_b
                else:
                    g_w_last, g_b_last, G_w, G_b, delta_w, delta_b = self.dfp(x, y, lr, g_w_last, g_b_last, G_w, G_b,
                                                                              delta_w, delta_b)
            param_line.append((self.w, self.b))
            ypred = self.predict(x)
            loss = self.logloss(y, ypred)
            loss_line.append(loss)
            delta_param = np.sqrt(np.sum(np.square(np.concatenate([delta_w, delta_b]))))
            delta_line.append(delta_param)
            if if_print:
                print('now epoch is {},loss is {},delta is {}'.format(ep + 1, loss, delta_param))
            # 停止条件
            if stop_delta['criterion'] == 'param':
                if delta_param <= stop_delta['delta']:
                    print('early stop, stop epoch is {}'.format(ep + 1))
                    break
            elif stop_delta['criterion'] == 'loss':
                if loss_line[-2] - loss <= stop_delta['delta']:
                    print('early stop, stop epoch is {}'.format(ep + 1))
                    break
        print('train finished!')
        return {'param_line': param_line, 'delta_line': delta_line, 'loss_line': loss_line}

    def predict(self, x):
        # 模型预测
        return 1 / (1 + np.exp(-np.dot(x, self.w) - self.b))

    def logloss(self, ytrue, ypred):
        l = 0
        for yt, yp in zip(ytrue, ypred):
            if yt:
                l -= np.log(yp)
            else:
                l -= np.log(1 - yp)
        return l / ytrue.shape[0]