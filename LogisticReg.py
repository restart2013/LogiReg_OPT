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

    def adam(self, x, y, lr, m_w, m_b, v_w, v_b, ep, alpha=0.9, beta=0.999, epsilon=1e-9):
        # adam算法
        g_w, g_b = self.gradient(x, y)
        m_w = alpha * m_w + (1 - alpha) * g_w
        m_b = alpha * m_b + (1 - alpha) * g_b
        v_w = beta * v_w + (1 - beta) * (g_w ** 2)
        v_b = beta * v_b + (1 - beta) * (g_b ** 2)
        m_w_hat = m_w / (1 - alpha ** ep)
        m_b_hat = m_b / (1 - alpha ** ep)
        v_w_hat = v_w / (1 - beta ** ep)
        v_b_hat = v_b / (1 - beta ** ep)
        if lr == 'wolfe':
            wolfe_lr = self.wolfe(x, y, -m_w_hat / (np.sqrt(v_w_hat) + epsilon),
                                  -m_b_hat / (np.sqrt(v_b_hat) + epsilon))
            return m_w, m_b, v_w, v_b, wolfe_lr * m_w_hat / (np.sqrt(v_w_hat) + epsilon), wolfe_lr * m_b_hat / (
                        np.sqrt(v_b_hat) + epsilon)
        else:
            self.w -= lr * m_w_hat / (np.sqrt(v_w_hat) + epsilon)
            self.b -= lr * m_b_hat / (np.sqrt(v_b_hat) + epsilon)
            return m_w, m_b, v_w, v_b, lr * m_w_hat / (np.sqrt(v_w_hat) + epsilon), lr * m_b_hat / (
                        np.sqrt(v_b_hat) + epsilon)

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
        param_line.append((self.w.copy(), self.b.copy()))
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
            # adam
            if method == 'adam':
                if ep == 0:
                    m_w, m_b, v_w, v_b = 0, 0, 0, 0
                    m_w, m_b, v_w, v_b, delta_w, delta_b = self.adam(x, y, lr, m_w, m_b, v_w, v_b, 1)
                else:
                    m_w, m_b, v_w, v_b, delta_w, delta_b = self.adam(x, y, lr, m_w, m_b, v_w, v_b, ep + 1)
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
            param_line.append((self.w.copy(), self.b.copy()))
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



#L1正则逻辑回归
class LogisticRegression_L1:
    def __init__(self, dimension, L1=0):
        self.w = np.zeros(dimension)
        self.b = np.zeros(1)
        self.d = dimension
        self.L1 = L1

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

    def subgradient_descent(self, x, y, lr):
        # 次梯度算法
        g_w, g_b = self.subgradient(x, y)
        self.w -= lr * g_w
        self.b -= lr * g_b
        return lr * g_w, lr * g_b

    def proximal_gradient_descent(self, x, y, lr):
        # 近似点梯度算法
        g_w, g_b = self.gradient_without_L1(x, y)
        w, b = self.w.copy(), self.b.copy()
        self.w = self.proximal(w - lr * g_w, lr)
        self.b = self.proximal(b - lr * g_b, lr)
        return w - self.w, b - self.b

    def fista(self, x, y, lr, theta, w_last, b_last):
        # fista或者说加速近似点梯度算法
        theta_new = (np.sqrt(theta ** 4 + 4 * theta ** 2) - theta) / 2
        w, b = self.w.copy(), self.b.copy()
        y_w = w + theta_new * (1 / theta - 1) * (w - w_last)
        y_b = b + theta_new * (1 / theta - 1) * (b - b_last)
        g_w, g_b = self.gradient_without_L1(x, y, True, y_w, y_b)
        self.w = self.proximal(y_w - lr * g_w, lr)
        self.b = self.proximal(y_b - lr * g_b, lr)
        return theta_new, w, b, w - self.w, b - self.b

    def gradient_without_L1(self, x, y, customization=False, w=None, b=None):
        # 舍去L1正则项的一阶导
        g_w = np.zeros(self.d)
        g_b = np.zeros(1)
        for x_, y_ in zip(x, y):
            if customization:
                g_w += (1 / (1 + np.exp(-np.dot(w, x_) - b)) - y_) * x_
                g_b += (1 / (1 + np.exp(-np.dot(w, x_) - b)) - y_)
            else:
                g_w += (1 / (1 + np.exp(-np.dot(self.w, x_) - self.b)) - y_) * x_
                g_b += (1 / (1 + np.exp(-np.dot(self.w, x_) - self.b)) - y_)
        g_w, g_b = g_w / x.shape[0], g_b / x.shape[0]
        return g_w, g_b

    def subgradient(self, x, y):
        # 求次梯度
        g_w = np.zeros(self.d)
        g_b = np.zeros(1)
        for x_, y_ in zip(x, y):
            g_w += (1 / (1 + np.exp(-np.dot(self.w, x_) - self.b)) - y_) * x_
            g_b += (1 / (1 + np.exp(-np.dot(self.w, x_) - self.b)) - y_)
        g_w, g_b = g_w / x.shape[0], g_b / x.shape[0]
        g_w += self.L1_grad(self.w)
        g_b += self.L1_grad(self.b)
        return g_w, g_b

    def L1_grad(self, param):
        # 求次梯度辅助函数，L1正则项梯度
        L1_g = (param > 0).astype(np.float) - (param < 0).astype(np.float)
        L1_g += (L1_g == 0).astype(np.float) * np.random.uniform(-1, 1)
        return L1_g * self.L1

    def proximal(self, x, t):
        # L1正则项近似点算子
        plus = (np.abs(x + t * self.L1) < np.abs(x - t * self.L1)).astype(np.float)
        minus = (np.abs(x + t * self.L1) > np.abs(x - t * self.L1)).astype(np.float) * (-1)
        zero = (x == 0).astype(np.float) * np.random.choice([1, -1])
        return x + (plus + minus + zero) * t * self.L1

    def train(self, x, y, epoch, lr=1, stop_delta={'criterion': 'param', 'delta': 1e-2}, method='subgrad',
              if_print=False):
        param_line = []
        delta_line = []
        loss_line = []
        param_line.append((self.w.copy(), self.b.copy()))
        ypred = self.predict(x)
        loss = self.logloss(y, ypred)
        loss_line.append(loss)
        for ep in range(epoch):
            # 次梯度算法
            if method == 'subgrad':
                delta_w, delta_b = self.subgradient_descent(x, y, lr)
            # 近似点梯度下降
            if method == 'proxgraddescent':
                delta_w, delta_b = self.proximal_gradient_descent(x, y, lr)
            # FISTA算法
            if method == 'fista':
                if ep == 0:
                    theta = 1
                    w_last, b_last = self.w.copy(), self.b.copy()
                    delta_w, delta_b = self.proximal_gradient_descent(x, y, lr)
                else:
                    theta, w_last, b_last, delta_w, delta_b = self.fista(x, y, lr, theta, w_last, b_last)
            # 增广拉格朗日算法
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
            param_line.append((self.w.copy(), self.b.copy()))
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