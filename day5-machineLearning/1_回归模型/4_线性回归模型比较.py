import matplotlib.pyplot as plt
from sklearn import linear_model
import numpy as np

# 1. 创建数据
np.random.seed(0)
x_train = np.random.rand(100, 1)
y_train = 3 * np.squeeze(x_train) + np.random.normal(0, 0.3, 100)

# 3. 模型选择：实例化线性回归模型
model1 = linear_model.LinearRegression()
model2 = linear_model.Lasso()
model3 = linear_model.Ridge()

# 4. 模型训练
model1.fit(x_train, y_train)
model2.fit(x_train, y_train)
model3.fit(x_train, y_train)

# 回归方程的截距 b
b1, b2, b3 = model1.intercept_, model2.intercept_, model3.intercept_
print(b1, b2, b3)
# 回归方程的斜率 k
k1, k2, k3 = model1.coef_, model2.coef_, model3.coef_
print(k1, k2, k3)

# 建立回归方程
y1 = k1 * x_train + b1  # 线性回归
y2 = k2 * x_train + b2  # lasso回归
y3 = k3 * x_train + b3  # 岭回归
# 绘图
plt.figure('结果比较')
plt.scatter(x_train, y_train)
plt.plot(x_train, y1, color='r',label = 'Liner')
plt.plot(x_train, y2, color='b',label = 'Lasso')
plt.plot(x_train, y3, color='g',label = 'Ridge')
plt.xlabel('x')
plt.ylabel('y')
plt.legend()
plt.show()
