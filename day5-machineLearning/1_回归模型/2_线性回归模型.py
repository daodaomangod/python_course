from sklearn import datasets
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split  # 数据划分
from sklearn import linear_model
from sklearn.metrics import mean_squared_error  # 均方误差mse
from sklearn.metrics import r2_score  # r2
import joblib

# 1. 创建数据
x, y = datasets.make_regression(n_samples=100, n_features=1, n_targets=1,
                                noise=12, bias=10, random_state=0)

# 2. 数据划分
x_train, x_test, y_train, y_test = train_test_split(x, y, random_state=0, test_size=0.3)

print(x_test, y_test)

# 3. 模型选择：实例化线性回归模型
# model = linear_model.LinearRegression()
# model = linear_model.Lasso()
model = linear_model.Ridge()

# 4. 模型训练
model.fit(x_train, y_train)

# # 回归方程的截距 b
# b = model.intercept_
# print(b)
# # 回归方程的斜率 k
# k = model.coef_
# print(k)
# # 建立回归方程
# y1 = k * x_train + b
# # 绘图
# plt.figure('训练')
# plt.scatter(x_train, y_train)
# plt.plot(x_train, y1, color='r')
# plt.xlabel('x')
# plt.ylabel('y')

# 5. 模型测试
y_pred_test = model.predict(x_test)
y_pred_train = model.predict(x_train)
y_all = model.predict(x)

# # 6. 模型评估
mes = mean_squared_error(y_test, y_pred_test)
r2 = r2_score(y_test, y_pred_test)

print(f'MSE ={mes:.2f},R2 = {r2:.3f}')

#
plt.subplot(131)
plt.scatter(x_train, y_train, label='Ture')
plt.plot(x_train, y_pred_train, color='r', label='Predict')
plt.title('training set')
plt.legend()

plt.subplot(132)
plt.scatter(x_test, y_test, label='Ture')
plt.plot(x_test, y_pred_test, color='r', label='Predict')
plt.title('testing set')
plt.legend()

plt.subplot(133)
plt.scatter(x, y, label='Ture')
plt.plot(x, y_all, color='r', label='Predict')
plt.title('data set')
plt.legend()

plt.show()
