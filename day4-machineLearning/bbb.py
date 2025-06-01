import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# 1， 数据
np.random.seed(0)
x = np.linspace(0, 5, 100)
y = 0.5 * x ** 2 + x + 2 + np.random.normal(0, 1, 100)
x = x.reshape(-1, 1)

# 2. 数据划分
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=0)
# print(x_train.shape)

# 3. 模型
trans = PolynomialFeatures(degree=3, include_bias=True)
x_train_poly = trans.fit_transform(x_train)
x_test_poly = trans.fit_transform(x_test)
x_poly = trans.fit_transform(x)

model = LinearRegression()  # 线性模型

# 4. 训练
model.fit(x_train_poly, y_train)

# 5. 测试
y_pred_test = model.predict(x_test_poly)
y_pred_train = model.predict((x_train_poly))
y_all = model.predict(x_poly)

# 6. 模型评估
mes = mean_squared_error(y_test, y_pred_test)
r2 = r2_score(y_test, y_pred_test)
print(f'MSE = {mes}, R2 = {r2}')

# 7. 数据可视化
plt.figure('可视化', figsize=(8, 6))
plt.subplots_adjust(wspace=0.3, hspace=0.5)

plt.subplot(131)
plt.scatter(x_train, y_train, label='Ture')
plt.scatter(x_train, y_pred_train, color='r', label='Predict')
plt.title('training set')
plt.xlabel('x')
plt.ylabel('y')
plt.legend()

plt.subplot(132)
plt.scatter(x_test, y_test, label='Ture')
plt.scatter(x_test, y_pred_test, color='r', label='Predict')
plt.title('testing set')
plt.xlabel('x')
plt.legend()

plt.subplot(133)
plt.scatter(x, y, label='Ture')
plt.plot(x, y_all, color='r', linewidth=5, label='Predict')
plt.title('data set')
plt.xlabel('x')
plt.legend()

plt.show()
