import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

plt.rcParams['font.sans-serif'] = ['KaiTi']
plt.rcParams['axes.unicode_minus'] = False

# 生成非线性数据
np.random.seed(0)
x = np.random.uniform(-3, 3, 100)
y = np.sin(x) + 0.2 * np.random.normal(size=len(x))
x = x.reshape(-1, 1)  # 转换为二维数组

# 分割数据集
x_train, x_test, y_train, y_test = train_test_split( x, y, test_size=0.2, random_state=0)

# 创建多项式特征（调整为 degree=3 避免过拟合）
poly = PolynomialFeatures(degree=3)
x_train_poly = poly.fit_transform(x_train)
x_test_poly = poly.transform(x_test)

# 训练模型
model = LinearRegression()
model.fit(x_train_poly, y_train)

# 预测和评估
y_pred = model.predict(x_test_poly)
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
print(f"均方误差(MSE): {mse:.2f}")
print(f"判定系数(R²): {r2:.2f}")

# 可视化结果（使用 linspace 生成有序 x）
x_plot = np.linspace(-3, 3, 100).reshape(-1, 1)
x_plot_poly = poly.transform(x_plot)
y_plot = model.predict(x_plot_poly)

plt.figure('数据可视化')
plt.scatter(x, y, color='blue', label='原始数据')
plt.plot(x_plot, y_plot, color='red', linewidth=2, label='多项式回归')
plt.xlabel('X')
plt.ylabel('y')
plt.title('多项式回归拟合结果')
plt.legend()
plt.show()

# 输出模型参数
print(f"\n模型系数: {model.coef_}")
print(f"截距项: {model.intercept_:.2f}")