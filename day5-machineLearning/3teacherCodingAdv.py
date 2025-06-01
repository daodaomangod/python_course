import pandas as pd
import numpy as np

import seaborn as sns
import matplotlib.pyplot as plt
from numpy.f2py.cb_rules import cb_map
# 归一化 ，标准控件
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error

# 决策树回归
from sklearn.tree import DecisionTreeRegressor
from sklearn.svm import SVR  # 支持向量机
from sklearn.neighbors import KNeighborsRegressor  # KNN回归
from sklearn.ensemble import RandomForestRegressor, AdaBoostRegressor, GradientBoostingRegressor

# 1. 读取数据
# 0代表第一列，‘’代表具体字段
pd = pd.read_csv('data_advertising.csv', index_col=0)

# 看一下前五行
print(pd.head(5))
print(pd.info())

# 2. 数据相关性分析

print(pd.corr())

# 热力图，annot打开值显示
# sns.heatmap(pd.corr(),annot=True,cmap='BuGn')
# plt.show()
# 按照区域自动画图 kind代表模式


# 3 特征值处理
x = np.array(pd[['TV', 'radio', 'newspaper']])

#  0代表x轴剔除，1代表y轴剔除
# x =np.array(pd.drop('sales',axis=1))

y = np.array(pd['sales'])

# print(y)


# 4 对数据进行数据归一化处理
# 设置区间范围，默认0-1
scale = MinMaxScaler(feature_range=(0, 1))
x_scale = scale.fit_transform(x)

# 5 训练集和测试集划分


x_train, x_test, y_train, y_test = train_test_split(x_scale, y, test_size=0.3, random_state=0)

# print(x_train.shape)

# 6. 选择模型

# model = LinearRegression()  # 线性模型

model = GradientBoostingRegressor()
# 7. 训练模型

model.fit(x_train, y_train)

# 8. 预测

# reshape -1代表旋转90度，行转列 ，列转行
x_vail = np.array([[180, 50, 10], [120, 50, 10], [220, 80, 30]]).reshape(-1, 3)
x_vail_scaled = scale.fit_transform(x_vail)
print(x_vail_scaled)
y_vail = model.predict(x_vail_scaled)
print(y_vail)
y_pred = model.predict(x_test)
y_pred_train = model.predict(x_train)

# 9. 结果评估

mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
print(f'测试结果：MAE：{mae:.3f},MSE={mse:.3f},R2={r2:.3f}')

# 10 结果可视化
# 真实值和预测值比较
plt.figure()
plt.subplots_adjust(wspace=0.5)
plt.subplot(131)
plt.scatter(range(len(y_train)), y_train, label='True')
# 预测值
plt.plot(range(len(y_pred_train)), y_pred_train, c='r', label='Predict')

plt.scatter(range(len(y_pred_train)), y_pred_train, c='b')
plt.title('Training Set')
plt.ylabel('sales')
plt.xlabel('test number')
plt.legend()

# 测试集
plt.subplot(132)
plt.scatter(range(len(y_test)), y_test, label='True')
# 预测值
plt.plot(range(len(y_pred)), y_pred, c='r', label='Predict')
plt.title('Test Set')

plt.scatter(range(len(y_pred)), y_pred, c='b')
plt.ylabel('sales')
plt.xlabel('test number')
plt.legend()

plt.figure()
plt.scatter(y_test, y_pred)
plt.plot([0, 30], [0, 30], 'r--')
plt.title('Testing set')
plt.xlim(0, 30)
plt.ylim(0, 30)
plt.show()
