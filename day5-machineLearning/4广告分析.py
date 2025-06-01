import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.tree import DecisionTreeRegressor  # 决策树
from sklearn.svm import SVR  # 支持向量机回归
from sklearn.neighbors import KNeighborsRegressor  # knn 回归
from sklearn.ensemble import AdaBoostRegressor, GradientBoostingRegressor

# 1. 读入数据
df = pd.read_csv('data_advertising.csv', index_col=0)

# print(df.head())
# print(pd.info())

# 2. 相关性分析
# cor = df.corr()
# print(cor)
# sns.heatmap(cor,annot=True,cmap='BuGn')
# plt.show()
# 单因素数据可视化
# sns.pairplot(pd, x_vars=['TV','radio','newspaper'],y_vars='sales',height=4,kind='reg')
# plt.show()

# 3. 特征定义
x = np.array(df[['TV', 'radio', 'newspaper']])
# x= np.array(df.drop('sales',axis=1))
y = np.array(df['sales'])

# 4. 数据归一化
scale = MinMaxScaler()
x_scaled = scale.fit_transform(x)

# print(x_scaled)

# 5. 训练和测试集划分
x_train, x_test, y_train, y_test = train_test_split(x_scaled, y, test_size=0.3, random_state=0)
# print(x_train.shape)

# 6. 选择模型
# model = LinearRegression() # 线性模型
# model = DecisionTreeRegressor()   # 决策树
# model  =SVR()  # 支持向量机
model = GradientBoostingRegressor()  # 梯度提升回归

# 7. 训练
model.fit(x_train, y_train)

# 8. 预测
y_pred_test = model.predict(x_test)
y_pred_train = model.predict(x_train)

# 9. 结果评估
mae = mean_absolute_error(y_test, y_pred_test)
mse = mean_squared_error(y_test, y_pred_test)
r2 = r2_score(y_test, y_pred_test)

print(f'测试结果：MAE = {mae:.3f},MSE = {mse:.3f}, R2 = {r2:.3f}')

# 10 结果可视化
plt.figure()
plt.subplots_adjust(wspace=0.5)
# 训练集
plt.subplot(121)
plt.scatter(range(len(y_train)), y_train, label='Ture')
plt.plot(range(len(y_pred_train)), y_pred_train, c='r', label='Predict')
plt.title('Training Set')
plt.xlabel('Number')
plt.ylabel("Scale")
plt.legend()

# 测试集
plt.subplot(122)
plt.scatter(range(len(y_test)), y_test, label='Ture')
plt.plot(range(len(y_pred_test)), y_pred_test, c='r', label='Predict')
plt.title('Testing Set')
plt.xlabel('Number')
plt.ylabel("Scale")
plt.legend()

# 图2
plt.figure()
plt.scatter(y_test, y_pred_test)
plt.plot([0, 30], [0, 30], 'r--')
plt.title('Testing Set')
plt.xlim(0, 30)
plt.ylim(0, 30)

plt.show()
