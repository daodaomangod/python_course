import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error

from sklearn.linear_model import LinearRegression  # 线性回归
from sklearn.tree import DecisionTreeRegressor  # 决策树回归
from sklearn.svm import SVR  # SVM回归
from sklearn.neighbors import KNeighborsRegressor  # KNN回归
# from sklearn.ensemble import AdaBoostRegressor  # Adaboost回归
from sklearn.ensemble import GradientBoostingRegressor  #Boosting回归

# 1. 读取数据
df = pd.read_csv('data_advertising.csv', index_col=0)
# print(df)
# print(df.head())

# # 2. 相关性分析
# cor = df.corr()
# print(cor)
# sns.heatmap(cor,annot=True,fmt='.2f',cmap='Blues')
# plt.show()

# 3. 数据间可视化
# 多变量关系可视化 sns.pairplot()
# sns.pairplot(df, x_vars=['TV','radio','newspaper'],y_vars=['sales'],height=5,kind='reg')
# plt.show()

# 4. 定义特征和标签
x = np.array(df[['TV', 'radio', 'newspaper']])
y = np.array(df['sales'])

# 5. 划分数据，训练集和测试集,70%用于训练，30%用于测试
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=10)
# print(x_train.shape)

# 6. 模型实例化
# model = LinearRegression()  # 线性回归
# model = DecisionTreeRegressor()  # 决策树回归
# model = SVR()  # SVM回归
# model = KNeighborsRegressor()  # KNN回归
# model = AdaBoostRegressor()  # Adaboost回归
model = GradientBoostingRegressor()   #Boosting回归
# 7. 模型训练
model.fit(x_train, y_train)

# 8. 模型预测-测试集
y_pred_test = model.predict(x_test)  # 测试数据的预测值
y_pred_train = model.predict(x_train)

# 9. 评估指标
mae = mean_absolute_error(y_test, y_pred_test)  # 均方绝对误差
mse = mean_squared_error(y_test, y_pred_test)  # 均方误差
r2 = r2_score(y_test, y_pred_test)  # 决定系数

print(f'模型测试结果：均方绝对误差={mae:.3f},均方误差 = {mse:.3f},决定系数 = {r2:.3f}')

# 10. 数据可视化
plt.figure(figsize=(10, 6))
plt.subplot(121)
plt.scatter(range(len(y_train)), y_train, label="predict")
plt.plot(range(len(y_pred_train)), y_pred_train, 'r', label="test")
plt.xlabel("The number of sales")
plt.ylabel("Value of sales")
plt.legend()

plt.subplot(122)
plt.scatter(range(len(y_test)), y_test, label='Ture')
plt.plot(range(len(y_pred_test)), y_pred_test, color='r', label='Predict')
plt.title('testing set')
plt.xlabel("The number of sales")
plt.ylabel("Value of sales")
plt.legend()

plt.figure()
plt.scatter(y_test, y_pred_test)
# plt.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], 'r--')
plt.plot([0, 30], [0, 30], 'r--')
plt.xlim(0, 30)
plt.ylim(0, 30)
plt.xlabel("Ture")
plt.ylabel("Predict")
plt.show()
