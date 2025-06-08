import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error

plt.rcParams['font.sans-serif'] = ['KaiTi']
plt.rcParams['axes.unicode_minus'] = False

# 1. 读取数据
df = pd.read_csv('boston.csv')
# print(df)
# # 显示数据前5行
# print(df.head())
# # 查看数据是否存在空值，从结果来看数据不存在空值。
# print(df.isnull().sum())
# # 查看数据大小
# print(df.shape)
# # 查看数据的描述信息，每个特征的均值，最大值，最小值等信息。
# print(df.describe())
# print(df.info())

# 相关性检验,计算相关系数
corr = df.corr()
plt.figure('因子分析', figsize=(8, 6))
sns.heatmap(corr, annot=True, fmt='.2f', cmap='Blues')
plt.show()

# 每个特征和目标变量MEDV之间的相关系数
print(corr['MEDV'].sort_values())

# 数据可视化-自变量和因变量关系
plt.figure('房价可视化', figsize=(8, 6))
plt.subplots_adjust(wspace=0.3, hspace=0.3)
# 房东低等收入 vs 房屋房价
plt.subplot(221)
plt.scatter(df['LSTAT'], df['MEDV'])
plt.xlabel('LSTAT')
plt.ylabel('MEDV')
# 住宅房间数量 vs 房屋房价
plt.subplot(222)
plt.scatter(df['RM'], df['MEDV'])
plt.xlabel('RM')
plt.ylabel('MEDV')
# 师生比 vs 房屋房价
plt.subplot(223)
plt.scatter(df['PIRATIO'], df['MEDV'])
plt.xlabel('PIRATIO')
plt.ylabel('MEDV')
# 零售业比例 vs 房屋房价
plt.subplot(224)
plt.scatter(df['INDUS'], df['MEDV'])
plt.xlabel('INDUS')
plt.ylabel('MEDV')
plt.show()

# 2. 定义特征值、目标值
# 分析房屋的’RM’， ‘LSTAT’，'CRIM’ 特征与MEDV的相关性性，所以，将其余不相关特征移除
boston_df = df[['LSTAT', 'PIRATIO', 'RM', 'INDUS', 'MEDV']]
# print(boston_df)

# 特征值features
x = np.array(df.drop('MEDV', axis=1))
# x = np.array(df[['LSTAT', 'PIRATIO', 'RM', 'INDUS']])
# 目标值target
y = np.array(df['MEDV'])

# 训练集和测试集,70%用于训练，30%用于测试
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=10)
# print(x_train.shape, x_test.shape, y_train.shape, y_test.shape)

# 线性回归
model = LinearRegression()
# 使用训练数据进行参数估计
model.fit(x_train, y_train)
# 输出线性回归的系数
print(f'线性回归的系数为:\n w ={model.coef_} \n b = {model.intercept_, 3}')

# 模型预测
y_test_pred = model.predict(x_test)

# 模型评估
mse = mean_squared_error(y_test, y_test_pred)
r2 = r2_score(y_test, y_test_pred)
mae = mean_absolute_error(y_test, y_test_pred)
print(f'均方误差：{mse},决定系数：{r2},绝对误差：{mae}')

# 绘图：真实结果 vs 预测结果
plt.figure('真实 vs 预测', figsize=(8, 6))
plt.scatter(y_test, y_test_pred)  # 绘制真实值和预测值

plt.plot([y.min(), y.max()], [y.min(), y.max()], 'r--')  # 绘制对角线

plt.xlabel('目标值')
plt.ylabel('预测值')
plt.show()
# 真实结果 vs 预测结果
plt.figure(figsize=(8, 6))
plt.scatter(range(len(y_test)), y_test, color='blue', label='真实值')  # 绘制真实值的散点图
plt.plot(range(len(y_test)), y_test_pred, color='red', label='预测值')  # 绘制预测值的折线图
plt.xlabel('测试样本')
plt.ylabel('预测值')
plt.show()
# 绘制绝对误差图
absolute_error = np.abs(y_test - y_test_pred)
# 绘制绝对误差图
plt.figure(figsize=(8, 6))
plt.plot(range(len(absolute_error)), absolute_error, marker='o', linestyle='-', color='b')
plt.xlabel('测试样本')
plt.ylabel('绝对误差')
plt.show()