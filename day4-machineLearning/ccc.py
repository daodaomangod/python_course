import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# 1 读入数据
df = pd.read_csv('boston.csv')
# print(df.head())
# print(df.info())

# 2 相关性分析
cor = df.corr()
# print(cor)
# plt.figure(figsize=(8,6))
# sns.heatmap(cor,annot=True,fmt='.2f',cmap='Blues')
print(cor['MEDV'].sort_values())

# plt.subplot(121)
# plt.scatter(df['LSTAT'],df['MEDV'])
# plt.xlabel('LSTAT')
# plt.ylabel('MEDV')

# plt.subplot(122)
# plt.scatter(df['RM'],df['MEDV'])
# plt.xlabel('RM')
# plt.ylabel('MEDV')
# plt.show()

# 3. 数据处理
data = df[['LSTAT', 'RM', 'PIRATIO', 'INDUS', 'TAX', 'NOX', 'MEDV']]
# print(data)

# x= np.array(data[['LSTAT','RM']])
# x = np.array(data.iloc[:2])
x = np.array(data.drop('MEDV', axis=1))
y = np.array(data['MEDV'])

# print(x)

# 4 划分数据
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=0)

# print(x_train.shape)

# 5. 模型
trans = PolynomialFeatures(degree=3, include_bias=True)
x_train_poly = trans.fit_transform(x_train)
x_test_poly = trans.fit_transform(x_test)
x_poly = trans.fit_transform(x)

model = LinearRegression()  # 线性模型

# 6. 训练
model.fit(x_train_poly, y_train)

# 7. 测试
y_pred_test = model.predict(x_test_poly)
y_pred_train = model.predict((x_train_poly))
y_all = model.predict(x_poly)

# 8. 模型评估
mes_train = mean_squared_error(y_train, y_pred_train)
mes_test = mean_squared_error(y_test, y_pred_test)
mes_all = mean_squared_error(y, y_all)

r2_train = r2_score(y_train, y_pred_train)
r2_test = r2_score(y_test, y_pred_test)
r2_all = r2_score(y, y_all)

print(f'训练集：MSE = {mes_train}, R2 = {r2_train}')
print(f'测试集，MSE = {mes_test}, R2 = {r2_test}')
print(f'数据集：MSE = {mes_all}, R2 = {r2_all}')

# print(range(len(y)))

plt.scatter(range(len(y_test)), y_test, label='Ture')
plt.plot(range(len(y_pred_test)), y_pred_test, c='r', label='Predict')
plt.xlabel('Number')
plt.ylabel('MEDV')
plt.legend()
plt.show()
