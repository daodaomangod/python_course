import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sympy import rotations
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.neighbors import KNeighborsRegressor
# 梯度提升回归模型
from sklearn.ensemble import GradientBoostingRegressor
# 随机森林
from sklearn.ensemble import RandomForestRegressor

from sklearn.ensemble import BaggingRegressor

# 1 读取数据

df = pd.read_csv('data_advertising.csv')

print(df.head(5))

print(df.info())

# 2 相关性的分析

cor = df.corr()

print(cor)

# 热力图
# plt.figure(figsize=(8,6))
# sns.heatmap(cor,annot=True,fmt='.2f',cmap='Blues')
#
# plt.show()
# 排序
# 负相关变量
print(cor['sales'].sort_values())
# plt.subplot(131)
# plt.scatter(df['LSTAT'], df['MEDV'])
# plt.xlabel("LSTAT")
# plt.ylabel("MEDV", rotation=0)
# # 正相关变量
# plt.subplot(132)
# plt.scatter(df['RM'], df['MEDV'])
# plt.xlabel("RM")
# plt.ylabel("MEDV", rotation=0)
# plt.show()

# 3.  数据处理


data_df = df[['TV','radio','newspaper','sales']]

print(type(data_df))
#  pandas.core.frame.DataFrame转换数据
# x=np.array(data_df[['LSTAT','RM']])
# x=np.array(data_df.iloc[:2])

# 去除一列
x = np.array(data_df.drop('sales', axis=1))

y = np.array(data_df['sales'])

# 4 划分测试数据

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=0)

# print(x_train.shape)

# 5 模型
trans = PolynomialFeatures(
    degree=3,  # 多项式次数，最高是3次方
    include_bias=True,  # 包含偏置量

)
# 转换 x
x_train_poly = trans.fit_transform(x_train)
x_test_poly = trans.fit_transform(x_test)
x_all = trans.fit_transform(x)
'''
训练集mse_train[0:30]=0.13,r2_train=0.995
测试集MSE_test=0.56,R2_test=0.980
所有集mse_all=0.25,r2_all=0.991'''
#model = LinearRegression()  #创建一个线性模型
'''
KNN回归模型
训练集mse_train[0:30]=3.36,r2_train=0.874
测试集MSE_test=4.13,R2_test=0.849
所有集mse_all=3.59,r2_all=0.867'''
#model = KNeighborsRegressor()
'''
随机森林
训练集mse_train[0:30]=0.04,r2_train=0.998
测试集MSE_test=0.33,R2_test=0.988
所有集mse_all=0.13,r2_all=0.995'''
#model=RandomForestRegressor()

'''
梯度提升回归模型
训练集mse_train[0:30]=0.01,r2_train=1.000
测试集MSE_test=0.21,R2_test=0.992
所有集mse_all=0.07,r2_all=0.997'''
model = GradientBoostingRegressor()

'''
Bagging回归模型
训练集mse_train[0:30]=0.07,r2_train=0.997
测试集MSE_test=0.35,R2_test=0.987
所有集mse_all=0.15,r2_all=0.994
'''
#model = BaggingRegressor()

# 6 训练
model.fit(x_train_poly, y_train)

# 7 测试

y_pred = model.predict(x_test_poly)
y_pred_train = model.predict(x_train_poly)
y_pred_all = model.predict(x_all)
# 8 模型评估

mse = mean_squared_error(y_test, y_pred)

r2 = r2_score(y_test, y_pred)

mse_train = mean_squared_error(y_train, y_pred_train)

r2_train = r2_score(y_train, y_pred_train)

mse_all = mean_squared_error(y, y_pred_all)

r2_all = r2_score(y, y_pred_all)
#print(f'model.score:{model.score(x_test,y_test)}')
print(f'训练集mse_train[0:30]={mse_train:.2f},r2_train={r2_train:.3f}')
print(f'测试集MSE_test={mse:.2f},R2_test={r2:.3f}')
print(f'所有集mse_all={mse_all:.2f},r2_all={r2_all:.3f}')

# print(range(len(y)))


# plt.scatter(range(len(y)),y,label='true')

plt.scatter(range(len(y_test)), y_test, label='True')
plt.scatter(range(len(y_pred)), y_pred, c='r', label="Predict")
plt.xlabel('Advertising')
plt.ylabel('sales')
plt.legend()

plt.show()
