import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import  PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error,r2_score
# 1. 数据
# 0-5生成100个点  seed=0就第一次随机，后面不随机
np.random.seed(0)
x = np.linspace(0, 5, 100)
# 随机分布点
y = 0.5 * x ** 2 + x + 2 + np.random.normal(0,1,100)
print(x.ndim)
# -1代表一行变一列
x = x.reshape(-1,1)

print(x.ndim)
# plt.figure("图")
# plt.scatter(x, y)
# plt.show()


#2.  数据划分

x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.3,random_state=0)

print(x_train.shape)


#创建模型 多项式特征

trans =PolynomialFeatures(
    degree=3, #多项式次数，最高是3次方
    include_bias=True, #包含偏置量

)
#转换 x
x_train_poly=trans.fit_transform(x_train)
x_test_poly=trans.fit_transform(x_test)
x_all=trans.fit_transform(x)

model = LinearRegression()  #创建一个线性模型
# 4 训练
model.fit(x_train_poly,y_train)


# 5 测试

y_pred = model.predict(x_test_poly)
y_pred_train=model.predict(x_train_poly)
y_pred_all = model.predict(x_all)
#6 模型评估

mse =mean_squared_error(y_test,y_pred)

r2 = r2_score(y_test,y_pred)

mse_train =mean_squared_error(y_train[0:30],y_pred)

r2_train = r2_score(y_train[0:30],y_pred)

print(f'MSE={mse:.2f},R2={r2:.3f}')

print(f'mse_train[0:30]={mse_train:.2f},r2_train={r2:.3f}')

#7 可视化
plt.figure("图",figsize=(8,6))
#训练集
#调整间距
plt.subplots_adjust(wspace=0.5,hspace=0.5)
plt.subplot(131)
plt.scatter(x_train,y_train,label="True")
plt.scatter(x_train,y_pred_train,color='red',label="Predict")
plt.title('training set')
plt.xlabel('x')
plt.ylabel('y', rotation=0)
plt.legend()

#测试集
plt.subplot(132)
plt.scatter(x_test,y_test,label="True")
plt.scatter(x_test,y_pred,color='red',label="Predict")
plt.title('testing set')
plt.xlabel('x')
plt.legend()
#所有集
plt.subplot(133)
plt.scatter(x,y,label="True")
plt.scatter(x,y_pred_all,color='red',linewidth=1.5,label="Predict")
plt.title('data set')
plt.xlabel('x')
plt.legend()
plt.show()