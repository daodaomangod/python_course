import joblib
from sklearn import datasets
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Ridge
from sklearn.linear_model import Lasso


from sklearn.metrics import mean_squared_error,r2_score
# 线性回归分析

# 1.生成数据
x, y = datasets.make_regression(
    n_samples=100,  # 样本数
    n_features=1,  # 特征数量
    n_targets=1,  # 目标值，标签 y
    noise=5,      #噪声，越大越分散
    random_state= 10,  #随机状态=0,就第一次随机，以后不随机
    bias= 3, #偏置量
)
#画布
# plt.figure('数据')
# plt.scatter(x,y)
# plt.show()


#2.划分训练集合和测试集合

x_train,x_test,y_train,y_test = train_test_split(
    x,y,
    test_size=0.3,   #测试集合分布
    random_state=0
)


# print(len(x_train))
# print(type(x_test))
# print(x_train.shape)


#3.  模型实例化
line_model =LinearRegression()
# line_model = Ridge()
# line_model =Lasso()

#4. 模型训练
line_model.fit(x_train,y_train)

k = line_model.coef_
b= line_model.intercept_

y1=k*x+b

#看直线
# plt.figure('数据')
# plt.scatter(x,y)
# plt.plot(x,y1,color='red')
# plt.show()

#5. 模型测试

y_pred=line_model.predict(x_test)
y_pred_train =line_model.predict(x_train)
y_pred_all =line_model.predict(x)
#6. 模型评估

mes =mean_squared_error(y_test,y_pred)

r2=r2_score(y_test,y_pred)

print(f'MSE={mes:.2f},R2={r2:.3f}')

plt.figure('测试集')
#训练集
plt.subplot(131)
plt.scatter(x_train,y_train)
plt.plot(x_train,y_pred_train,color='red')
plt.title('training list')
#测试集
plt.subplot(132)
plt.scatter(x_test,y_test)
plt.plot(x_test,y_pred,color='red')
#所有集
plt.subplot(133)
plt.scatter(x,y)
plt.plot(x,y_pred_all,color='red')

#保存模型
joblib.dump(line_model,'modelTestRegression.joblib')





