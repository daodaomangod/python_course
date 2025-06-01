
from sklearn import datasets
import matplotlib.pyplot as plt
from sklearn.model_selection import  train_test_split
from sklearn import linear_model
from sklearn.metrics import mean_squared_error,r2_score
import joblib

# 1. 生成数据
x, y = datasets.make_regression(
    n_samples=100,  # 样本数
    n_features=1,  # 特征
    n_targets=1, # 目标值，标签
    noise=5, # 噪声，越大与分散
    bias=10, # 偏置
    random_state= 10

)
#2. 划分训练集和测试集
x_train, x_test, y_train, y_test = train_test_split(x,y,test_size=0.3,random_state=0)

# print(len(x_train))
# print(type(x_train))
# print(x_train.shape)

# 3. 模型实例化
# model = linear_model.LinearRegression()
# model = linear_model.Lasso()
model = linear_model.Ridge()

# 4. 模型训练
model.fit(x_train,y_train)

joblib.dump(model,'model.joblib')


# k = model.coef_
# b = model.intercept_
# y1 = k*x+b

# 5. 模型测试
y_pred_test= model.predict(x_test)
y_pred_train = model.predict(x_train)
y_all = model.predict(x)

# # 6. 模型评估
mes = mean_squared_error(y_test,y_pred_test)
r2 = r2_score(y_test,y_pred_test)

print(f'MSE ={mes:.2f},R2 = {r2:.3f}')

#
plt.subplot(131)
plt.scatter(x_train,y_train,label = 'Ture')
plt.plot(x_train,y_pred_train, color = 'r',label ='Predict')
plt.title('training set')
plt.legend()

plt.subplot(132)
plt.scatter(x_test,y_test,label = 'Ture')
plt.plot(x_test,y_pred_test, color = 'r',label ='Predict')
plt.title('testing set')
plt.legend()

plt.subplot(133)
plt.scatter(x,y,label = 'Ture')
plt.plot(x,y_all, color = 'r',label ='Predict')
plt.title('data set')
plt.legend()

plt.show()