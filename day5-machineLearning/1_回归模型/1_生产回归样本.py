from sklearn import datasets
import matplotlib.pyplot as plt

# 创建回归数据
x, y = datasets.make_regression(n_samples=50, n_features=1, n_targets=1,
                                noise=5, bias=10, random_state=0)

print(x)
print(y)
print(type(x))
print(x.shape, y.shape)

# 绘图
plt.scatter(x,y)
plt.xlabel('x')
plt.ylabel('y')
plt.show()