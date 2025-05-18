# 统计运算
import numpy as np

arr = np.arange(12).reshape(3, 4)
print(arr)


print(np.sum(arr))  # 求和
print(np.mean(arr))  # 求均值
print(np.std(arr))  # 标准差
print(np.var(arr))  # 方差
print(np.max(arr))  # 求最大值
print(np.mean(arr, axis=1))  # 计算每一行的平均值
print(np.mean(arr, axis=0))  # 计算每一列的平均值
print(np.cumsum(arr))  # 对所有元素累计求和
#
# arr = np.array([[1, 2], [3, 4], [5, 6]])
# print(arr)
# print(np.cumprod(arr))  # 对所有元素累计求积
