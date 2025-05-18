# 索引访问
import numpy as np

# # 一维数组
# a = np.arange(10)
# print(a)
# print(a[5])
# print(a[-2])
# print(a[:-1])  # 反向切片
# print(a[:3])  # 隔两个取一个
# print(a[:4])
# print(a[:])

# 多维数组
# arr = np.array([[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12]])
# print(arr)
# print(arr[2, :])
# print(arr[:, 1])
# print(arr[1, 2:4])
#
# 数组变化
b = np.arange(15)
print(b)


# c = b.reshape(3, 5)
c = np.reshape(b, (5, 3))
print(c)

d = np.ravel(c)
print(d)
