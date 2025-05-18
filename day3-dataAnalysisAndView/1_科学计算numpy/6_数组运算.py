# 数组运算
import numpy as np

# 创建数组
arr = np.arange(1, 6)
print(arr)
# 数组加1
print(arr + 1)
# 数组与数值相乘
print(arr * 3)
# 幂运算
print(arr ** 3)

arr1 = np.array([1, 2, 3, 4])
arr2 = np.array([5, 6, 7, 8])
# 等长数组相加
print(arr1 + arr2)
# 等长数组相乘
print(arr1 * arr2)
# 等长数组整除
print(arr2 // arr1)
# 等长数组的幂运算
print(arr2 ** arr1)

#两个数组对应位置元素的乘积之和
arr1 = np.array([1, 2, 3, 4])
arr2 = np.array([5, 6, 7, 8])
res = np.dot(arr1, arr2)
print(res)
