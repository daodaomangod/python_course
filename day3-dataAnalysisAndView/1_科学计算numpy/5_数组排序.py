# 数组排序
import numpy as np
# arr = np.array([5, 7, 3, 8, 9, 2, 4])  # 创建一维数组
# print(arr)
#
# b= np.sort(arr)
# print(b)
#
# c = np.argsort(arr)
# print(c)




arr = np.random.randint(1, 10, (3, 5))  # 创建二维数组
print(arr)

arr.sort(axis=0)  # 横向排序，默认为横向排序，axis=0为纵向排序
print(arr)





# arr = np.array([5, 7, 3, 8, 9, 2, 4])  # 创建一维数组
# arr.argsort()  # 排序
# print(arr.argsort())
#
# arr = np.random.randint(1, 10, (3, 5))  # 创建二维数组
# print(arr)
# arr.sort(axis=1)  # 横向排序，默认为横向排序，axis=0为纵向排序
# print(arr)
#
#
# arr = np.array([3, 5, 1, 6])  # 创建一维数组
# print(arr)
# b = np.argsort(arr)
# print(b)  # 返回排序后元素在原数组中的下标

# #
# arr = np.array([1, 3, 5, 6])
# print(arr.max(), arr.min())
# print(arr.argmax(), arr.argmin())  # 返回最大值和最小值的下标
