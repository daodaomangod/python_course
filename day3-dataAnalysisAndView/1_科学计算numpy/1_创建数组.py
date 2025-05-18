# 创建数组
import numpy as np

# item = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]
# print(type(item))
# print(item)
#
# # 列表创建一维数组
# a = np.array(item)
# print(type(a))
# print(a)
#
# print(a.ndim)
# print(a.shape)
# print(a.size)
# print(a.dtype)





# # 列表创建多维数组
# a = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
# print(a)
# print(type(a))
# print(a.ndim)
# print(a.shape)
# print(a.size)
# print(a.dtype)



# # 元组创建数组
# b = np.array((1, 2, 3, 4))
# b = np.array(((1, 2, 3), (4, 5, 6)))
# print(b)
# print(type(b))
# #


# # arange创建数组
# # c = np.arange(10)
# c = np.arange(1, 10, 2)
# print(c)
# print(type(c))



# # linspace创建数组
# d = np.linspace(0, 1, 10)
# print(d)
# print(type(d))



# 创建全是0或1的数组
print(np.zeros(5))
print(np.zeros((2, 3)))
print(np.ones(5))
print(np.ones((4, 5)))

# 创建空数组
print(np.empty(5))
print(np.empty((2, 5)))

# 创建对角数组
print(np.diag([1, 1, 1, 1]))
print(np.identity(4))
