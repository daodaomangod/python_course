import numpy as np
import pandas as pd

a=['海淀', '房山', '顺义', '大兴']
print(a)

# 自动生成索引
s1 = pd.Series(['海淀', '房山', '顺义', '大兴'])
# s1 = pd.Series(a)
print(s1)

# 自定义索引
s2 = pd.Series(['海淀', '房山', '顺义', '大兴'], index=['a', 'b', 'c', 'd'])
print(s2)
a=['海淀', '房山', '顺义', '大兴']
index=['a', 'b', 'c', 'd']
s2 = pd.Series(a, index=index)
print(s2)

# # 使用字典创建Series数组
# dict = {'a1': '海淀', 'b2': '房山', 'c3': '顺义', 'd4': '大兴'}
# s3 = pd.Series(dict)
# print(s3)

# 使用ndarray对象创建Series数组
# s4 = pd.Series(np.random.randint(0, 20, 5), index=np.arange(5))
# print(s4)
#
# 数据访问
# s2 = pd.Series(['海淀', '房山', '顺义', '大兴'], index=['a', 'b', 'c', 'd'])
# print(s2.index)
# print(s2.values)
# print(s2['a'])  # 使用自定义索引访问数组的值
# print(s2[['a', 'c']])  # 使用自定义索引对数组进行切片
# print(s2[1])  # 使用自动索引访问数组的值, 不推荐
# print(s2[1:3])  # 使用自动索引对数组进行切片
# print(s2.get('a'))  # 使用get函数访问数组的值，类似字典操作
#
# 数据运算
# s8 = pd.Series(np.random.randint(0, 100, 5), index=['a', 'b', 'c', 'd', 'e'])
# print(s8)
# print(max(s8))  # 使用Python内置函数
# print(s8 / 2)  # 使用运算符，运算对象是Series的values，运算结果是Series类型
# print(np.floor(s8 / 2))  # 使用Numpy提供的向下取整函数，运算对象是Series的values
# print(s8.median())  # 使用Series对象提供的求中值函数
# print(s8[s8 > 50])  # 数据筛选
#
# 对齐操作
# x = np.array([1, 2, 3, 4])  # x变量是ndarray类型
# y = np.array([5, 6, 7])  # y变量是ndarray类型
# print(x + y)

# x = pd.Series([1, 2, 3, 4])  # x变量是Series类型
# y = pd.Series([5, 6, 7])  # y变量是Series类型
# print(x + y)
