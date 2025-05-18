# 数组修改
import numpy as np
arr = np.arange(0, 12, 2)  # 创建一维数组
print(arr)

b = np.append(arr, 12)
print(b)  # 在尾部追加一个元素，返回新数组

c = np.append(arr, [14, 16, 18])
print(c)  # 在尾部追加多个元素，返回新数组

# np.insert(原数组，位置，值，axis = 0行/1列）
d = np.insert(arr, 3, 12)
print(d)  # 在原数组下标为3的位置上插入12，返回新数组

arr[1] = 12
print(arr)  # 使用下标修改数组，原数组发生改变
arr[3:] = 12
print(arr)  # 使用切片修改多个元素值
