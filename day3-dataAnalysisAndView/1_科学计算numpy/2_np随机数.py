# 随机数
import numpy as np
# 生成随机数
print(np.random.random((5)))
print(np.random.random((3, 3)))

# 生成服从均匀分布的随机数
print(np.random.rand(5))
print(np.random.rand(3, 3))

# 生成服从正态分布的随机数
print(np.random.randn(5))
print(np.random.randn(3, 3))

# 生成随机整数
print(np.random.randint(1, 10, size=[2, 3]))
# random.normal(均值，标准差，size)
print(np.random.normal(2, 0.5, [2, 5]))
