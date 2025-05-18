import numpy as np
import matplotlib.pyplot as plt
# # 定义x，y数据
# N = 50
# x = np.random.rand(N)
# y = np.random.rand(N)
# colors = np.random.rand(N)
# area = (30 * np.random.rand(N)) ** 2  # 0 to 15 point radii
# # 绘制散点图，设置颜色及透明度
# plt.scatter(x, y, s=area, c=colors,alpha=0.4)
# # 设置标题
# plt.title("Scatter Test")
# plt.show()


# 定义x，y数据
x = np.array([5,7,8,7,2,17,2,9,4,11,12,9,6])
y = np.array([99,86,87,88,111,86,103,87,94,78,77,85,86])
colors = np.array([0, 10, 20, 30, 40, 45, 50, 55, 60, 70, 80, 90, 100])
# 绘制散点图，条形图
plt.scatter(x, y, c=colors, cmap='plasma')
plt.colorbar()
plt.show()