import matplotlib.pyplot as plt
import numpy as np

plt.rcParams['font.sans-serif'] = ['KaiTi']
plt.rcParams['axes.unicode_minus'] = False
# 定义x，y数据
x = np.array([1, 2, 3, 4, 5, 6, 7, 8])
y = np.array([2, 4, 9, 16, 7, 11, 23, 18])
# 设置画布
plt.figure('绘制散点图', figsize=(6, 5))
# 设置点大小
sizes = np.array([30, 50, 100, 200, 500, 1000, 60, 90])
# 设置点的颜色
colors = np.array(["r", "g", "b", "orange", "purple", "beige", "cyan", "magenta"])
# 绘制散点
plt.scatter(x, y, s=sizes, c=colors)
# 设置标题
plt.title('Python绘制散点图', fontsize=16)
plt.xlabel('自变量', fontsize=16)
plt.ylabel('数据分布', fontsize=16)
plt.xticks(np.arange(0, 10, 1), fontsize=14)
plt.yticks(np.arange(0, 25, 5), fontsize=14)
plt.show()