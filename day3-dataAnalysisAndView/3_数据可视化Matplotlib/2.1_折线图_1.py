import numpy as np
import matplotlib.pyplot as plt

plt.rcParams['font.sans-serif'] = ['KaiTi']
plt.rcParams['axes.unicode_minus'] = False
# 定义x，y数据
x = np.linspace(0, 10, 50)
y1 = np.sin(x)
y2 = np.cos(x)
# 设置画布
# plt.figure('绘制折线图', figsize=(8, 6))
plt.figure(1, figsize=(8, 6))
# 绘制折线图
# plt.plot(x, y1, color='red', linestyle='-', linewidth=5, label='sin()')
plt.plot(x,y1)
plt.plot(x, y2, color='b', linestyle='dashdot', marker='>', markersize=10, label='cos()')
# 设置标题
plt.title('Python绘制正弦函数图像', fontsize=18)
plt.xlabel('自变量x', fontsize=16)
plt.ylabel('三角函数值', fontsize=16)
plt.xticks(np.arange(0, 11, 1), fontsize=13)
plt.yticks(np.arange(-1, 1.25, 0.25), fontsize=13)
plt.legend()
plt.grid()
# 显示绘图
plt.show()