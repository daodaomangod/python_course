import numpy as np
import matplotlib.pyplot as plt

plt.rcParams['font.sans-serif'] = ['KaiTi']
plt.rcParams['axes.unicode_minus'] = False
# 定义x，y数据
x = np.linspace(0, 10, 50)
y1 = np.sin(x)
y2 = np.cos(x)

# 设置画布
plt.figure('绘制折线图', figsize=(8, 6))
# 绘制第一个图
plt.subplot(121)
plt.plot(x, y1, color='red', linestyle='-', label='sin()')
plt.title('正弦函数图像', fontsize=16)
plt.xlabel('自变量x', fontsize=14)
plt.ylabel('三角函数值', fontsize=14)
plt.legend()
plt.grid()
# 绘制第二个图
plt.subplot(122)
plt.plot(x, y2, color='b', linestyle='dashdot', marker='>', markersize=10, label='cos()')
plt.title('余弦函数图像', fontsize=16)
plt.xlabel('自变量x', fontsize=14)
plt.legend()
plt.grid()
# 显示绘图
plt.show()
