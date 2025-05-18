import matplotlib.pyplot as plt
import numpy as np

plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

# 设置两个画布
plt.figure(1)
plt.figure('图片2', figsize=(5, 5), facecolor='b')

# 一个画布两个子画布
plt.subplot(121)
plt.grid(color='b', linestyle='-')
plt.subplot(122)
plt.grid(axis='y', color='b', linestyle='-.')


plt.figure('图片1')
# 设置标题
plt.title('正弦函数值', fontsize=16)
# 设置坐标轴名称
plt.xlabel('x数值', fontsize=18)
plt.ylabel('函数数值', fontsize=18)
plt.xticks(np.arange(0, 5, 0.5), fontsize=14, color='red')
plt.grid(axis='y', color='b', linestyle='-.')
# plt.savefig('fig.png')
plt.show()
