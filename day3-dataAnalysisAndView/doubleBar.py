import matplotlib.pyplot as plt
import numpy as np

# 设置中文字体，确保中文正常显示
plt.rcParams["font.family"] = ["SimHei", "WenQuanYi Micro Hei", "Heiti TC"]

# 示例数据
categories = ['产品A', '产品B', '产品C', '产品D', '产品E']
values1 = [25, 30, 35, 20, 28]  # 第一组数据（如2023年销量）
values2 = [35, 25, 30, 15, 32]  # 第二组数据（如2024年销量）

# 设置图形大小
plt.figure(figsize=(10, 6))

# 设置柱状图宽度和位置
bar_width = 0.35
index = np.arange(len(categories))

# 绘制双柱状图
plt.bar(index, values1, bar_width, label='2023年销量', color='skyblue')
plt.bar(index + bar_width, values2, bar_width, label='2024年销量', color='lightcoral')

# 添加标题和标签
plt.title('2023年与2024年产品销量对比')
plt.xlabel('产品类别')
plt.ylabel('销量（单位：千件）')

# 设置x轴刻度位置和标签
plt.xticks(index + bar_width/2, categories)

# 添加图例
plt.legend()

# 显示网格线
plt.grid(axis='y', linestyle='--', alpha=0.7)

# 显示图形
plt.tight_layout()
plt.show()