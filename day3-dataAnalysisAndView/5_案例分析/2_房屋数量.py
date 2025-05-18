import pandas as pd
import matplotlib.pyplot as plt

plt.rcParams['font.sans-serif'] = ['KaiTi']
plt.rcParams['axes.unicode_minus'] = False

# 读取csv数据
df = pd.read_csv('newbj_lianJia.csv', encoding='gbk')
# print(df.head())
# 按照城区进行分组统计房屋数量
df1 = df.groupby('district')['ID'].count()
print(df1)
# 计算城区名
region = df1.index.tolist()
print(region)
# 计算各城区的房屋数量
count = df1.values.tolist()
print(count)
# 绘制各城区房屋数量柱状图
plt.figure('各城区房屋数量', figsize=(10, 7))
plt.bar(region, count, width=0.6, color='red')

plt.ylabel('数量', fontsize=14)
plt.xlabel('城区', fontsize=14)
plt.title('各城区房屋数量分布柱状图', fontsize=20)

plt.xticks(rotation=30)
plt.show()
