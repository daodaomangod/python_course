import pandas as pd
import matplotlib.pyplot as plt

plt.rcParams['font.sans-serif'] = ['KaiTi']
plt.rcParams['axes.unicode_minus'] = False

# 读取csv数据
df = pd.read_csv('newbj_lianJia.csv', encoding='gbk')
# print(df.head())
# 按照地区的房价平均值
data_mean = df.groupby(by=['district'])['rent'].mean()
print(data_mean)
# 获取城区名
region = data_mean.index.tolist()
print(region)
# 获取各城区的平均租金
# rent = [round(x,2) for x in data_mean.values.tolist()]
rent = data_mean.values.round(2).tolist()
print(rent)
# 绘制各城区房屋租金折线图
plt.figure('平均房租', figsize=(8, 6))
plt.plot(region, rent, c='r', marker='o', linestyle='--')

# 设置坐标轴标签文本
plt.xlabel('城区', fontsize=12)
plt.ylabel('租金/元', fontsize=12)
# 设置图形标题
plt.title('各城区房屋平均租金折线图', fontsize=14)
# 设置横坐标字体倾斜角度
plt.xticks(rotation=15)
# 显示图形
plt.show()
