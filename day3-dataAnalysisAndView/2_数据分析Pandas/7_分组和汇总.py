import pandas as pd

# 读取csv数据
df = pd.read_csv('bj_lianJia.csv', encoding='gbk')
# print(df.head())

#按照城区进行分组统计房屋数量
df= df.groupby('district')['ID'].count()
print(df)


# 按照地区的房价平均值
data_mean1 = df.groupby(by=['district'])['rent'].mean()
# data_mean1 = df.groupby(by=['district'])['rent'].mean().round(2)# 四舍五入保留2位小数
# data_mean1 = df.groupby(by=['district'])['rent'].mean().round(2).sort_values()
print(data_mean1)
# 按照地区和有无电梯的房价平均值
data_mean2 = df.groupby(by=['district', 'lift'])['rent'].mean()
print(data_mean2)
#采取不同的聚合方式
data_mean3 = df.groupby(by=['district'])['rent'].agg(['min','max','mean','median','sum','std','count'])
print(data_mean3)