import pandas  as pd

# 1.  读取文件
df = pd.read_csv('customer_data.csv')

print('数据集的基本信息：')
print(df.info())

# 2. 提取信息
data = df[(df['price']>100) & (df['price']<500)]['price'].mean().round(3)
print('群体的平均消费金额：')
print(data)

