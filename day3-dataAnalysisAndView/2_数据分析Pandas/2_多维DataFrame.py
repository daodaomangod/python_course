import pandas as pd
import numpy as np

# 使用字典创建
dict = {
    '区域': ['海淀', '房山', '顺义', '大兴'],
    '街道': ['清河', '良乡', '后沙峪', '西红门'],
    '房租': [8500, 3500, 6900, 3800]
}
# df1 = pd.DataFrame(dict, index=['a', 'b', 'c', 'd'])
df1 = pd.DataFrame(dict)
print(df1)

print(df1['房租'][1:3])




# print(df1[['街道','房租']])






#
# 使用ndarray对象创建
# df2 = pd.DataFrame(np.arange(12).reshape(3, 4))
# print(df2)

# # 使用ndarray对象创建
# data = np.arange(12).reshape(3, 4)
# df3 = pd.DataFrame(data, index=np.arange(3), columns=['a', 'b', 'c', 'd'])
# print(df3)
#
# 读取csv文件
# df4=pd.read_csv('data_job.csv')
# print(df4)
#
# df5 = pd.read_csv('data_job.csv',encoding='utf-8',usecols=[0,1,3,4])
# print(df5[5:10])
#
# df6 = pd.read_csv('hr_data.csv',encoding='gbk')
# print(df6)
#
# 读取excel文件
# df7=pd.read_excel('sport_data.xlsx')
# print(df7)
#
#
