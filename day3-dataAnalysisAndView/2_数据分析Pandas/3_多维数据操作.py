import numpy as np
import pandas as pd

# 读取csv文件

# df4 = pd.read_csv('bj_lianJia.csv',index_col='ID',encoding='gbk')
# print(df4)

# # 使用head()方法查看前几行数据
# print(df4.head())
# # 使用tail()方法查看最后几行数据
# print(df4.tail())


# 三个属性index、columns和values
# print(df4.index)
# print(df4.columns)
# print(df4.values)
# print(df4['rent'].values)

# print(df4.info())



#
#
#
# # 切片
# print(df4[5:10])
# # 获取某一列
# print(df4['职位'])
# # 获取多列
# print(df4[['职位', '公司']])
# # 获取某几行某几列
# print(df4[['职位', '公司']][:3])
# print(df4[:3][['职位', '公司']])
#
#
# # loc访问器是基于“标签”选择数据
# print(df4.loc[0])   #选择某一行
# print(df4.loc[:,'职位'])    #选择某一列
#
# # 行列混合选择
# print(df4.loc[0:5,['职位','公司','薪资']])  #选择前6行的'职位','公司','薪资'列
# print(df4.loc[[0,3,4], ['职位','公司']])   #选择第0,3,4行的'职位','公司'列
#
#
# df5 = pd.read_excel('sport_data.xlsx', names=['a', 'b', 'c'])
# print(df5)
# # loc可以按照一定条件筛选数据
# print(df5.loc[df5.a > 50, ['a', 'b', 'c']])
# # iloc通过数字选择某些行和列
# print(df5.iloc[0:5, :3])                #选择前5行前3列
# print(df5.iloc[[0,3,4], [0,2]])      #选择第0,3,4行的第0和3列
#
# # at和iat访问器选择某个位置的值
# print(df5.at[0,'c'])  #打印第0行a列对应的单元格的值
# print(df5.iat[0,2])      #结果同上
#
# print(df5[df5.a > 50][['a', 'b', 'c']])