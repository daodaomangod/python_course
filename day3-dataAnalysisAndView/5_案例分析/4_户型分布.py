import pandas as pd
import matplotlib.pyplot as plt

import matplotlib.font_manager as fm
# 获取所有可用字体
fonts = fm.findSystemFonts()
for font in fonts:
    try:
        font_name = fm.FontProperties(fname=font).get_name()
        print(font_name, "->", font)
    except:
        pass
plt.rcParams["font.family"] = ["SimHei", "WenQuanYi Micro Hei", "Heiti TC", "Arial Unicode MS"]

plt.rcParams['font.sans-serif'] = ['KaiTi']
plt.rcParams['axes.unicode_minus'] = False

# 读取csv数据
df = pd.read_csv('newbj_lianJia.csv', encoding='gbk')
print(df.head())

# 根据房屋户型分组
df1 = df.groupby('model')['ID'].count().sort_values()
print(df1)
# #计算房屋户型数量，排序并取前10名
data_model = df1[-10:]
print(data_model)
# 户型
model = data_model.index.tolist()
# 数量
count = data_model.values.tolist()
# 绘制房屋户型占比饼图
plt.pie(count, labels=model, autopct='%1.2f%%',shadow=True)
plt.title('房屋户型前10名的占比情况', fontsize=18)
plt.show()


