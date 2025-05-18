import pandas as pd
from wordcloud import WordCloud
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

# 读取文件
df = pd.read_csv('newbj_lianJia.csv', encoding='gbk')
# 读取房屋地区信息到列表
street = df['street']
print(street)
# 列表转换为字符串
string = ' '.join(street)
print(string)
# 打开图片
img = Image.open('background.PNG')
# 将图片转化为数组
img_array = np.array(img)
# 创建wc对象
wc = WordCloud(
    width=1024,
    height=1024,
    background_color='white',  # 设置背景颜色
    mask=img_array,  # 设置背景图片
    font_path="c:\Windows\Fonts\SIMLI.ttf"  # 设置字体
)
# 绘图
wc.generate_from_text(string)
# 对图像进行处理
plt.imshow(wc)
# 隐藏坐标轴
plt.axis("off")
# 显示图片
plt.show()
# 保存图片到当前文件夹
wc.to_file('street.png')

