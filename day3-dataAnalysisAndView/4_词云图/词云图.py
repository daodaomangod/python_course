import jieba
from wordcloud import WordCloud
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image  # PIL库的导入方式更常见

# 文本数据读入
path_text = "Report_2024.txt"  # 文件路径
text = open(path_text, 'r', encoding='utf-8').read()  # 读取全文

# jieba库中文分词
jieba.add_word('中国特色社会主义思想')
jieba.add_word('马克思主义')

words = jieba.lcut(text)  # 精确模式分词
word_str = ' '.join(words)  # 列表转换字符串，使用空格分隔

# 打开图片
img = Image.open('map_mark.png')
# 将图片转化为数组
img_array = np.array(img)
# 去除词库
stopwords = set(['的', '地', '得', '和', '同志们'])
# 创建wc对象
wc = WordCloud(
    width=1024,
    height=1024,
    background_color='white',  # 设置背景颜色
    mask=img_array,  # 设置背景图片
    stopwords=stopwords,
    font_path=r"c:\Windows\Fonts\SIMLI.ttf"  # 设置字体，使用原始字符串
)
# 绘图
wc.generate(word_str)
# 对图像进行处理
plt.imshow(wc, interpolation='bilinear')  # 添加插值方式，使图像更平滑
# 隐藏坐标轴
plt.axis("off")
# 显示图片
plt.show()
# 保存图片到当前文件夹
wc.to_file('report.png')
