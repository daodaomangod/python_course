import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

# 1. 读取数据
df = pd.read_csv('iris.csv')
# print(df.head())
# # 查看数据是否存在空值，从结果来看数据不存在空值。
# print(df.isnull().sum())
# print(df.info())

# 2.载入特征和标签集
x = np.array(df[['Sepal.Length', 'Sepal.Width', 'Petal.Length', 'Petal.Width']])
y = df['Species']
# 对标签集进行编码
encoder = LabelEncoder()
y = encoder.fit_transform(y)

# # 可视化数据
# plt.figure(figsize=(10, 6))
# scatter = plt.scatter(x[:, 0], x[:, 1], c=y, s=50, cmap='plasma')
# # 自定义图例标签
# legend_labels = ['Iris Setosa', 'Iris Versicolour', 'Iris Virginica']
# legend = plt.legend(scatter.legend_elements()[0],  # 获取默认图例元素
#                     legend_labels,  # 替换为 A, B, C
#                     title="Species",  # 图例标题
#                     loc="upper right"  # 图例位置
#                     )
# plt.xlabel('Sepal.Length', fontsize=12)
# plt.ylabel('Sepal.Width', fontsize=12)
# plt.show()

# 3. 训练集和测试集,70%用于训练，30%用于测试
# StandardScaler标准化，均值为 0，标准差为 1 的标准正态分布
# MinMaxScaler归一化，MinMaxScaler(feature_range=(0, 1)) 默认]0,1]之间
scaler = StandardScaler()
x_scaled = scaler.fit_transform(x)
x_train, x_test, y_train, y_test = train_test_split(x_scaled, y, test_size=0.2, random_state=10)
# 查看分割后的数据集大小
print(f"训练集大小: {x_train.shape[0]}, 测试集大小: {x_test.shape[0]}")

# 4. 实例化分类模型
# model=LogisticRegression()  # 逻辑回归
# model = SVC() #支持向量机
model = DecisionTreeClassifier()
# model = KNeighborsClassifier(n_neighbors=3)

# 5. 训练模型
model.fit(x_train, y_train)  # 使用训练数据训练 K 近邻分类器模型

# 6. 预测测试集
y_pred = model.predict(x_test)  # 使用训练好的模型对测试数据进行预测，得到预测标签

# 7. 评估模型
print(f"正确率: {accuracy_score(y_test, y_pred)}")  # 计算并打印模型在测试集上的准确率
print("分类报告:\n", classification_report(y_test, y_pred))  # 打印分类报告，包括精确率、召回率、F1 分数等

# 8. 混淆矩阵--可视化
cm = confusion_matrix(y_test, y_pred)
print("混淆矩阵:\n", cm)  # 打印混淆矩阵，显示真实标签和预测标签之间的关系

labels = ['Setosa', 'Versicolour', 'Virginica']
cm_display = ConfusionMatrixDisplay(cm,display_labels = labels).plot(cmap = 'Greens')
plt.title("Confusion Matrix")
plt.show()
