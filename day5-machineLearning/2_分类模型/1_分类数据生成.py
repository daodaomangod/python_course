import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, classification_report

# 1. 生成三分类数据
x, y = make_classification(
    n_samples=300,  # 样本数量
    n_features=2,  # 特征数量
    n_informative=2,  # 有信息的特征数量
    n_redundant=0,  # 冗余特征数量
    n_repeated=0,  # 重复特征数量
    n_classes=3,  # 类别数量(设置为3)
    n_clusters_per_class=1,  # 每个类的簇数
    weights=None,  # 类别权重
    flip_y=0.05,  # 随机噪声比例
    class_sep=1.2,  # 类别分离度(值越大越容易分类)
    random_state=10
)
# # 可视化数据
# plt.figure(figsize=(10, 6))
# scatter = plt.scatter(x[:, 0], x[:, 1], c=y, s=50, cmap='plasma')
# # 自定义图例标签
# legend_labels = ['A', 'B', 'C']
# legend = plt.legend(scatter.legend_elements()[0],  # 获取默认图例元素
#                     legend_labels,  # 替换为 A, B, C
#                     title="species",  # 图例标题
#                     loc="upper right"  # 图例位置
#                     )
# plt.xlabel('Feature1', fontsize=12)
# plt.ylabel('Feature2', fontsize=12)
# plt.show()

# 2. 划分训练集和测试集
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=42)

# 3. 创建并训练逻辑回归模型
model = LogisticRegression()
# model = DecisionTreeClassifier()
# model = SVC()

model.fit(x_train, y_train)

# 4. 预测与评估
y_pred = model.predict(x_test)

# 准确率
accuracy = accuracy_score(y_test, y_pred)
print(f"准确率: {accuracy:.4f}")

# # 分类报告（精确率、召回率、F1等）
# print("\n分类报告:")
# print(classification_report(y_test, y_pred))

# # 混淆矩阵
# cm = confusion_matrix(y_test, y_pred)
# print("\n混淆矩阵:")
# print(cm)
#
# cm_display = ConfusionMatrixDisplay(cm,display_labels = ['A', 'B', 'C']).plot(cmap = 'Greens')
# plt.title("Confusion Matrix")
# plt.show()

# 5. 结果可视化
# 创建网格，判断网格每一个坐标点的类别，用于绘制决策边界
h = 0.02  # 步长
x_min, x_max = x[:, 0].min() - 1, x[:, 0].max() + 1
y_min, y_max = x[:, 1].min() - 1, x[:, 1].max() + 1
xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
print(xx.shape)

# ravel()方法将数组维度拉成一维数组,
# numpy.c_ 用于连接两个矩阵
x_mesh = np.c_[xx.ravel(), yy.ravel()]
# 预测整个网格每一个坐标点的类别
Z = model.predict(x_mesh)
Z = Z.reshape(xx.shape)
# print(Z)

# # 绘制决策边界, pcolormesh 绘制二维数据的彩色网格图
plt.pcolormesh(xx, yy, Z, cmap='GnBu', shading='auto', alpha=0.8)

# # 绘制训练数据点
scatter = plt.scatter(x_train[:, 0], x_train[:, 1], c=y_train, marker='s', cmap='plasma', s=50, label='train_set')

# # 绘制测试数据点
# plt.scatter(x_test[:, 0], x_test[:, 1], c=y_pred, marker='^', cmap='cool', s=100, label='test_set')

plt.xlabel("Feature1")
plt.ylabel("Feature2")
plt.xlim(xx.min(), xx.max())
plt.ylim(yy.min(), yy.max())
plt.legend()
plt.show()
