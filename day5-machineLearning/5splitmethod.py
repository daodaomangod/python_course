from sklearn.datasets import make_classification

import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score,precision_score,recall_score,f1_score
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix,ConfusionMatrixDisplay
import seaborn as sns
# 1. 数据生成
x, y = make_classification(
    n_samples=200,  # 样本数量
    n_features=3,  # 样本特征数量
    n_informative=3,  # 信息特征个数
    n_repeated=0,
    n_redundant=0,
    n_classes=3,  # 种类
    n_clusters_per_class=1,
    class_sep=1.5,  # 越大越容易分类
    random_state=0

)

# scatter = plt.scatter(x[:, 0], x[:, 1], c=y, s=80, cmap='plasma')
#
# # 自动生成图例
# label = ['A', 'B', 'C']
# legend = plt.legend(scatter.legend_elements()[0],
#                     label)
# plt.xlabel('Feature 1')
# plt.ylabel('Feature 2')
# plt.show()

# 2 划分数据

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=0)

# 3. 导入模型

model = LogisticRegression()  # 逻辑回归模型

# 4. 训练

model.fit(x_train, y_train)

# 5. 测试
y_pred_test = model.predict(x_test)

# 6.评估
acc =accuracy_score(y_test,y_pred_test)
# recall=recall_score(y_test,y_pred_test)


print(f'测试结果：准确率={acc}')
# print(classification_report(y_test,y_pred_test)) #分类报告

# 混淆矩阵
cm=confusion_matrix(y_test,y_pred_test)
print(cm)
# 两种都可以展示
sns.heatmap(cm,annot=True,cmap='Greens')
#cm_display = ConfusionMatrixDisplay(cm).plot(cmap='Greens')
plt.show()
