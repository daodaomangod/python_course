import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier

# 1. 读取数据
df = pd.read_csv('data_horse.csv', encoding='gbk')
# print(df.head())

# 2.载入特征和标签集
x = np.array(df.drop('y', axis=1))
y = np.array(df['y'])

# 3. 训练集和测试集,70%用于训练，30%用于测试
scaler = MinMaxScaler()
x_scaled = scaler.fit_transform(x)

x_train, x_test, y_train, y_test = train_test_split(x_scaled, y, test_size=0.2, random_state=10)
print(f"训练集大小: {x_train.shape[0]}, 测试集大小: {x_test.shape[0]}")

# 4. 实例化分类模型
# model = SVC() #支持向量机
# model = RandomForestClassifier()  # 随机深林
model = AdaBoostClassifier()
# model  = GradientBoostingClassifier()

# 5. 训练模型
model.fit(x_train, y_train)  # 使用训练数据训练 K 近邻分类器模型

# 6. 预测测试集
y_pred = model.predict(x_test)  # 使用训练好的模型对测试数据进行预测，得到预测标签

# 7. 评估模型
print(f"正确率: {accuracy_score(y_test, y_pred)}")  # 计算并打印模型在测试集上的准确率
print("分类报告:\n", classification_report(y_test, y_pred))  # 打印分类报告，包括精确率、召回率、F1 分数等

# 8. 混淆矩阵--可视化
cm = confusion_matrix(y_test, y_pred)

labels = ['A', 'B']
cm_display = ConfusionMatrixDisplay(cm, display_labels=labels).plot(cmap='Greens')
plt.title("Confusion Matrix")
plt.show()
