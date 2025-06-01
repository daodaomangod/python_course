import pandas as pd

import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler, MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
import matplotlib.pyplot as plt

from sklearn.metrics import confusion_matrix, classification_report, accuracy_score, ConfusionMatrixDisplay

# 1. 读数据
dateSet = pd.read_csv('iris.csv', index_col=0)

print(dateSet.head())

# 2. 提取特征和标签

y = np.array(dateSet['Species'])

x = np.array(dateSet.drop('Species', axis=1))

# 标签编码

ecoding = LabelEncoder()  # 定义编码
y_ecode = ecoding.fit_transform(y)
print(y)

# 3. 划分

scale = MinMaxScaler()
x_scaled = scale.fit_transform(x)

x_train, x_test, y_train, y_test = train_test_split(x_scaled, y_ecode, test_size=0.3, random_state=0)

# print(x_scaled)

# 4. 选择模型
model = SVC()  # 支持向量机

# 5 训练+测试

model.fit(x_train, y_train)

y_pred_test = model.predict(x_test)

# 6. 结果评估 分类报告和混淆矩阵

print(classification_report(y_test, y_pred_test))

cm = confusion_matrix(y_test, y_pred_test)

cm_display = ConfusionMatrixDisplay(cm).plot(cmap='Greens')
plt.show()
