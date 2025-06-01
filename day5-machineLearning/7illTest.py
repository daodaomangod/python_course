import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler, MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestClassifier,GradientBoostingRegressor

from sklearn.metrics import confusion_matrix, classification_report, accuracy_score, ConfusionMatrixDisplay

# 1. 读取数据
pf = pd.read_csv('data_horse.csv', encoding='gbk')

# 2 转换数据
x = np.array(pf.drop('y', axis=1))
y = np.array(pf['y'])

scale = MinMaxScaler()
x_scaled = scale.fit_transform(x)

# 3. 划分数据
x_train, x_test, y_train, y_test = train_test_split(x_scaled, y,test_size=0.3,random_state=0)

# 4. 选择模型
# 0.759
#model=RandomForestClassifier()
# 准确率=0.7777777777777778
model = SVC()



# 5. 模型训练和验证
model.fit(x_train, y_train)
y_pred_test = model.predict(x_test)

# 6. 结果评估 分类报告和混淆矩阵
acc = accuracy_score(y_test, y_pred_test)
print(f'准确率={acc}')
print('分类报告')
print(classification_report(y_test, y_pred_test))

cm = confusion_matrix(y_test, y_pred_test)

cm_display = ConfusionMatrixDisplay(cm).plot(cmap='Greens')
plt.show()
