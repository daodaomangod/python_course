from sklearn.datasets import make_blobs
import matplotlib.pyplot as plt

from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score  #轮廓系数
# 1. 数据
x ,y= make_blobs(
    n_samples=100,
    n_features=2,
    centers=4,  # 中心数量
    cluster_std=1,
    shuffle=True,
    random_state=0
)
# plt.scatter(x[:,0],x[:,1],c=y)
# plt.show()

#2. 聚类模型

model=KMeans(
    n_clusters=3, #簇种类
    init='k-means++', #初始点生成
    max_iter=100, #最大迭代次数
    random_state=10
)

#3. 训练模型

model.fit(x,y)

#4. 测试
y_pred = model.predict(x)

#5. 评估
# 轮廓系数，1是最好的，-1是最差的。0是重叠
sil = silhouette_score(x,y_pred)
print(f'轮廓系数={sil:.3f}')

#6. 结果可视化
plt.scatter(x[:,0],x[:,1],c=y,s=80,cmap='viridis')
center = model.cluster_centers_   #通过模型调用中心点坐标
plt.scatter(center[:,0],center[:,1],c='r',s=300,marker='>')
print(center)

plt.show()






