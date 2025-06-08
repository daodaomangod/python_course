import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

# 1. 生成聚类数据（4个簇，共500个样本）
x, y_true = make_blobs(
    n_samples=500,
    n_features=2,
    centers=4,  # 簇数量
    cluster_std=1.8,  # 簇标准差（控制分散程度）
    shuffle=True,
    random_state=10
)
# # 可视化原始数据
# plt.scatter(x[:, 0], x[:, 1], s=20, c='r')
# plt.title("Data")
# plt.xlabel('Feature1')
# plt.ylabel('Feature2')
# plt.show()

# 2. 定义 K-Means模型
kmeans = KMeans(
    n_clusters=3,  # 指定簇数
    init='k-means++',  # 随机选择'random',智能初始化'k-means++'
    max_iter=300,  # 最大迭代次数
    random_state=10
)
# 3. 训练模型
kmeans.fit(x, y_true)

# 4. 预测簇标签
y_pred = kmeans.predict(x)

# 评估聚类效果
# silhouette_score 轮廓系数,最佳为1，最差为-1,重叠是0
silhouette = silhouette_score(x, y_pred)
print(f"轮廓系数: {silhouette:.3f}")

# 5. 绘制聚类结果
plt.figure(figsize=(8,6))
plt.subplot(121)
plt.scatter(x[:,0],x[:,1],c=y_true,s=80,cmap='viridis')
plt.title('Origin data')
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')

plt.subplot(122)
plt.scatter(x[:,0],x[:,1],c=y_pred,s=80,cmap='viridis',label = 'Cluster Data')

center = kmeans.cluster_centers_  # 中心点坐标
plt.scatter(center[:,0],center[:,1],c='r',s=300,marker='*',label = 'Center')
plt.title('K-Means cluster')
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.legend()

plt.show()


