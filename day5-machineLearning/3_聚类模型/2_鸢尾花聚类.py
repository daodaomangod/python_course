import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score
from sklearn.decomposition import PCA

# 1. 加载鸢尾花数据集
iris = datasets.load_iris()
x = iris.data  # 特征数据 (150个样本, 4个特征)
y = iris.target  # 真实标签 (用于对比)
feature_names = iris.feature_names
target_names = iris.target_names

# 2. 数据预处理 - 标准化
scaler = StandardScaler()
x_scaled = scaler.fit_transform(x)

# # 3. 使用WCSS法则确定最佳K值
# # 保存每个K值的WCSS(Within-Cluster Sum of Square)
# # WCSS：最小簇内节点平方偏差之和
# wcss = []
# k_range = range(1, 11)
#
# for k in k_range:
#     kmeans = KMeans(n_clusters=k, init='k-means++', random_state=42)
#     kmeans.fit(x_scaled)
#     wcss.append(kmeans.inertia_)  # inertia_属性就是WCSS
#
# # 绘制WCSS法则图
# plt.figure(figsize=(8, 5))
# plt.plot(k_range, wcss, 'bo-')
# plt.xlabel('Number of clusters (K)')
# plt.ylabel('Within-Cluster Sum of Squares (WCSS)')
# plt.xticks(k_range)
# plt.grid()
# plt.show()

# 4. 根据WCSS法则选择K=3进行K-means聚类
k = 4
kmeans = KMeans(n_clusters=k, init='k-means++', random_state=42)
kmeans_labels = kmeans.fit_predict(x_scaled)

# 5. 评估聚类效果 - 轮廓系数
silhouette = silhouette_score(x_scaled, kmeans_labels)
print(f"轮廓系数,当簇K={k}: {silhouette:.3f}")

# 6. 可视化聚类结果
# 使用PCA降维后可视化（更适合多维数据）
pca = PCA(n_components=2)
X_pca = pca.fit_transform(x_scaled)

plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.scatter(X_pca[:, 0], X_pca[:, 1], c=y, cmap='viridis')
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
plt.title('Actual Classes (PCA)')

plt.subplot(1, 2, 2)
plt.scatter(X_pca[:, 0], X_pca[:, 1], c=kmeans_labels, cmap='viridis')
plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1],
            s=200, c='red', marker='*')
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
plt.title(f'K-means Clustering (K={k}, PCA)')

plt.tight_layout()
plt.show()
