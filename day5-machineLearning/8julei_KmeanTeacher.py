import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

plt.rcParams['font.sans-serif'] = ['KaiTi']
plt.rcParams['axes.unicode_minus'] = False

#1. 数据
x, y= make_blobs(
    n_samples=100,
    n_features=2,
    centers=4,
    cluster_std=1,
    shuffle=True,
    random_state=0
)
# 2. 聚类模型
model = KMeans(
    n_clusters= 4,  #簇种类
    init='k-means++', #初始点生成
    max_iter=100, # 最大迭代次数
    random_state=10
)
# 3. 训练模型
model.fit(x,y)

# 4. 测试
y_pred= model.predict(x)

# 5. 评估
# 轮廓系数，1是做好的，-1是做差的
sil = silhouette_score(x,y_pred)
print(f'轮廓系数={sil:.3f}')

# 6. 结果可视化
plt.subplot(121)
plt.scatter(x[:,0],x[:,1],c=y,s=80,cmap='viridis')
plt.title('原数据')

plt.subplot(122)
plt.scatter(x[:,0],x[:,1],c=y_pred,s=80,cmap='viridis')

center = model.cluster_centers_  # 中心点坐标
plt.scatter(center[:,0],center[:,1],c='r',s=300,marker='<')
plt.title('聚类数据')
plt.show()