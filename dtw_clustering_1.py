import pandas as pd
import numpy as np
from tslearn.metrics import cdist_dtw
from scipy.cluster.hierarchy import linkage, fcluster, dendrogram
from sklearn.metrics import silhouette_score
import matplotlib.pyplot as plt
import matplotlib.cm as cm

# -----------------------------
# 데이터 불러오기
# -----------------------------
panel = pd.read_csv("data/panel_wide_matrix.csv", parse_dates=['year_month'])
panel.set_index('year_month', inplace=True)

# 25개 구 이름
gu_list = panel.columns.tolist()
series_matrix = panel.values.T  # shape: (n_gu, n_time)

# -----------------------------
# DTW 거리 행렬 계산
# -----------------------------
print("DTW 거리 행렬 계산 중...")
distance_matrix = cdist_dtw(series_matrix)  # shape: (25, 25)

# -----------------------------
# Hierarchical Clustering
# -----------------------------
linkage_matrix = linkage(distance_matrix, method='average')

# -----------------------------
# 최적 클러스터 개수 탐색 (Silhouette score)
# -----------------------------
max_clusters = 10  # 최대 클러스터 개수
best_score = -1
best_k = 2  # 최소 2개 클러스터부터
sil_scores = []

for k in range(2, max_clusters + 1):
    labels = fcluster(linkage_matrix, k, criterion='maxclust')
    score = silhouette_score(distance_matrix, labels, metric='precomputed')
    sil_scores.append(score)
    if score > best_score:
        best_score = score
        best_k = k

print(f"최적 클러스터 개수: {best_k}, Silhouette score: {best_score:.3f}")

# -----------------------------
# 최적 k로 클러스터링
# -----------------------------
cluster_labels = fcluster(linkage_matrix, best_k, criterion='maxclust')
cluster_dict = {gu: cluster_labels[i] for i, gu in enumerate(gu_list)}

# -----------------------------
# Dendrogram 시각화
# -----------------------------
plt.figure(figsize=(12, 6))
dendrogram(linkage_matrix, labels=gu_list, leaf_rotation=90)
plt.title("Hierarchical Clustering Dendrogram")
plt.ylabel("Distance")
plt.show()

# -----------------------------
# 시계열 + 클러스터 색상 시각화
# -----------------------------
colors = cm.get_cmap('tab10', best_k)
plt.figure(figsize=(12, 6))

for i, gu in enumerate(gu_list):
    plt.plot(panel.index, panel[gu], label=f"{gu} (C{cluster_dict[gu]})",
             color=colors(cluster_dict[gu]-1), alpha=0.7)

plt.title(f"25개 구 시계열 (클러스터: {best_k}개)")
plt.xlabel("Year")
plt.ylabel("Normalized Real Price Index")
plt.legend(loc='upper left', bbox_to_anchor=(1, 1))
plt.tight_layout()
plt.show()
