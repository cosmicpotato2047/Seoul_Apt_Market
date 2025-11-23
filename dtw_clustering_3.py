import pandas as pd
import numpy as np
from statsmodels.tsa.filters.hp_filter import hpfilter
from tslearn.metrics import cdist_dtw
from scipy.cluster.hierarchy import linkage, dendrogram, fcluster
from sklearn.metrics import silhouette_score
import matplotlib.pyplot as plt
import os

os.makedirs("figure", exist_ok=True)
os.makedirs("output", exist_ok=True)

from matplotlib import rc

rc('font', family='Malgun Gothic')
plt.rcParams['axes.unicode_minus'] = False

# -----------------------------
# 1. 데이터 불러오기
# -----------------------------
df_panel = pd.read_csv("data/panel_wide_matrix.csv", parse_dates=['year_month'])
df_panel.set_index('year_month', inplace=True)
gu_list = df_panel.columns.tolist()

# -----------------------------
# 2. HP Filter로 cycle 추출
# -----------------------------
cycle_matrix = []
lamb = 129600  # 12개월 단위 월간 데이터 기준, 필요 시 조정
for gu in gu_list:
    cycle, trend = hpfilter(df_panel[gu], lamb=lamb)
    cycle_matrix.append(cycle.values)

cycle_matrix = np.array(cycle_matrix)  # shape: n_gu x n_time

# -----------------------------
# 3. DTW 거리 계산
# -----------------------------
print("Computing DTW distance matrix on HP-filtered cycles...")
dtw_dist = cdist_dtw(cycle_matrix)  # shape: n_gu x n_gu
pd.DataFrame(dtw_dist, index=gu_list, columns=gu_list).to_csv("output/dtw_distance_matrix_hp.csv")
print("DTW distance matrix saved -> output/dtw_distance_matrix_hp.csv")

# -----------------------------
# 4. 계층적 클러스터링
# -----------------------------
linkage_matrix = linkage(dtw_dist, method='ward')
plt.figure(figsize=(12,6))
dendrogram(linkage_matrix, labels=gu_list, leaf_rotation=90)
plt.title("Dendrogram (HP-filtered cycles + DTW)")
plt.tight_layout()
plt.savefig("figure/dendrogram_hp_dtw.png", dpi=300)
plt.show()

# -----------------------------
# 5. 최적 클러스터 개수 탐색 (Silhouette)
# -----------------------------
best_k = 2
best_score = -1
for k in range(2, 7):  # 2~6개 권역 후보
    labels = fcluster(linkage_matrix, k, criterion='maxclust')
    score = silhouette_score(dtw_dist, labels, metric='precomputed')
    print(f"k={k}, Silhouette={score:.3f}")
    if score > best_score:
        best_score = score
        best_k = k

print(f"Best k by Silhouette: {best_k}, score={best_score:.3f}")
cluster_labels = fcluster(linkage_matrix, best_k, criterion='maxclust')

# -----------------------------
# 6. 클러스터별 평균 시계열 생성 및 시각화
# -----------------------------
plt.figure(figsize=(12,6))
for c in range(1, best_k+1):
    idx = np.where(cluster_labels == c)[0]
    cluster_cycle_avg = cycle_matrix[idx].mean(axis=0)
    plt.plot(df_panel.index, cluster_cycle_avg, label=f'Cluster {c} (n={len(idx)})')
plt.title("Cluster Average Cycles (HP-filtered)")
plt.xlabel("Year")
plt.ylabel("Cycle")
plt.legend()
plt.tight_layout()
plt.savefig("figure/cluster_avg_cycles_hp.png", dpi=300)
plt.show()
