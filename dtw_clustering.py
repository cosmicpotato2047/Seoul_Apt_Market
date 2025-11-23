# run_three_methods_ksearch_and_dendrograms.py
import os
import numpy as np
import pandas as pd
from tqdm import tqdm
from scipy.spatial.distance import squareform
from scipy.cluster.hierarchy import linkage, dendrogram, fcluster
from statsmodels.tsa.filters.hp_filter import hpfilter
from sklearn.metrics import silhouette_score
from tslearn.metrics import cdist_dtw
import matplotlib.pyplot as plt
from matplotlib import rc

# ----------------------------
# 설정
# ----------------------------
DATA_DIR = "data"
RESULTS_DIR = "results"
OUTPUT_DIR = "Output"
FIG_DIR = "figure"

for d in [RESULTS_DIR, OUTPUT_DIR, FIG_DIR]:
    os.makedirs(d, exist_ok=True)

# 한글 폰트 (Windows)
rc('font', family='Malgun Gothic')
plt.rcParams['axes.unicode_minus'] = False

K_RANGE = range(2, 10)  # 2..9
HP_LAMB = 129600

# ----------------------------
# 유틸리티: per-series robust scaling
# ----------------------------
def robust_scale_per_series(X):
    # X: ndarray (n_series, series_len)
    Xs = np.zeros_like(X, dtype=float)
    for i in range(X.shape[0]):
        row = X[i].astype(float)
        med = np.nanmedian(row)
        q25 = np.nanpercentile(row, 25)
        q75 = np.nanpercentile(row, 75)
        iqr = q75 - q25
        if (iqr == 0) or np.isnan(iqr):
            # fallback to std
            s = np.std(row)
            if s == 0 or np.isnan(s):
                s = 1.0
            iqr = s
        Xs[i] = (row - med) / iqr
    return Xs

# ----------------------------
# 거리행렬 계산 함수 (tslearn.cdist_dtw 사용)
# ----------------------------
def compute_dtw_distance_matrix_from_df(df, desc="DTW"):
    # df: index=time, columns=series
    X = df.T.values  # shape (n_series, series_len)
    Xs = robust_scale_per_series(X)
    print(f"{desc}: computing DTW (tslearn.cdist_dtw) ...")
    D = cdist_dtw(Xs)  # shape (n, n)
    D = np.nan_to_num(D, nan=0.0)
    maxv = D.max()
    if maxv > 0:
        D = D / maxv
    return D

# ----------------------------
# hierarchical labels helper
# ----------------------------
def hierarchical_labels_from_distance(D, k, method="average"):
    condensed = squareform(D, checks=True)
    Z = linkage(condensed, method=method)
    labels = fcluster(Z, k, criterion="maxclust")
    return labels, Z

def silhouette_precomputed_safe(D, labels):
    try:
        s = silhouette_score(D, labels, metric="precomputed")
    except Exception:
        s = -1.0
    return s

# ----------------------------
# 1) 데이터 로드 & 정렬/동일성 강제
# ----------------------------
print("Loading data...")
panel = pd.read_csv(os.path.join(DATA_DIR, "panel_wide_matrix.csv"), parse_dates=['year_month']).set_index('year_month').sort_index()
idio = pd.read_csv(os.path.join(DATA_DIR, "dfm_idiosyncratic_components.csv"), parse_dates=['year_month']).set_index('year_month').sort_index()

# align to common index
common_idx = panel.index.intersection(idio.index)
panel = panel.loc[common_idx]
idio = idio.loc[common_idx]

gu_list = panel.columns.tolist()
n_gu = len(gu_list)
print(f"Loaded {n_gu} districts x {len(panel)} time points.")

# ----------------------------
# 2) HP-filter cycle 생성
# ----------------------------
print("Computing HP-filter cycles...")
cycle_dict = {}
for gu in tqdm(gu_list, desc="HP filter"):
    cyc, tr = hpfilter(panel[gu].values, lamb=HP_LAMB)
    cycle_dict[gu] = cyc
df_cycle = pd.DataFrame(cycle_dict, index=panel.index)

# save aligned series
panel.to_csv(os.path.join(RESULTS_DIR, "panel_aligned.csv"))
idio.to_csv(os.path.join(RESULTS_DIR, "idiosyncratic_aligned.csv"))
df_cycle.to_csv(os.path.join(RESULTS_DIR, "cycle_hp.csv"))

# ----------------------------
# 3) 거리행렬 계산 (Panel / Idio / HP)
# ----------------------------
D_panel = compute_dtw_distance_matrix_from_df(panel, desc="Panel")
np.save(os.path.join(RESULTS_DIR, "D_panel.npy"), D_panel)
print("Saved data/D_panel.npy")

D_idio = compute_dtw_distance_matrix_from_df(idio, desc="Idiosyncratic")
np.save(os.path.join(RESULTS_DIR, "D_idio.npy"), D_idio)
print("Saved data/D_idio.npy")

D_hp = compute_dtw_distance_matrix_from_df(df_cycle, desc="HP-cycle")
np.save(os.path.join(RESULTS_DIR, "D_hp.npy"), D_hp)
print("Saved data/D_hp.npy")

# ----------------------------
# 4) 각 방법별 k 탐색 (2..9) : silhouette 계산, best k 선택
#    + 덴드로그램 저장
# ----------------------------
methods = {
    "panel": D_panel,
    "idio": D_idio,
    "hp": D_hp
}

for mname, D in methods.items():
    print(f"\nMethod: {mname}")
    rows = []
    best_sil = -999
    best_k = None
    best_labels = None
    best_Z = None

    for k in K_RANGE:
        labels, Z = hierarchical_labels_from_distance(D, k=k, method="average")
        sil = silhouette_precomputed_safe(D, labels)
        rows.append({"k": k, "silhouette": sil})
        print(f"  k={k:2d}  silhouette={sil:.4f}")
        if sil > best_sil:
            best_sil = sil
            best_k = k
            best_labels = labels
            best_Z = Z

    # save k search table
    df_k = pd.DataFrame(rows)
    df_k.to_csv(os.path.join(OUTPUT_DIR, f"{mname}_k_search.csv"), index=False)
    print(f"Method {mname}: best_k={best_k}, best_silhouette={best_sil:.4f}")
    # save best info
    pd.Series({"best_k": best_k, "best_silhouette": best_sil}).to_csv(os.path.join(OUTPUT_DIR, f"{mname}_best_k_summary.csv"))

    # save dendrogram figure (use best_Z to show full tree; annotate best_k by coloring leaves)
    plt.figure(figsize=(12, 6))
    dendrogram(best_Z, labels=gu_list, leaf_rotation=90)
    plt.title(f"{mname.upper()} - Dendrogram (best_k={best_k}, sil={best_sil:.4f})")
    plt.tight_layout()
    figpath = os.path.join(FIG_DIR, f"{mname}_dendrogram_bestk{best_k}.png")
    plt.savefig(figpath, dpi=300)
    plt.close()
    print(f"Saved dendrogram: {figpath}")

print("\nAll done. Check Output/ and figure/ for results.")

# - HP
# 강남 용산 서초 송파 양천
# 성북 도봉 관악 중 강서 마포 노원 은평
# 구로 강동 서대문

# - IDIO
# 강남 서초 송파 양천
# 성북 노원 도봉
# 동작 서대문
# 강동 강서 은평 관악 중

# - Panel
# 강남 송파 서초 강서 마포 서대문
# 노원 성북
# 강동 동대문 동작
# 도봉 은평
# 금천 중랑 관악 영등포

# - 반복 적으로 보이는 묶음
# 강남 서초 송파 (양천)
# 노원 성북 도봉 (은평)
# 강서 마포
# 관악 중

# - 개별 움직임을 반복적으로 보이는 곳
# 종로 영등포 강북 금천 중랑 (성동) (광진) (동대문)