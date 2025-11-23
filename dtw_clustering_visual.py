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
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import IsolationForest
from scipy.signal import welch
import ruptures as rpt
import stumpy
import seaborn as sns

# --- 0-2) 디렉토리/환경 설정 ---
DATA_DIR = "data"
RESULTS_DIR = "results"
OUTPUT_DIR = "Output"
FIG_DIR = "figure"
SHAPEFILE_PATH = "data/SEOUL_SIG.shp"

for d in [RESULTS_DIR, OUTPUT_DIR, FIG_DIR]:
    os.makedirs(d, exist_ok=True)

# 한글 폰트 (Windows)
rc('font', family='Malgun Gothic')
plt.rcParams['axes.unicode_minus'] = False

# --- 0-3) 고정값/파라미터 ---
HP_LAMB = 129600  # HP filter lambda (월별 데이터 기준)
RANDOM_STATE = 42  # 재현성을 위한 랜덤 시드

# --- 0-4) 데이터 불러오기 ---
panel_file = os.path.join(DATA_DIR, "panel_wide_matrix.csv")
idio_file  = os.path.join(DATA_DIR, "dfm_idiosyncratic_components.csv")
common_file = os.path.join(DATA_DIR, "dfm_common_factors.csv")

panel = pd.read_csv(panel_file, index_col=0, parse_dates=True)
idiosync = pd.read_csv(idio_file, index_col=0, parse_dates=True)
common = pd.read_csv(common_file, index_col=0, parse_dates=True)

# --- 0-5) 시간/구 목록 ---
time_index = panel.index
gu_list = panel.columns.tolist()
n_gu = len(gu_list)

# --- 0-6) 데이터 정규화 (z-score) ---
series_norm = (panel - panel.mean(axis=0)) / panel.std(axis=0)

# --- 0-7) 수동 클러스터 정의 (사용자가 임의 판단한 그룹) ---
manual_clusters = {
    1: ["강남구", "서초구", "송파구", "양천구"],
    2: ["노원구", "성북구", "도봉구", "은평구"],
    3: ["강서구", "마포구"],
    4: ["관악구", "중구"],
}
manual_solo = ["종로구", "영등포구", "강북구", "금천구", "중랑구", "성동구", "광진구", "동대문구"]

best_k = len(manual_clusters)

cluster_labels = np.zeros(n_gu, dtype=int)
gu_to_cluster_id = {}
for cid, g_list in manual_clusters.items():
    for gu in g_list:
        gu_to_cluster_id[gu] = cid

# 미분류 구는 0으로 남겨두거나, 새로운 클러스터 ID를 부여할 수 있습니다.
# 여기서는 0으로 남겨두고, 플롯에서는 무시되도록 처리합니다.
# 주의: 2번 코드의 for c in range(1, best_k+1): 루프는 1부터 시작하므로,
# 미분류 구(cluster_labels=0)는 처리되지 않습니다.
for i, gu in enumerate(gu_list):
    cluster_labels[i] = gu_to_cluster_id.get(gu, 0)

# 최종 클러스터 결과 DataFrame (섹션 9에서 사용)
cluster_df = pd.DataFrame({'구': gu_list, 'cluster': cluster_labels})

print(f"Manually defined clusters set. best_k = {best_k}")

# ---------------------------
# 4) Cluster-wise average series + plot members (수정)
# ---------------------------
# series_norm: DFM의 특이 성분을 정규화한 값이라고 가정하고 플롯에 사용합니다.
cluster_means = {}
plt.figure(figsize=(14,6))
palette = sns.color_palette("tab10", n_colors=best_k)

for c in range(1, best_k+1):
    # 1. 클러스터 멤버 추출
    # cluster_labels 배열에서 현재 클러스터 ID(c)와 일치하는 인덱스를 찾습니다.
    members_idx = np.where(cluster_labels == c)[0]
    members = [gu_list[i] for i in members_idx]
    
    # 2. 클러스터 멤버 시계열 추출 (수정: DataFrame 접근)
    # series_norm은 DataFrame 형태이므로 열 이름을 사용하여 접근해야 합니다.
    mat = series_norm[members].values # DataFrame.values를 사용해 Numpy 배열로 변환
    
    if mat.size == 0:
        mean_series = np.zeros(len(time_index))
    else:
        # Numpy 배열 mat (Time x Member)에서 축 1 (Member)을 기준으로 평균 계산
        mean_series = np.nanmean(mat, axis=1)
        
    cluster_means[c] = mean_series
    
    # 3. plot members faintly
    for gu in members:
        # DataFrame 접근: series_norm[gu]
        plt.plot(time_index, series_norm[gu], color=palette[c-1], alpha=0.25)
        
    # 4. plot cluster mean bold
    plt.plot(time_index, mean_series, color=palette[c-1], lw=2.3, label=f"Cluster {c} (n={len(members)})")

plt.title("Cluster members (faint) and cluster mean (bold) - normalized idiosync")
plt.xlabel("Year")
plt.ylabel("Normalized value")
plt.legend(loc='upper left', bbox_to_anchor=(1,1))
plt.tight_layout()
plt.savefig(os.path.join(FIG_DIR, "cluster_members_and_means.png"), dpi=200) # 경로 수정
plt.close()
print(f"Cluster members and means saved -> {os.path.join(FIG_DIR, 'cluster_members_and_means.png')}")

# Save cluster means to CSV (unnormalized original scale also)
cluster_mean_df = pd.DataFrame({f"cluster_{c}": cluster_means[c] for c in cluster_means}, index=time_index)
cluster_mean_df.to_csv(os.path.join(OUTPUT_DIR, "cluster_means_norm.csv")) # 경로 수정
print(f"Cluster means saved -> {os.path.join(OUTPUT_DIR, 'cluster_means_norm.csv')}")


# Also create non-normalized cluster means (using idiosync original scale) (수정)
cluster_means_orig = {}
for c in range(1, best_k+1):
    members_idx = np.where(cluster_labels == c)[0]
    if members_idx.size > 0:
        # idiosync (DataFrame)에서 열 인덱스(members_idx)를 사용하여 추출
        cluster_means_orig[c] = idiosync.iloc[:, members_idx].mean(axis=1).values
    else:
        cluster_means_orig[c] = np.zeros(len(time_index))
cluster_means_orig_df = pd.DataFrame({f"cluster_{c}": cluster_means_orig[c] for c in cluster_means_orig}, index=time_index)
cluster_means_orig_df.to_csv(os.path.join(OUTPUT_DIR, "cluster_means_orig.csv")) # 경로 수정


# ---------------------------
# 5) Cluster-wise variance decomposition: (수정)
# ---------------------------
from sklearn.linear_model import LinearRegression
explained_ratios = {}
for c in range(1, best_k+1):
    # Ensure the cluster exists (in case manual clusters don't cover 1..best_k completely)
    if f"cluster_{c}" in cluster_means_orig_df.columns:
        y = cluster_means_orig_df[f"cluster_{c}"].values.reshape(-1,1)
        X = common.iloc[:,0].values.reshape(-1,1) # common factor 1 (F1)
        # simple OLS (no intercept to measure fraction? we'll include intercept)
        reg = LinearRegression().fit(X, y)
        y_pred = reg.predict(X)
        ss_total = np.sum((y - y.mean())**2)
        ss_res = np.sum((y - y_pred)**2)
        r2 = 1 - ss_res/ss_total if ss_total>0 else np.nan
        explained_ratios[c] = float(r2)
    else:
        explained_ratios[c] = np.nan # 클러스터가 없는 경우
        
pd.Series(explained_ratios).to_csv(os.path.join(OUTPUT_DIR, "cluster_common_explained_ratio.csv")) # 경로 수정
print(f"Cluster explained ratios by common factor saved -> {os.path.join(OUTPUT_DIR, 'cluster_common_explained_ratio.csv')}")

# plot explained ratios
plt.figure(figsize=(6,4))
# NaN 값을 제외하고 플롯해야 합니다.
valid_keys = [k for k, v in explained_ratios.items() if not np.isnan(v)]
valid_values = [explained_ratios[k] for k in valid_keys]
plt.bar(valid_keys, valid_values, color=palette[:len(valid_keys)])
plt.xlabel("Cluster")
plt.ylabel("R^2 (cluster mean ~ Factor1)")
plt.title("Fraction of cluster-mean variance explained by common Factor1")
plt.tight_layout()
plt.savefig(os.path.join(FIG_DIR, "cluster_explained_ratio.png"), dpi=200) # 경로 수정
plt.close()

# ---------------------------
# 6) Cluster-wise frequency analysis (Welch + dominant period detection)
# ---------------------------
dominant_periods = {}
for c in range(1, best_k+1):
    if f"cluster_{c}" in cluster_means_orig_df.columns: # 클러스터가 존재하는 경우에만
        series_c = cluster_means_orig_df[f"cluster_{c}"].values
        # optionally HP filter to isolate cycle
        cycle_c, trend_c = hpfilter(series_c, lamb=HP_LAMB)
        freqs_w, psd_w = welch(cycle_c, nperseg=min(len(cycle_c), 256))
        # ignore zero-frequency
        pos = freqs_w>0
        freqs_pos = freqs_w[pos]
        psd_pos = psd_w[pos]
        # dominant frequency -> period in months (assuming monthly data and freq units = cycles per sample)
        if len(freqs_pos)==0:
            dominant_periods[c] = np.nan
        else:
            idx = np.argmax(psd_pos)
            dominant_freq = freqs_pos[idx]
            dominant_period_months = 1.0 / dominant_freq if dominant_freq>0 else np.nan
            dominant_periods[c] = float(dominant_period_months)

# save
pd.Series(dominant_periods).to_csv("output/cluster_dominant_period_months.csv")
print("Cluster dominant periods saved -> output/cluster_dominant_period_months.csv")

# plot PSDs
plt.figure(figsize=(10,6))
for c in range(1, best_k+1):
    series_c = cluster_means_orig_df[f"cluster_{c}"].values
    cycle_c, trend_c = hpfilter(series_c, lamb=HP_LAMB)
    freqs_w, psd_w = welch(cycle_c, nperseg=min(len(cycle_c), 256))
    pos = freqs_w>0
    plt.plot(1.0/freqs_w[pos], psd_w[pos], label=f"Cluster {c} (T~{dominant_periods[c]:.1f} mo)")
plt.xscale('log')
plt.xlabel("Period (months)")
plt.ylabel("Power")
plt.title("Welch PSD of Cluster Means")
plt.legend()
plt.tight_layout()
plt.savefig("figure/cluster_welch_psd.png", dpi=200)
plt.close()

# ---------------------------
# 7) Cluster-wise event analysis (ruptures + stumpy + outliers)
# ---------------------------
event_summary = {}
for c in range(1, best_k+1):
    s = cluster_means_orig_df[f"cluster_{c}"].values
    # ruptures (Binseg rbf)
    algo = rpt.Binseg(model="rbf").fit(s)
    # try k=3..6 candidate; choose elbow-like via cost
    costs = []
    candidates = range(1,6)
    for k in candidates:
        bkps = algo.predict(n_bkps=k)
        costs.append(algo.cost.sum_of_costs(bkps))
    # elbow:
    def elbow_pt(costs):
        pts = np.vstack([range(len(costs)), costs]).T
        first, last = pts[0], pts[-1]
        line = last-first
        ln = line/np.sqrt(np.sum(line**2))
        vec = pts-first
        proj = np.outer(np.sum(vec*ln,axis=1), ln)
        dist = np.sqrt(np.sum((vec-proj)**2, axis=1))
        return np.argmax(dist)+1
    opt_k = elbow_pt(costs)
    bkps = algo.predict(n_bkps=opt_k)
    cp_dates = list(time_index[np.array(bkps[:-1])-1])  # exclude endpoint
    # stumpy (discord via matrix profile)
    m = max(12, int(round(len(s)/6)))  # window heuristic
    try:
        mp = stumpy.stump(s, m)
        discord_idx = int(np.argmax(mp[:,0]))
        discord_date = time_index[discord_idx]
    except Exception:
        discord_idx = None
        discord_date = None
    # outliers (IsolationForest)
    iso = IsolationForest(contamination=0.05, random_state=RANDOM_STATE)
    labs = iso.fit_predict(s.reshape(-1,1))
    outlier_dates = list(time_index[np.where(labs==-1)[0]])
    event_summary[c] = {
        'change_points': [d.strftime("%Y-%m") for d in cp_dates],
        'discord': discord_date.strftime("%Y-%m") if discord_date is not None else None,
        'outliers': [d.strftime("%Y-%m") for d in outlier_dates]
    }

# save event summary
pd.DataFrame.from_dict(event_summary, orient='index').to_csv("output/cluster_event_summary.csv")
print("Cluster event summary saved -> output/cluster_event_summary.csv")

# ---------------------------
# 8) Common factor vs cluster mean comparison
#    - correlation and lag (cross-correlation argmax)
# ---------------------------
cf = common.iloc[:,0].values.flatten()
comp_summary = {}
for c in range(1, best_k+1):
    cm = cluster_means_orig_df[f"cluster_{c}"].values
    # correlation
    corr = np.corrcoef(cf, cm)[0,1]
    # cross-correlation to find lag (max of xcorr)
    xcorr = np.correlate((cm - cm.mean()), (cf - cf.mean()), mode='full')
    lags = np.arange(-len(cf)+1, len(cf))
    best_lag = lags[np.argmax(xcorr)]
    comp_summary[c] = {'corr': float(corr), 'best_lag_months': int(best_lag)}
pd.DataFrame.from_dict(comp_summary, orient='index').to_csv("output/cluster_factor_comparison.csv")
print("Cluster vs common factor comparison saved -> output/cluster_factor_comparison.csv")

# small plot: Factor vs cluster means
plt.figure(figsize=(12,6))
plt.plot(time_index, cf, color='black', lw=2, label='Common Factor (F1, raw)')
for c in range(1, best_k+1):
    plt.plot(time_index, cluster_means_orig_df[f"cluster_{c}"], color=palette[c-1], lw=1.5, label=f"Cluster {c} mean")
plt.title("Common Factor vs Cluster Means")
plt.legend(loc='upper left', bbox_to_anchor=(1,1))
plt.tight_layout()
plt.savefig("figure/common_vs_cluster_means.png", dpi=200)
plt.close()

# ---------------------------
# 9) (Optional) Map visualization if shapefile provided
# ---------------------------
if 'SHAPEFILE_PATH' in locals() and SHAPEFILE_PATH is not None:
    try:
        import geopandas as gpd
        gdf = gpd.read_file(SHAPEFILE_PATH, encoding='cp949')
        # Expect gdf to have a column with gu names matching gu_list. User may need to rename.
        # Merge cluster info
        gdf = gdf.merge(cluster_df, left_on="SIG_KOR_NM", right_on='구')  # <-- user must set 'name_column' correctly
        # plot
        fig, ax = plt.subplots(1,1, figsize=(8,8))
        gdf.plot(column='cluster', categorical=True, cmap='tab10', legend=True, ax=ax, edgecolor='k')
        ax.set_title("Cluster map (gu-level)")
        plt.tight_layout()
        plt.savefig("figure/cluster_map.png", dpi=200)
        plt.close()
        print("Cluster map saved -> figure/cluster_map.png")
    except Exception as e:
        print("Map plotting skipped: could not run geopandas/shapefile step.", e)

print("=== Pipeline finished. Outputs in 'figure/' and 'output/' directories.")
