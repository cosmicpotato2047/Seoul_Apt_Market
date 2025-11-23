import os
os.makedirs("figure", exist_ok=True)
os.makedirs("output", exist_ok=True)

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.cluster.hierarchy import linkage, dendrogram, fcluster
from scipy.spatial.distance import squareform
from sklearn.metrics import silhouette_score
from sklearn.ensemble import IsolationForest
import ruptures as rpt
import stumpy
from scipy.signal import welch
from statsmodels.tsa.filters.hp_filter import hpfilter
import warnings
warnings.filterwarnings("ignore")
from matplotlib import rc

rc('font', family='Malgun Gothic')
plt.rcParams['axes.unicode_minus'] = False

# Try DTW implementations (prefer tslearn, fallback to dtaidistance)
try:
    from tslearn.metrics import cdist_dtw
    DTW_IMPL = "tslearn"
except Exception:
    try:
        from dtaidistance import dtw
        DTW_IMPL = "dtaidistance"
    except Exception:
        raise ImportError("Install tslearn or dtaidistance for DTW functionality.")


# ---------------------------
# User inputs / config
# ---------------------------
PANEL_WIDE_PATH = "data/panel_wide_matrix.csv"
IDIO_PATH = "data/dfm_idiosyncratic_components.csv"
COMMON_FACTOR_PATH = "data/dfm_common_factors.csv"
# Optional shapefile for mapping (set to None if not available)
SHAPEFILE_PATH = "data/SEOUL_SIG.shp"  # e.g. "seoul_gu_shapefile.shp"

# Settings
MAX_CLUSTERS = 8         # silhouette 탐색 상한 (2..MAX_CLUSTERS)
RANDOM_STATE = 42
HP_LAMB = 129600         # HP filter lambda for monthly housing
DTW_STEP_PATTERN = None  # for tslearn you can set global_constraint if desired

# ---------------------------
# 0) Load data
# ---------------------------
panel = pd.read_csv(PANEL_WIDE_PATH, parse_dates=['year_month']).set_index('year_month')
idiosync = pd.read_csv(IDIO_PATH, parse_dates=['year_month']).set_index('year_month')
common = pd.read_csv(COMMON_FACTOR_PATH, parse_dates=['year_month']).set_index('year_month')

# Align time index (intersection)
common = common.reindex(panel.index).interpolate().ffill().bfill()
idiosync = idiosync.reindex(panel.index).interpolate().ffill().bfill()

gu_list = panel.columns.tolist()
n_gu = len(gu_list)
time_index = panel.index

print(f"Loaded data. n_gu={n_gu}, n_time={len(time_index)}")

# Choose series to compare: idiosyncratic or standardized panel
# Using idiosync (common component removed) recommended
series_matrix = idiosync[gu_list].values.T  # shape (n_gu, n_time)

# Optional normalization (z-score) to focus on shape rather than scale
def zscore_each(arr2d):
    out = np.zeros_like(arr2d, dtype=float)
    for i in range(arr2d.shape[0]):
        a = arr2d[i, :]
        if np.std(a) == 0:
            out[i, :] = 0.0
        else:
            out[i, :] = (a - np.mean(a)) / np.std(a)
    return out

series_norm = zscore_each(series_matrix)

# ---------------------------
# 1) DTW distance matrix
# ---------------------------
print("Computing DTW distance matrix...")
if DTW_IMPL == "tslearn":
    # tslearn expects shape (n_ts, sz)
    dist_sq = cdist_dtw(series_norm)  # returns full square matrix
elif DTW_IMPL == "dtaidistance":
    n = series_norm.shape[0]
    dist_sq = np.zeros((n, n))
    for i in range(n):
        for j in range(i+1, n):
            dist = dtw.distance(series_norm[i], series_norm[j])
            dist_sq[i, j] = dist
            dist_sq[j, i] = dist
else:
    raise RuntimeError("No DTW implementation available.")
np.fill_diagonal(dist_sq, 0.0)

# save distance matrix
pd.DataFrame(dist_sq, index=gu_list, columns=gu_list).to_csv("output/dtw_distance_matrix.csv")
print("DTW distance matrix saved -> output/dtw_distance_matrix.csv")

# ---------------------------
# 2) Hierarchical clustering (linkage)
# ---------------------------
# linkage expects condensed distance vector
condensed = squareform(dist_sq, checks=False)
linkage_method = "average"  # average is robust for DTW distances
Z = linkage(condensed, method=linkage_method)

# dendrogram plot
plt.figure(figsize=(14, 5))
dendrogram(Z, labels=gu_list, leaf_rotation=90, color_threshold=None)
plt.title("Dendrogram (DTW distances) - linkage='{}'".format(linkage_method))
plt.ylabel("Distance")
plt.tight_layout()
plt.savefig("figure/dendrogram_dtw.png", dpi=200)
plt.close()
print("Dendrogram saved -> figure/dendrogram_dtw.png")

# ---------------------------
# 3) Choose optimal k via silhouette (precomputed distances)
# ---------------------------
print("Searching optimal cluster count via Silhouette score...")
best_k = 2
best_score = -1.0
scores = {}
for k in range(2, min(MAX_CLUSTERS, n_gu)+1):
    labels = fcluster(Z, k, criterion='maxclust')
    try:
        score = silhouette_score(dist_sq, labels, metric='precomputed')
    except Exception as e:
        score = -1.0
    scores[k] = score
    if score > best_score:
        best_score = score
        best_k = k

print(f"Best k by silhouette: k={best_k}, score={best_score:.4f}")
pd.Series(scores).to_csv("output/silhouette_scores.csv")

# plot silhouette vs k
plt.figure(figsize=(6,4))
plt.plot(list(scores.keys()), list(scores.values()), marker='o')
plt.xlabel("k (number of clusters)")
plt.ylabel("Silhouette score")
plt.title("Silhouette scores for different k")
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig("figure/silhouette_k.png", dpi=200)
plt.close()

# assign cluster labels
cluster_labels = fcluster(Z, best_k, criterion='maxclust')
cluster_df = pd.DataFrame({'구': gu_list, 'cluster': cluster_labels})
cluster_df.to_csv("output/cluster_labels.csv", index=False)
print("Cluster labels saved -> output/cluster_labels.csv")

# ---------------------------
# 4) Cluster-wise average series + plot members
# ---------------------------
cluster_means = {}
plt.figure(figsize=(14,6))
palette = sns.color_palette("tab10", n_colors=best_k)
for c in range(1, best_k+1):
    members = [gu_list[i] for i in range(n_gu) if cluster_labels[i]==c]
    mat = series_norm[[i for i in range(n_gu) if cluster_labels[i]==c], :]
    if mat.size == 0:
        mean_series = np.zeros(len(time_index))
    else:
        mean_series = np.nanmean(mat, axis=0)
    cluster_means[c] = mean_series
    # plot members faintly
    for i, gu in enumerate(members):
        plt.plot(time_index, series_norm[gu_list.index(gu)], color=palette[c-1], alpha=0.25)
    # plot cluster mean bold
    plt.plot(time_index, mean_series, color=palette[c-1], lw=2.3, label=f"Cluster {c} (n={len(members)})")
plt.title("Cluster members (faint) and cluster mean (bold) - normalized idiosync")
plt.xlabel("Year")
plt.ylabel("Normalized value")
plt.legend(loc='upper left', bbox_to_anchor=(1,1))
plt.tight_layout()
plt.savefig("figure/cluster_members_and_means.png", dpi=200)
plt.close()
print("Cluster members and means saved -> figure/cluster_members_and_means.png")

# Save cluster means to CSV (unnormalized original scale also)
cluster_mean_df = pd.DataFrame({f"cluster_{c}": cluster_means[c] for c in cluster_means}, index=time_index)
cluster_mean_df.to_csv("output/cluster_means_norm.csv")
print("Cluster means saved -> output/cluster_means_norm.csv")

# Also create non-normalized cluster means (using idiosync original scale)
cluster_means_orig = {}
for c in range(1, best_k+1):
    members_idx = [i for i in range(n_gu) if cluster_labels[i]==c]
    if members_idx:
        cluster_means_orig[c] = idiosync.iloc[:, members_idx].mean(axis=1).values
    else:
        cluster_means_orig[c] = np.zeros(len(time_index))
cluster_means_orig_df = pd.DataFrame({f"cluster_{c}": cluster_means_orig[c] for c in cluster_means_orig}, index=time_index)
cluster_means_orig_df.to_csv("output/cluster_means_orig.csv")

# ---------------------------
# 5) Cluster-wise variance decomposition:
#    regress cluster mean on common factor -> R^2 as explained ratio
# ---------------------------
from sklearn.linear_model import LinearRegression
explained_ratios = {}
for c in range(1, best_k+1):
    y = cluster_means_orig_df[f"cluster_{c}"].values.reshape(-1,1)
    X = common.values.reshape(-1,1)  # Factor1
    # simple OLS (no intercept to measure fraction? we'll include intercept)
    reg = LinearRegression().fit(X, y)
    y_pred = reg.predict(X)
    ss_total = np.sum((y - y.mean())**2)
    ss_res = np.sum((y - y_pred)**2)
    r2 = 1 - ss_res/ss_total if ss_total>0 else np.nan
    explained_ratios[c] = float(r2)
pd.Series(explained_ratios).to_csv("output/cluster_common_explained_ratio.csv")
print("Cluster explained ratios by common factor saved -> output/cluster_common_explained_ratio.csv")

# plot explained ratios
plt.figure(figsize=(6,4))
plt.bar(list(explained_ratios.keys()), list(explained_ratios.values()), color=palette[:len(explained_ratios)])
plt.xlabel("Cluster")
plt.ylabel("R^2 (cluster mean ~ Factor1)")
plt.title("Fraction of cluster-mean variance explained by common Factor1")
plt.tight_layout()
plt.savefig("figure/cluster_explained_ratio.png", dpi=200)
plt.close()

# ---------------------------
# 6) Cluster-wise frequency analysis (Welch + dominant period detection)
# ---------------------------
dominant_periods = {}
for c in range(1, best_k+1):
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
if SHAPEFILE_PATH is not None:
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
