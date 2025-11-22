import pandas as pd
import numpy as np
import scipy.stats as stats
import statsmodels.api as sm
from statsmodels.stats.diagnostic import acorr_ljungbox, het_breuschpagan
from statsmodels.tsa.stattools import acf, pacf
import matplotlib.pyplot as plt
from linearmodels.panel import PanelOLS, RandomEffects
from sklearn.model_selection import TimeSeriesSplit
from tqdm import tqdm
import warnings
warnings.filterwarnings("ignore")

import matplotlib.pyplot as plt
import os

# figure 폴더 생성
os.makedirs("figure", exist_ok=True)

# ======================
# 0) Load Data
# ======================

# DFM 결과
factors = pd.read_csv("data/dfm_common_factors.csv", index_col=0, parse_dates=True)
idiosync = pd.read_csv("data/dfm_idiosyncratic_components.csv", index_col=0, parse_dates=True)

# 패널 데이터 준비
panel = pd.read_csv("data/panel_wide_matrix.csv", index_col=0, parse_dates=True)
panel.index.name = 'year_month'  # 인덱스 이름 지정
panel_long = panel.reset_index().melt(
    id_vars='year_month', 
    var_name='구', 
    value_name='real_std'
)


# ======================
# 1) Variance Decomposition
# ======================

# 패널 전체 variance (cov 포함)
total_var = panel.values.var()

# idiosyncratic variance (cov 포함)
idiosync_var = idiosync.values.var()

# 설명된 비중
explained_ratio = (total_var - idiosync_var) / total_var

print("\nVariance decomposition:")
print(f"Total variance: {total_var:.4f}")
print(f"Idiosyncratic variance: {idiosync_var:.4f}")
print(f"Explained by common factors: {explained_ratio*100:.2f}%\n")

idiosync_ratio = idiosync_var / total_var

# -----------------------------
# (1) 절대 분산 그래프
# -----------------------------
plt.figure(figsize=(6, 4))
plt.bar(['Common', 'Idiosync'], 
        [total_var * explained_ratio, idiosync_var],
        color=['#6bb6ff', '#ff8c8c'])

plt.ylabel('Variance')
plt.title('Variance Decomposition (Absolute)')
plt.tight_layout()

plt.savefig("figure/variance_decomposition_absolute.png")
plt.show()
plt.close()


# -----------------------------
# (2) 비율(%) 그래프
# -----------------------------
plt.figure(figsize=(6, 4))
plt.bar(['Common', 'Idiosync'], 
        [explained_ratio * 100, idiosync_ratio * 100],
        color=['#6bb6ff', '#ff8c8c'])

plt.ylabel('Contribution (%)')
plt.title('Variance Contribution Ratio (%)')

for i, v in enumerate([explained_ratio * 100, idiosync_ratio * 100]):
    plt.text(i, v + 1, f"{v:.1f}%", ha='center')

plt.ylim(0, 110)
plt.tight_layout()

plt.savefig("figure/variance_decomposition_ratio.png")
plt.show()
plt.close()

# ======================
# 2) Panel Regression
# ======================

# Merge panel with common factors as exogenous
panel_long_reset = panel_long.reset_index()
factor_cols = factors.columns.tolist()
factors_reset = factors.reset_index()
factors_reset.rename(columns={'index':'year_month'}, inplace=True)
panel_reg = pd.merge(panel_long_reset, factors_reset, on='year_month', how='left')

panel_reg.set_index(['구','year_month'], inplace=True)
panel_reg = panel_reg.sort_index()  # MultiIndex 정렬

y = panel_reg['real_std']
X = panel_reg[factor_cols]  # 상수항 제거

# PanelOLS with entity + time effects, drop absorbed variables
# Pass drop_absorbed=True to the PanelOLS constructor
po_model = PanelOLS(y, X, entity_effects=True, time_effects=True, drop_absorbed=True)

# Now, the .fit() method only gets covariance-related arguments
# Note: 'cluster_entity=True' is equivalent to 'clusters=panel_reg.index.get_level_values(0)'
po_res = po_model.fit(cov_type='clustered', cluster_entity=True)

print(po_res.summary)


# ======================
# 3) Residual Diagnostics
# ======================

residuals = po_res.resids

print("\n--- Residuals Head ---")
print(residuals.head())

print("\n--- Residuals Description ---")
print(residuals.describe())

# 3-1) 잔차 ACF/PACF
lags = 24
acf_vals = acf(residuals, nlags=lags)
pacf_vals = pacf(residuals, nlags=lags)


print(f"\n--- ACF (lags 0-{lags}) ---")
for lag, val in enumerate(acf_vals):
    print(f"Lag {lag}: {val:.4f}")

print(f"\n--- PACF (lags 0-{lags}) ---")
for lag, val in enumerate(pacf_vals):
    print(f"Lag {lag}: {val:.4f}")

plt.figure(figsize=(10,4))
plt.subplot(1,2,1)
plt.stem(range(lags+1), acf_vals)
plt.title('Residual ACF')
plt.xlabel('Lag')
plt.ylabel('ACF')

plt.subplot(1,2,2)
plt.stem(range(lags+1), pacf_vals)
plt.title('Residual PACF')
plt.xlabel('Lag')
plt.ylabel('PACF')

plt.tight_layout()
plt.savefig("figure/residual_acf_pacf.png")
plt.show()
plt.close()

# 3-2) 잔차 히스토그램 & QQ-plot
hist_counts, hist_edges = np.histogram(residuals, bins=30)
hist_mid = 0.5 * (hist_edges[1:] + hist_edges[:-1])

print("=== Residual Histogram ===")
for x, y in zip(hist_mid, hist_counts):
    print(f"Bin center: {x:.4f}, Count: {y}")

qq = stats.probplot(residuals, dist="norm")
theoretical_quantiles = qq[0][0]
sample_quantiles = qq[0][1]

# print("\n=== QQ-Plot Points ===")
# print("Theoretical Quantile | Sample Quantile")
# for t, s in zip(theoretical_quantiles, sample_quantiles):
#     print(f"{t:.4f} | {s:.4f}")

plt.figure(figsize=(10,4))
plt.subplot(1,2,1)
plt.hist(residuals, bins=30, color='#6bb6ff', edgecolor='k')
plt.title('Residual Histogram')
plt.xlabel('Residual')
plt.ylabel('Frequency')

plt.subplot(1,2,2)
import statsmodels.api as sm
sm.qqplot(residuals, line='45', ax=plt.gca(), fit=True)  # <- 여기 ax 추가
plt.title('Residual QQ-plot')

plt.tight_layout()
plt.savefig("figure/residual_hist_qq.png")
plt.show()
plt.close()

# 3-3) Ljung-Box Test
ljung_box = acorr_ljungbox(residuals, lags=[12,24], return_df=True)
print("\nLjung-Box Test for autocorrelation:")
print(ljung_box)

# 3-4) Heteroskedasticity Test
X_bp = sm.add_constant(panel_reg[factor_cols])
bp_test = het_breuschpagan(residuals, X_bp)
labels = ['Lagrange multiplier stat', 'p-value', 'f-value', 'f p-value']
print("\nBreusch-Pagan Test for heteroskedasticity:")
print(dict(zip(labels, bp_test)))

# # ======================
# # 4) Bootstrap CI for Random Effects Coefficients
# # ======================

# n_boot = 500
# coef_boot = np.zeros((n_boot, X.shape[1]))

# for i in tqdm(range(n_boot)):
#     sample_idx = np.random.choice(panel_reg.index.get_level_values(0).unique(), 
#                                   size=panel_reg.index.get_level_values(0).nunique(), replace=True)
#     sample_df = pd.concat([panel_reg.loc[g] for g in sample_idx])
#     y_boot = sample_df['real_std']
#     X_boot = sample_df[factor_cols]
#     X_boot = sm.add_constant(X_boot)
#     re_model_boot = RandomEffects(y_boot, X_boot)
#     re_res_boot = re_model_boot.fit()
#     coef_boot[i,:] = re_res_boot.params.values

# # Bootstrap CI
# ci_lower = np.percentile(coef_boot, 2.5, axis=0)
# ci_upper = np.percentile(coef_boot, 97.5, axis=0)
# params = re_res.params.values

# plt.figure(figsize=(8,5))
# plt.bar(range(len(params)), params, color='#6bb6ff', edgecolor='k')
# plt.errorbar(range(len(params)), params, yerr=[params-ci_lower, ci_upper-params],
#              fmt='none', ecolor='r', capsize=5)
# plt.xticks(range(len(params)), ['const'] + factor_cols, rotation=45)
# plt.title('Random Effects Coefficients with 95% Bootstrap CI')
# plt.tight_layout()
# plt.savefig("figure/bootstrap_coeff_ci.png")
# plt.show()
# plt.close()

# # ======================
# # 5) Rolling Time-Block CV (MSE)
# # ======================

# tscv = TimeSeriesSplit(n_splits=12)  # 12 rolling splits
# mse_list = []

# for train_idx, test_idx in tscv.split(panel_reg):
#     y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
#     X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
#     model_cv = RandomEffects(y_train, X_train).fit()
#     y_pred = model_cv.predict(X_test)
#     mse_list.append(np.mean((y_test - y_pred)**2))

# plt.figure(figsize=(8,4))
# plt.plot(range(1,len(mse_list)+1), mse_list, marker='o', color='#ff8c8c')
# plt.xlabel('Rolling Split')
# plt.ylabel('MSE')
# plt.title('Rolling Time-Block CV MSE')
# plt.tight_layout()
# plt.savefig("figure/rolling_cv_mse.png")
# plt.show()
# plt.close()

# # ======================
# # 6) Robustness Checks
# # ======================

# # 예: 표준화 방식 변경 시 Factor1 계수 비교
# # 여기서는 단순히 Z-score 표준화 vs Min-Max 예시
# from sklearn.preprocessing import MinMaxScaler, StandardScaler

# scalers = {'Z-score': StandardScaler(), 'MinMax': MinMaxScaler()}
# robust_coef = {}

# for key, scaler in scalers.items():
#     X_scaled = scaler.fit_transform(panel_reg[factor_cols])
#     X_scaled = sm.add_constant(X_scaled)
#     model_rb = RandomEffects(y, X_scaled).fit()
#     robust_coef[key] = model_rb.params.values

# plt.figure(figsize=(8,5))
# for i, key in enumerate(scalers.keys()):
#     plt.bar(np.arange(len(params)) + i*0.25, robust_coef[key], width=0.25, label=key)
# plt.xticks(np.arange(len(params)) + 0.25/2, ['const'] + factor_cols, rotation=45)
# plt.title('Robustness Check: Coefficients by Standardization Method')
# plt.legend()
# plt.tight_layout()
# plt.savefig("figure/robustness_coeff.png")
# plt.show()
# plt.close()
