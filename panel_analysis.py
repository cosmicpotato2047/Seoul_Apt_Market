import pandas as pd
import numpy as np
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

# # ======================
# # 2) Panel Regression
# # ======================

# # Merge panel with common factors as exogenous
# panel_long_reset = panel_long.reset_index()
# factor_cols = factors.columns.tolist()
# factors_reset = factors.reset_index()
# factors_reset.rename(columns={'index':'year_month'}, inplace=True)
# panel_reg = pd.merge(panel_long_reset, factors_reset, on='year_month', how='left')
# panel_reg.set_index(['구','year_month'], inplace=True)

# # Random Effects Panel Regression
# y = panel_reg['real_std']
# X = panel_reg[factor_cols]
# X = sm.add_constant(X)

# re_model = RandomEffects(y, X)
# re_res = re_model.fit()
# print(re_res.summary)


# # ======================
# # 3) Residual Diagnostics
# # ======================

# residuals = re_res.resids

# # 3a) Auto-correlation
# lb_test = acorr_ljungbox(residuals, lags=[12], return_df=True)
# print("\nLjung-Box test (lag=12):")
# print(lb_test)

# # ACF/PACF plot
# fig, ax = plt.subplots(2,1, figsize=(10,6))
# acf_vals = acf(residuals, nlags=24)
# pacf_vals = pacf(residuals, nlags=24)
# ax[0].stem(acf_vals); ax[0].set_title("Residuals ACF")
# ax[1].stem(pacf_vals); ax[1].set_title("Residuals PACF")
# plt.tight_layout()
# plt.show()

# # 3b) Heteroskedasticity (Breusch-Pagan)
# import statsmodels.api as sm
# bp_test = het_breuschpagan(residuals, X)
# bp_df = pd.DataFrame({'LM stat':[bp_test[0]], 'LM p-value':[bp_test[1]], 'F stat':[bp_test[2]], 'F p-value':[bp_test[3]]})
# print("\nBreusch-Pagan test for heteroskedasticity:")
# print(bp_df)


# # ======================
# # 4) Rolling CV (Time-Block)
# # ======================

# tscv = TimeSeriesSplit(n_splits=5)
# mse_list = []

# for train_idx, test_idx in tqdm(tscv.split(panel_reg)):
#     X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
#     y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
#     model_cv = sm.OLS(y_train, X_train).fit()
#     y_pred = model_cv.predict(X_test)
#     mse = ((y_test - y_pred)**2).mean()
#     mse_list.append(mse)

# print(f"\nRolling CV MSE: {np.mean(mse_list):.4f}")


# # ======================
# # 5) Robustness Checks
# # ======================

# # 예: 표준화/디트렌딩 방식 변경 후 회귀
# panel_std_alt = (panel - panel.mean())/panel.std()
# panel_std_alt.index.name = 'year_month'
# panel_long_alt = panel_std_alt.reset_index().melt(
#     id_vars='year_month',
#     var_name='구',
#     value_name='real_std'
# )
# panel_long_alt.set_index(['구','year_month'], inplace=True)
# # 동일 회귀 수행 가능

# print("\nRobustness check ready (alternative standardization)\n")


# # ======================
# # 6) Bootstrap for Random Effects CI
# # ======================

# n_boot = 500
# coef_boot = []

# for i in tqdm(range(n_boot)):
#     sample_idx = np.random.choice(panel_reg.index.get_level_values(0).unique(), replace=True, size=len(panel_reg.index.get_level_values(0).unique()))
#     sample_data = panel_reg.loc[sample_idx]
#     y_b = sample_data['real_std']
#     X_b = sm.add_constant(sample_data[factor_cols])
#     model_b = RandomEffects(y_b, X_b)
#     res_b = model_b.fit()
#     coef_boot.append(res_b.params.values)

# coef_boot = np.array(coef_boot)
# ci_lower = np.percentile(coef_boot, 2.5, axis=0)
# ci_upper = np.percentile(coef_boot, 97.5, axis=0)

# ci_df = pd.DataFrame({
#     'coef': re_res.params.values,
#     'CI_lower': ci_lower,
#     'CI_upper': ci_upper
# }, index=['const'] + factor_cols)

# ci_df.to_csv("data/panel_random_effects_bootstrap_CI.csv")
# print("\nBootstrap CI saved → data/panel_random_effects_bootstrap_CI.csv")


# # ======================
# # 7) Visualization
# # ======================



# # 7b) DFM Common Factors 시계열
# factors.plot(figsize=(12,4), title='DFM Common Factors')
# plt.xlabel('Year-Month')
# plt.ylabel('Factor Value')
# plt.tight_layout()
# plt.savefig("figure/dfm_common_factors.png")
# plt.show()
# plt.close()

# # 7c) Residual distribution
# plt.figure(figsize=(6,4))
# residuals.hist(bins=30, color='lightgreen')
# plt.title('Residual Distribution')
# plt.xlabel('Residual')
# plt.ylabel('Frequency')
# plt.tight_layout()
# plt.savefig("figure/residual_hist.png")
# plt.show()
# plt.close()

# plt.figure(figsize=(6,4))
# plt.boxplot(residuals, vert=False)
# plt.title('Residual Boxplot')
# plt.tight_layout()
# plt.savefig("figure/residual_boxplot.png")
# plt.show()
# plt.close()

# # 7d) Residuals ACF/PACF
# from statsmodels.tsa.stattools import acf, pacf

# acf_vals = acf(residuals, nlags=24)
# pacf_vals = pacf(residuals, nlags=24)

# fig, ax = plt.subplots(2,1, figsize=(10,6))
# ax[0].stem(acf_vals)
# ax[0].set_title("Residuals ACF")
# ax[1].stem(pacf_vals)
# ax[1].set_title("Residuals PACF")
# plt.tight_layout()
# plt.savefig("figure/residual_acf_pacf.png")
# plt.show()
# plt.close()

# # 7e) Rolling CV MSE
# plt.figure(figsize=(8,4))
# plt.plot(mse_list, marker='o', linestyle='-', color='orange')
# plt.title('Rolling CV MSE over splits')
# plt.xlabel('CV Split')
# plt.ylabel('MSE')
# plt.tight_layout()
# plt.savefig("figure/rolling_cv_mse.png")
# plt.show()
# plt.close()

# # 7f) Bootstrap Coefficients CI
# plt.figure(figsize=(10,5))
# coef_names = ['const'] + factor_cols
# coefs = ci_df['coef']
# ci_lower = ci_df['CI_lower']
# ci_upper = ci_df['CI_upper']

# plt.errorbar(coef_names, coefs, yerr=[coefs-ci_lower, ci_upper-coefs], fmt='o', capsize=5, color='purple')
# plt.title('Bootstrap 95% CI of Coefficients')
# plt.ylabel('Coefficient Value')
# plt.xticks(rotation=45)
# plt.tight_layout()
# plt.savefig("figure/bootstrap_coef_ci.png")
# plt.show()
# plt.close()
