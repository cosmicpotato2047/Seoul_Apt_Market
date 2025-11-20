import pandas as pd
import numpy as np
from statsmodels.tsa.statespace.dynamic_factor import DynamicFactor
import warnings
warnings.filterwarnings("ignore")


# ======================
# 1) Load Data
# ======================

df_wide = pd.read_csv("data/panel_wide_matrix.csv", index_col=0)
df_wide.index = pd.to_datetime(df_wide.index)


# # ======================
# # 2) Grid Search for (k, p)
# # ======================

# max_factors = 5
# factor_orders = [1, 2]

# results = []  # (k, p, AIC, BIC, res)

# print("\n=== Running DFM model selection for (k, p) ===\n")

# for k in range(1, max_factors + 1):
#     for p in factor_orders:
#         print(f"Testing k={k}, p={p} ...")

#         try:
#             model = DynamicFactor(
#                 df_wide,
#                 k_factors=k,
#                 factor_order=p,
#                 error_order=0
#             )

#             res = model.fit(maxiter=300, disp=False)  # 수렴 안정성 ↑

#             results.append((k, p, res.aic, res.bic, res))

#         except Exception as e:
#             print(f"  Failed (k={k}, p={p}): {e}")
#             continue


# # ======================
# # 3) Select Optimal Combination (BIC 최소 기준)
# # ======================

# results_df = pd.DataFrame(
#     [(k, p, aic, bic) for (k, p, aic, bic, _) in results],
#     columns=["k", "p", "AIC", "BIC"]
# )

# best_idx = results_df["BIC"].idxmin()
# best_k = int(results_df.loc[best_idx, "k"])
# best_p = int(results_df.loc[best_idx, "p"])

# print("\n=== Model selection result ===\n")
# print(results_df)
# print(f"\n*** Optimal (k, p) = ({best_k}, {best_p}) ***\n")

best_k = 1
best_p = 2

# ======================
# 4) Refit Final Model
# ======================

print(f"Training final DFM with k={best_k}, p={best_p}...\n")

final_model = DynamicFactor(
    df_wide,
    k_factors=best_k,
    factor_order=best_p,
    error_order=0
)

final_res = final_model.fit(maxiter=300, disp=False)

print("Final model fitted.\n")


# ======================
# 5) Extract Factors
# ======================

k = final_model.k_factors  # 학습 시 지정한 요인 수
factors = pd.DataFrame(final_res.factors.filtered.T,  # <-- transpose
                       index=df_wide.index,
                       columns=[f"Factor{i+1}" for i in range(k)])
factors.to_csv("data/dfm_common_factors.csv")

print("Common factors saved → data/dfm_common_factors.csv")


# ======================
# 6) Extract Idiosyncratic Components
# ======================

# Factor loadings
loadings = final_res.coefficient_matrices_var[0]  # <-- 수정

# idiosyncratic components 계산
idiosync = df_wide.values - np.dot(factors.values, loadings.T)
idiosync_df = pd.DataFrame(idiosync, index=df_wide.index, columns=df_wide.columns)
idiosync_df.to_csv("data/dfm_idiosyncratic_components.csv")

print("Idiosyncratic components saved → data/dfm_idiosyncratic_components.csv")
