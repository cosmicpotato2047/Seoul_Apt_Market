# (결측 없음 가정, 외생변수는 나중 merge)

import pandas as pd

# 1) 로드
df = pd.read_csv("data/weighted_index_real.csv")

# 2) year_month → datetime 변환
df['year_month'] = pd.to_datetime(df['year_month'])

# 3) 정렬
df = df.sort_values(["구", "year_month"]).reset_index(drop=True)

# 4) weighted_price_index 제거
df_model = df.drop(columns=["weighted_price_index"])

# 5) 구별 표준화(z-score): DFM/State-Space 입력 전용
df_model['real_std'] = df_model.groupby("구")['real_price_index'].transform(
    lambda x: (x - x.mean()) / x.std()
)

# 6) 패널 형태 그대로 유지
#   year_month: 시계열 축
#   구: 패널 단위
#   real_price_index: 원시 실질가격 (회귀용)
#   real_std: 표준화 가격 (DFM/State-Space용)
df_model = df_model[['year_month', '구', 'real_price_index', 'real_std']]

# 7) wide 형태 행렬도 DFM용으로 준비 (선택)
#   열 = 구, 값 = real_std
df_wide = df_model.pivot(index='year_month', columns='구', values='real_std')

# 8) 저장
df_model.to_csv("data/panel_prepared.csv", index=False)
df_wide.to_csv("data/panel_wide_matrix.csv")

print("Panel and matrix prepared successfully.")
