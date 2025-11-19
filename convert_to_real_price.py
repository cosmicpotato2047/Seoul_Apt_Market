import pandas as pd

# 1️⃣ CSV 불러오기
price_df = pd.read_csv("data/weighted_index.csv", dtype={"year_month": str})
cpi_df = pd.read_csv("data/소비자물가지수_2020100__20251119115206.csv", encoding="utf-8")  # 한글 CSV

# 2️⃣ CPI 데이터 전처리
# 첫 번째 열 제거 후, 2006-01 ~ 2025-08 컬럼만 사용
cpi_df = cpi_df.iloc[0, 1:]  # 첫 번째 행, 두 번째 열부터
cpi_df.index = pd.period_range(start='2006-01', periods=len(cpi_df), freq='M')  # PeriodIndex
cpi_df = cpi_df.astype(float)

# 3️⃣ price_df의 year_month를 PeriodIndex로 변환
price_df['year_month'] = pd.to_datetime(price_df['year_month'], format='%Y-%m').dt.to_period('M')

# 4️⃣ CPI 매칭하여 실질가격 계산
def compute_real_price(row):
    return row['weighted_price_index'] / cpi_df[row['year_month']] * 100  # 2020년=100 기준

price_df['real_price_index'] = price_df.apply(compute_real_price, axis=1)

# 5️⃣ CSV로 저장
price_df.to_csv("data/weighted_index_real.csv", index=False)

print(price_df.head())
