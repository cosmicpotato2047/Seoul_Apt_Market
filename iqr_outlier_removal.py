import pandas as pd

# 1) 버킷팅 완료 CSV 읽기
df = pd.read_csv("data/apt_sale_cleaned_bucketing.csv", parse_dates=["date"])

# 2) 이상치 제거 함수 (IQR)
def remove_outliers_iqr(group, column="거래금액(만원)"):
    Q1 = group[column].quantile(0.25)
    Q3 = group[column].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    return group[(group[column] >= lower_bound) & (group[column] <= upper_bound)]

# 3) 그룹별(IQR 적용)
# 그룹: 단지명 + area_bucket
df_clean = df.groupby(["단지명", "area_bucket"], group_keys=False).apply(remove_outliers_iqr)

# 4) 결과 CSV 저장
df_clean.to_csv("data/apt_sale_cleaned_iqr.csv", index=False)

print("IQR 이상치 제거 완료! 결과가 'data/apt_sale_cleaned_iqr.csv'에 저장되었습니다.")
