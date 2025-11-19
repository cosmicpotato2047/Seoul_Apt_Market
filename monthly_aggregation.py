import pandas as pd

def run_monthly_aggregation(input_csv: str, output_csv: str):
    """
    1단계: 월별 구 × 면적버킷 × 노후도버킷 중앙값과 거래건수 계산
    """
    # CSV 불러오기
    df = pd.read_csv(input_csv, parse_dates=["date"])

    # YYYY-MM 컬럼 생성
    df['year_month'] = df['date'].dt.to_period('M')

    # 월별 그룹화: 구 × 면적버킷 × 노후버킷
    monthly_grouped = df.groupby(
        ['year_month', '구', 'area_bucket', 'age_bucket']
    )['거래금액(만원)'].median().reset_index()

    monthly_grouped.rename(columns={'거래금액(만원)': 'price_median'}, inplace=True)

    # 거래건수 계산
    monthly_grouped['count'] = df.groupby(
        ['year_month', '구', 'area_bucket', 'age_bucket']
    )['거래금액(만원)'].count().values

    # CSV로 저장
    monthly_grouped.to_csv(output_csv, index=False)
    print(f"[monthly_aggregation] 완료: {output_csv}")

# 실행용
if __name__ == "__main__":
    run_monthly_aggregation("data/apt_sale_cleaned_iqr.csv", "data/monthly_grouped.csv")
