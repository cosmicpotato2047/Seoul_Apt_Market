import pandas as pd

def run_weighted_index(input_csv: str, output_csv: str):
    """
    2단계: 구별 월별 가중평균 가격지수 계산
    """
    df = pd.read_csv(input_csv, dtype={"year_month": str})

    # 월별 구별 가중평균 계산
    weighted_index = df.groupby(['year_month', '구']).apply(
        lambda x: (x['price_median'] * x['count']).sum() / x['count'].sum()
    ).reset_index(name='weighted_price_index')

    # year_month 문자열 유지
    weighted_index['year_month'] = weighted_index['year_month'].astype(str)

    # CSV로 저장
    weighted_index.to_csv(output_csv, index=False)
    print(f"[weighted_index] 완료: {output_csv}")

# 실행용
if __name__ == "__main__":
    run_weighted_index("data/monthly_grouped.csv", "data/weighted_index.csv")
