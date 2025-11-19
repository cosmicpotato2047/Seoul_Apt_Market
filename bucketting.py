import pandas as pd

# 1) CSV 읽기
df = pd.read_csv("data/apt_sale_cleaned.csv", parse_dates=["date"])

# 2) 전용면적 버킷
def area_bucket(area):
    if area < 40:
        return "<40"
    elif area < 60:
        return "40-60"
    elif area < 85:
        return "60-85"
    else:
        return ">85"

df["area_bucket"] = df["전용면적(㎡)"].apply(area_bucket)

# 3) 노후도 계산
df["age"] = df["date"].dt.year - df["건축년도"]

# 4) 노후도 버킷
def age_bucket(age):
    if age < 5:
        return "<5"
    elif age < 15:
        return "5-15"
    elif age < 30:
        return "15-30"
    else:
        return ">30"

df["age_bucket"] = df["age"].apply(age_bucket)

# 5) 새 CSV로 저장
df.to_csv("data/apt_sale_cleaned_bucketing.csv", index=False)

print("버킷팅 완료! 결과가 'data/apt_sale_cleaned_bucketing.csv'에 저장되었습니다.")
