import pandas as pd

# 1. 엑셀 불러오기
df1 = pd.read_excel("data/Apt_Sale_Price.xlsx", sheet_name="2006~2015")
df2 = pd.read_excel("data/Apt_Sale_Price.xlsx", sheet_name="2016~2025.8")

# 2. 두 시트 결합
df = pd.concat([df1, df2], ignore_index=True)

# 3. 계약년월/계약일 → date 생성
# NaN은 그대로 NaT로 처리
df['계약년월'] = df['계약년월'].apply(lambda x: str(int(x)) if pd.notna(x) else None)
df['계약일'] = df['계약일'].apply(lambda x: str(int(x)).zfill(2) if pd.notna(x) else None)

# 합쳐서 datetime 생성, 변환 실패하면 NaT
df['date'] = pd.to_datetime(
    df['계약년월'] + df['계약일'],
    format="%Y%m%d",
    errors='coerce'
)

# 4. 날짜 없는 행 제거
df = df[df['date'].notna()]

# 5. 시군구 분해: 시 제거하고 '구', '동'만 추출
df[['시', '구', '동']] = df['시군구'].str.split(expand=True)
df = df.drop(columns=['시'])

# 6. 숫자 변환
df['거래금액(만원)'] = pd.to_numeric(
    df['거래금액(만원)'].astype(str).str.replace(",", "", regex=False),
    errors='coerce'
).fillna(0).astype(int)

df['전용면적(㎡)'] = df['전용면적(㎡)'].astype(float)

df['건축년도'] = pd.to_numeric(df['건축년도'], errors='coerce').fillna(0).astype(int)

# 7. 컬럼 순서 재정렬
df = df[[
    'date',
    '구',
    '동',
    '전용면적(㎡)',
    '거래금액(만원)',
    '건축년도',
    '단지명',
    '도로명'
]]

# 8. CSV로 저장
df.to_csv("data/apt_sale_cleaned.csv", index=False, encoding="utf-8-sig")
