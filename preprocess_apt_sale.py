import pandas as pd

# 1. 엑셀 불러오기
# 1) 엑셀 파일의 두 시트 읽기
df1 = pd.read_excel("/data/Apt_Sale_Price.xlsx", sheet_name="2006~2015")
df2 = pd.read_excel("/data/Apt_Sale_Price.xlsx", sheet_name="2016~2025.8")

# 2) 두 시트 결합
df = pd.concat([df1, df2], ignore_index=True)

# 2. 계약년월/계약일 → date 생성
df['계약년월'] = df['계약년월'].astype(str)
df['계약일'] = df['계약일'].astype(str).str.zfill(2)

df['date'] = pd.to_datetime(df['계약년월'] + df['계약일'], format="%Y%m%d", errors='coerce')

# 3. 시군구 분해: 시 제거하고 '구', '동'만 추출
# 예: "서울특별시 동대문구 답십리동"
df[['시', '구', '동']] = df['시군구'].str.split(expand=True)

# 시는 모두 "서울특별시"이므로 삭제
df = df.drop(columns=['시'])

# 4. 숫자 변환
df['거래금액(만원)'] = df['거래금액(만원)'] \
    .astype(str).str.replace(",", "").astype(int)

df['전용면적(㎡)'] = df['전용면적(㎡)'].astype(float)

# 5. 컬럼 순서 재정렬
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

# 6. 저장 (원하면 CSV로)
df.to_csv("clean.csv", index=False, encoding="utf-8-sig")
