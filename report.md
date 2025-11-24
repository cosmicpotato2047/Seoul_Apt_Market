목차 만들고 클릭시 목차로 빠르게 이동하는 기능 본문에서 제목 누르면 최 상단 목차 테이블로 이동

# **1. Introduction**

# **2. Data Description**

본 프로젝트에서는 국토교통부 실거래가 공개시스템 ([https://rt.molit.go.kr](https://rt.molit.go.kr/pt/xls/xls.do?&mobileAt=))에서 제공하는 **서울특별시 아파트 실거래 자료(2006.01.01–2025.08.31)**를 활용하였다.


자료는 다음의 검색 조건을 통해 Excel 파일로 수집하였다.

* **계약 일자**: 2006년 1월 1일 ~ 2025년 8월 31일
* **매물 유형**: 아파트
* **주소 기준**: 도로명
* **지역 범위**: 서울특별시 전체 25개 구
* **면적·금액 필터**: 제한 없음

실거래 자료는 건별 계약 정보로 구성되어 있으며, 주요 변수는 아래와 같다.

| 변수명        | 설명                     |
| ---------- | ---------------------- |
| 시군구        | “서울특별시 XX구 YY동” 형식의 주소 |
| 전용면적(㎡)    | 실사용 면적                 |
| 계약년월 / 계약일 | 계약 체결 시점               |
| 거래금액(만원)   | 실거래 신고 금액              |
| 단지명        | 아파트 단지명                |
| 건축년도       | 준공년도                   |
| 도로명        | 도로명 주소 정보              |

다음 Column은 삭제: NO, 번지, 본번, 부번, 동, 층, 매수자, 매도자, 해제사유발생일, 거래유형, 중개사소재지, 등기일자, 주택유형

---

# **3. Preprocessing**

## **3.1. 두 시트 결합 및 날짜 생성**

엑셀 파일의 두 시트(2006~2015, 2016~2025.8)를 하나의 DataFrame으로 통합한 후,
`계약년월`과 `계약일` 정보를 기반으로 단일 날짜 변수(`date`)를 생성하였다.

원본 데이터는 계약년월이 `YYYYMM`, 계약일이 `DD` 형태로 분리되어 있으며,
두 값을 문자열로 변환한 뒤 `"YYYYMMDD"` 형태로 합쳐 날짜로 파싱하였다.
날짜 변환이 불가능한 경우는 자연스럽게 `NaT`로 처리되며,
해당 행은 분석에서 제외하였다.

```python
df['date'] = pd.to_datetime(
    df['계약년월'] + df['계약일'],
    format="%Y%m%d",
    errors='coerce'
)
df = df[df['date'].notna()]
```

이를 통해 **모든 거래 건이 월·일 단위까지 명확히 정렬된 시계열 구조**로 재구성되었다.

---

## **3.2. 주소 분해: 구/동 추출**

`시군구` 변수는 `"서울특별시 중랑구 면목동"`과 같은 문자열로 구성되어 있어,
본 연구에서는 분석 단위를 구별로 다루기 위해 이를 세 부분으로 나누어 사용하였다.

* `"서울특별시"` → 제거
* `"XX구"` → 구 단위 변수
* `"YY동"` → 동 단위 변수

```python
df[['시', '구', '동']] = df['시군구'].str.split(expand=True)
df = df.drop(columns=['시'])
```

이 과정에서 **서울 25개 구 단위의 시계열 분석이 가능하도록 주소 체계가 정규화**되었다.

---

## **3.3. 수치형 변수 정제**

실거래 금액, 건축년도, 전용면적 등의 변수는 문자열·결측치·콤마 등이 포함되어 있어
이를 정수 또는 실수로 변환하였다.

* `거래금액(만원)` → 콤마 제거 후 정수 변환
* `전용면적(㎡)` → float
* `건축년도` → 숫자 변환 불가 값은 0 처리

```python
df['거래금액(만원)'] = pd.to_numeric(
    df['거래금액(만원)'].astype(str).str.replace(",", ""),
    errors='coerce'
).fillna(0).astype(int)
```

해당 변환을 통해 **모든 거래 기록이 통계 분석 및 시계열 변환에 바로 활용 가능한 형태**로 표준화되었다.

---

## **3.4. 분석용 컬럼 정리 및 저장**

필요한 변수만 선별하여 다음 순서로 재배치하였다.

* `date`
* `구`
* `동`
* `전용면적(㎡)`
* `거래금액(만원)`
* `건축년도`
* `단지명`
* `도로명`

최종 전처리된 자료는 CSV 파일로 저장해 이후 단계의
월별 집계, 실질가격 변환, Dynamic Factor Model,
그리고 DTW 클러스터링 등에 사용된다.

```python
df.to_csv("data/apt_sale_cleaned.csv", index=False, encoding="utf-8-sig")
```

## **3.5. 전용면적 및 노후도 버킷팅(Bucketing)**

아파트 가격은 전용면적과 건축년도 같은 물리적 특성에 따라 구조적인 차이를 보이기 때문에,
분석 과정에서 **거래 금액의 분포를 안정화하고 구간별 비교가 가능하도록**
전용면적(㎡)과 노후도(건축년도 기준)를 범주형 변수로 재구성하였다.

### **(1) 전용면적 버킷 생성**

전용면적은 다음 네 가지 구간으로 구분하였다.

* **< 40㎡**
* **40–60㎡**
* **60–85㎡**
* **> 85㎡**

이 구간은 서울 아파트 시장에서 일반적으로 구분되는 **소형–중형–중대형–대형** 면적대와 일치하며,
각 면적대는 가격 수준과 변동 패턴이 뚜렷하게 다르기 때문에
시계열 분석 및 군집 분석에서 군집 왜곡을 막는 데 유효하다.

```python
def area_bucket(area):
    if area < 40:
        return "<40"
    elif area < 60:
        return "40-60"
    elif area < 85:
        return "60-85"
    else:
        return ">85"
```

### **(2) 노후도 계산 및 버킷팅**

각 거래 건의 **노후도(age)**는

```python
age = (거래년도) - (건축년도)
```

로 계산하였으며, 다시 다음의 네 구간으로 범주화하였다.

* **< 5년** (신축)
* **5–15년** (준신축)
* **15–30년** (일반 노후)
* **> 30년** (노후)

이는 서울 아파트 시장에서 가격 형성 요인으로 가장 강력하게 작용하는 특성 중 하나가
건축년도이기 때문이다. 특히 신축과 30년 이상 노후 단지는
동일 면적이라도 가격 구조가 완전히 다르기 때문에
이를 구분하지 않을 경우 후속 분석에서 평균 왜곡이 발생한다.

최종 데이터는 다음의 형태로 확장되었다.

* `area_bucket`: 전용면적 구간
* `age`: 노후도(연 단위)
* `age_bucket`: 노후도 구간

가공된 데이터는 `"apt_sale_cleaned_bucketing.csv"`로 저장하여
이상치 제거 및 월별 집계 단계에서 사용하였다.

---

## **3.6. IQR 기반 이상치 제거**

서울 아파트 실거래 데이터는 단지명, 면적대, 층수, 특수 거래 등 다양한 요인의 영향으로
일부 **극단값(outlier)**이 존재한다.
특히 특정 단지의 일부 이례적 고가·저가 거래는
월평균 가격 및 구 단위 지수 계산 시 왜곡을 유발할 수 있다.

이를 완화하기 위해 **사분위수 기반 IQR(Interquartile Range) 방법**을 활용하여
면적대별·단지별로 이상치를 제거하였다.

### **(1) 제거 방식**

각 단지(`단지명`)와 면적 버킷(`area_bucket`)을 하나의 그룹으로 묶고,
각 그룹 내에서 다음 경계를 벗어난 거래를 제거한다.

* Q1 − 1.5×IQR 미만
* Q3 + 1.5×IQR 초과

```python
def remove_outliers_iqr(group, column="거래금액(만원)"):
    Q1 = group[column].quantile(0.25)
    Q3 = group[column].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    return group[(group[column] >= lower_bound) & (group[column] <= upper_bound)]
```

### **(2) 그룹별 IQR 적용 이유**

* 동일 단지라도 면적대에 따라 금액 수준과 분포가 완전히 다르기 때문에
  **단지 + 면적대** 수준에서 이상치를 제거해야 정상 분포를 보정할 수 있다.
* 구 단위로 한꺼번에 IQR을 적용하면, 특정 고급 단지의 고가 거래가 전체 상한선을 끌어올려
  실제 이상치가 제거되지 않는 문제가 발생한다.
* 면적대·단지 수준의 미세한 그룹화는 이후 **월별 지수화 작업의 안정성**을 크게 높여준다.

### **(3) 최종 데이터**

이상치 제거가 완료된 데이터는
`"apt_sale_cleaned_iqr.csv"`에 저장되며,
이후 단계인 월별 집계, 실질가격 변환, 패널 분석,
Dynamic Factor Model(DFM) 적용의 입력으로 사용된다.

아래는 **월별 집계 → 가중 지수 → 실질 가격 변환**의 전체 과정을
보고서에 바로 포함할 수 있는 서술 방식으로 정돈한 내용이야.
각 단계가 왜 필요한지, 어떤 방식으로 구현했는지, 분석 전체 흐름 속에서 어떤 역할을 하는지를 명확히 설명하도록 구성했다.

## **3.7. 월별 집계(Monthly Aggregation)**

전처리된 거래 자료는 건별 시점·면적·노후도·단지 특성이 모두 다르기 때문에,
그대로 사용하면 월 단위 시계열 비교가 어렵다.
이를 해결하기 위해 거래 데이터를 **구 × 면적버킷 × 노후도버킷** 단위로 묶어
매월의 대표값을 계산했다.

### **(1) 집계 단위 설정 이유**

* 각 구의 거래 특성을 면적별·노후도별로 세분화함으로써
  구조적으로 다른 유형의 아파트를 동일 선상에 놓고 단순 평균하는 왜곡을 방지한다.
* 중앙값(median)을 사용함으로써 극단값(outlier)의 영향을 추가로 완화한다.
* 구별 주택 시장의 변화를 월 단위로 분석할 수 있도록 시계열 정규화를 수행한다.

### **(2) 계산 방식**

각 거래의 날짜(`date`)에서 `"YYYY-MM"` 형식의 `year_month`를 생성한 뒤,
다음 집계를 수행했다.

* **price_median**:
  동일 월 × 구 × 면적버킷 × 노후도버킷 내 거래금액의 중앙값
* **count**:
  집계 구간 내 거래 건수

대표 코드:

```python
monthly_grouped = df.groupby(
    ['year_month', '구', 'area_bucket', 'age_bucket']
)['거래금액(만원)'].median().reset_index()

monthly_grouped['count'] = df.groupby(
    ['year_month', '구', 'area_bucket', 'age_bucket']
)['거래금액(만원)'].count().values
```

출력 파일: **`monthly_grouped.csv`**
이 데이터는 이후 **구별 월별 가격지수 계산**의 입력으로 사용된다.

---

## **3.8. 가중 지수 계산(Weighted Price Index)**

월별 집계 데이터는 면적·노후도별로 쪼개져 있기 때문에,
이를 다시 **구 단위의 단일 월별 가격지수**로 통합할 필요가 있다.

단순 평균을 사용할 경우 특정 면적대의 과대표집 문제가 발생하기 때문에,
본 연구에서는 **거래건수를 가중치로 사용하는 가중평균(weighted mean)**을 적용했다.

### **(1) 목적**

* 면적·노후도 분포가 시기별로 변하더라도
  구 전체의 가격 수준을 일관되게 측정한다.
* 거래량이 많은 구간에 더 높은 비중을 부여하여
  실제 시장 움직임과 가까운 지수를 생성한다.

### **(2) 계산 방식**

각 월·구 단위에서 다음 공식을 사용했다.

[
Index_{t, gu}=\frac{\sum_{i} (MedianPrice_{i} \times Count_{i})}{\sum_{i} Count_{i}}
]

코드 구현:

```python
weighted_index = df.groupby(['year_month', '구']).apply(
    lambda x: (x['price_median'] * x['count']).sum() / x['count'].sum()
).reset_index(name='weighted_price_index')
```

출력 파일: **`weighted_index.csv`**
이 값은 **명목 가격지수(Nominal Price Index)**이며,
물가(인플레이션) 반영 이전의 지수이다.

---

## **3.9. 실질 가격 변환(Real Price Index Conversion)**

명목 가격지수는 명확한 추세를 보여주지만,
2006~2025 기간 동안 소비자물가(CPI)가 꾸준히 상승해왔기 때문에
단순 비교만으로는 **실질적인 가격 변동**을 파악하기 어렵다.

따라서 KOSIS(국가통계포털) ([https://kosis.kr/](https://kosis.kr/statHtml/statHtml.do?sso=ok&returnurl=https%3A%2F%2Fkosis.kr%3A443%2FstatHtml%2FstatHtml.do%3Flist_id%3D%26obj_var_id%3D%26seqNo%3D%26docId%3D02881%26tblId%3DDT_1J22003%26vw_cd%3DMT_ZTITLE%26orgId%3D101%26conn_path%3DE1%26markType%3DS%26itm_id%3D%26lang_mode%3Dko%26scrId%3D%26itmNm%3D%EC%A0%84%EA%B5%AD%26))에서 제공하는 **소비자물가지수(CPI, 기준=2020=100)**를 활용하여
각 월의 가격을 실질 가격(real price)으로 디플레이트하였다.

### **(1) CPI 데이터 처리**

* 2020년을 기준값 100으로 맞춘 지수
* 첫 번째 행(헤더 행)을 제외한 하나의 행을 선택
* `"2006-01" ~ "2025-08"` 구간을 PeriodIndex로 정렬

대표 처리 코드:

```python
cpi_df = cpi_df.iloc[0, 1:]
cpi_df.index = pd.period_range(start='2006-01', periods=len(cpi_df), freq='M')
cpi_df = cpi_df.astype(float)
```

### **(2) 실질 가격 계산 공식**

명목 지수를 CPI로 나누어 실질 지수로 변환:

[
RealIndex_{t} = \frac{NominalIndex_{t}}{CPI_{t}} \times 100
]

### **(3) 코드 구현**

```python
price_df['year_month'] = pd.to_datetime(price_df['year_month'], format='%Y-%m').dt.to_period('M')

def compute_real_price(row):
    return row['weighted_price_index'] / cpi_df[row['year_month']] * 100

price_df['real_price_index'] = price_df.apply(compute_real_price, axis=1)
```

출력 파일:
**`weighted_index_real.csv`**

이 파일은 이후

* Dynamic Factor Model (DFM)
* 구별 사이클 추정
* DTW 기반 군집 분석
* Panel 분석

등 모든 분석 기법에서 사용되는 **최종 시계열 기준 가격지수**가 된다.

---

# **4. Methodology**

본 연구의 방법론은 총 네 단계로 구성된다:
① 개별 구 단위의 가격 사이클 탐지,
② 패널 데이터 구조 구축 및 패널 회귀 시도,
③ Dynamic Factor Model(DFM)을 통한 공통요인 추출,
④ 시계열 간 유사도 기반 클러스터링(DTW)이다.

이 순서는 **단계적으로 더 높은 구조를 포착하는 방향**으로 설계되었다.
먼저 지역별 고유 패턴을 파악하고, 이후 이를 패널 구조로 통합해 설명가능성을 평가하며, 마지막으로 공통요인을 도출하고 클러스터링으로 패턴을 분류하는 방식이다.

---
## **4.1. 개별 구의 가격 사이클 및 변곡점 탐지**

### **(1) 목적**

개별 구별(real_price_index) 시계열에서

1. 장·단기 추세 분리(STL, HP Filter),
2. 주기성 탐지(FFT 기반),
3. 이상 패턴 탐지(Matrix Profile),
4. 구조적 변화점(Ruptures),
5. 극단값 이상치(IsolationForest, LOF)  

을 통합적으로 수행하여 **각 구의 가격 사이클 구조와 변곡점(패턴 변화 지점)을 자동 탐지**하는 것이 목적이다.

모든 분석은 **사전 파라미터 고정 없이**, FFT 기반 자동 주기 추정(m), Ruptures elbow 탐색(k) 등 **데이터 구동적(data-driven) 방식**으로 수행되도록 설계했다.

---

### **(2) 주요 알고리즘**

#### **- STL Decomposition**

* `STL(series, period=12)`
* 계절성(12개월), 장기추세(trend), 잔차를 분해하여 기본 구조를 파악.

#### **- HP Filter (λ=129600)**

* 장기추세와 순환성분(cycle) 추출.
* 이후 FFT 기반 주기 추정에 cycle 사용.

#### **- FFT 기반 자동 주기 추정(m)**

* HP-filter cycle에 대해 FFT 적용 → 파워 스펙트럼 `Pxx` 산출
* 최대 파워 주파수 인덱스 추출
* `dominant_period = 1 / freq[top_idx]`
* `m = max(12, round(dominant_period))`
  → 노이즈성 짧은 주기 방지

또한 해당 주파수의 위상값(phase shift)을 추출해 코사인 기반 이상적 주기선(m-cycle line) 시각화에 활용.

#### **- Welch PSD 보조 분석**

* `nperseg = min(len(series), m*3)`
* 주기(m) 기반으로 세그먼트 길이 자동 설정.

#### **- Matrix Profile (Stumpy) 기반 Discord 탐지**

* `mp = stumpy.stump(values, m)`
* 최댓값 인덱스 → 가장 특이한 subsequence(Discord) 탐지
* 해당 시점은 “주기 구조를 가장 크게 벗어난 구간”.

#### **- Ruptures – Binseg + Elbow Method**

* 변화점 후보 k = 1~15 반복
* 각각 cost 계산
* **직선에서 가장 멀어진 지점 = Elbow**
  → `optimal_n_bkps` 자동 결정
* `predict(n_bkps=k)`로 최적 변화점 도출

#### **- 이상치 탐지 (IsolationForest / LOF)**

* IsolationForest: contamination=0.05
* LOF: n_neighbors=12
* 값 기반 이상치(극단값)를 판별

---

### **(3) 대표 코드**

아래는 분석 과정 구성에서 핵심 역할을 수행하는 코드 일부(바로 동작 가능한 형태 그대로)이다.

#### **- STL & HP Filter**

```python
stl = STL(series, period=12)
res = stl.fit()
trend = res.trend
seasonal = res.seasonal

cycle, trend_hp = hpfilter(values, lamb=129600)
```

#### **- FFT 기반 자동 주기 탐지**

```python
fft_result = np.fft.fft(cycle)
freqs = np.fft.fftfreq(N)

Pxx = np.abs(fft_result[1:N//2])**2
top_idx = np.argmax(Pxx) + 1

dominant_period = 1 / freqs[top_idx]
m = int(max(12, round(dominant_period)))

phase_rad = np.angle(fft_result[top_idx])
```

#### **- Matrix Profile – Discord 탐지**

```python
mp = stumpy.stump(values, m)
discord_idx = np.argmax(mp[:, 0])
discord_date = series.index[discord_idx]
```

#### **- Ruptures + Elbow Method**

```python
algo_binseg = rpt.Binseg(model="rbf").fit(values)
costs = []

for k in range(1, 16):
    bkps = algo_binseg.predict(n_bkps=k)
    costs.append(algo_binseg.cost.sum_of_costs(bkps))

optimal_n_bkps = find_elbow_point(costs)
change_points = algo_binseg.predict(n_bkps=optimal_n_bkps)
change_dates = series.index[np.array(change_points[:-1]) - 1]
```

#### **- 이상치 탐지 (IForest, LOF)**

```python
X = values.reshape(-1, 1)

iforest = IsolationForest(contamination=0.05, random_state=42)
outlier_dates_if = series.index[iforest.fit_predict(X) == -1]

lof = LocalOutlierFactor(n_neighbors=12)
outlier_dates_lof = series.index[lof.fit_predict(X) == -1]
```

---

## **4.2. 패널 데이터 구조 구축 및 패널 회귀 분석**

### **(1) 목적**

서울 25개 구의 실질가격 시계열을 단일 시계열로 취급하면 지역별 상이한 구조적 변화나 고유 패턴을 반영하기 어렵다. 이를 보완하기 위해 각 구를 독립 개체(entity)로 간주하고, 월별 시계열(year_month)을 공유하는 형태의 **패널 데이터(panel data)**를 구성했다.
또한 Dynamic Factor Model(DFM)에서 추출된 **공통 요인(common factors)**을 설명변수로 사용해, 지역별 가격 변동이 얼마나 공통 충격에 의해 설명되는지 검증하기 위한 **패널 회귀(panel regression)**를 수행했다.

---

### **(2) 패널 데이터 구축**

패널 회귀와 요인모형 모두에서 요구하는 정형화된 데이터 구조를 만들기 위해 다음 단계를 거쳤다.

#### **- 시계열 정렬 및 클리닝**

```python
df = pd.read_csv("data/weighted_index_real.csv")
df['year_month'] = pd.to_datetime(df['year_month'])
df = df.sort_values(["구", "year_month"]).reset_index(drop=True)
```

* 모든 구에 대해 시계열이 시간 순서로 정렬된 구조 확보.

#### **- 불필요 변수 제거 및 표준화(z-score)**

```python
df_model = df.drop(columns=["weighted_price_index"])

df_model['real_std'] = df_model.groupby("구")['real_price_index'].transform(
    lambda x: (x - x.mean()) / x.std()
)
```

* DFM, 패널 회귀 모두 **스케일 차이를 제거**해 안정적인 추정 수행.
* real_std는 이후 wide matrix와 회귀 모두에서 활용되는 핵심 변수.

#### **- Long-form 패널 구조 정리**

```python
df_model = df_model[['year_month', '구', 'real_price_index', 'real_std']]
df_model.to_csv("data/panel_prepared.csv", index=False)
```

#### **- DFM용 Wide Matrix 생성**

```python
df_wide = df_model.pivot(index='year_month', columns='구', values='real_std')
df_wide.to_csv("data/panel_wide_matrix.csv")
```

* 패널 회귀는 long-form을,
* DFM은 wide-form(행: 날짜, 열: 구)을 요구하므로 두 형태를 모두 생성.

---

### **(3) 패널 회귀 시도**

#### **- 요인모형 결과 결합**

패널 회귀의 목적은 real_std가 DFM의 공통 요인에 의해 얼마나 설명되는지 검증하는 것이다.

```python
factors = pd.read_csv("data/dfm_common_factors.csv", index_col=0, parse_dates=True)

panel = pd.read_csv("data/panel_wide_matrix.csv", index_col=0, parse_dates=True)
panel_long = panel.reset_index().melt(
    id_vars='year_month', 
    var_name='구', 
    value_name='real_std'
)

panel_reg = pd.merge(panel_long.reset_index(), factors.reset_index(), on='year_month', how='left')
panel_reg.set_index(['구','year_month'], inplace=True)
panel_reg = panel_reg.sort_index()
```

#### **- PanelOLS 구성 (Fixed Effects + Time Effects)**

```python
y = panel_reg['real_std']
X = panel_reg[factor_cols]

po_model = PanelOLS(
    y, X, 
    entity_effects=True, 
    time_effects=True, 
    drop_absorbed=True
)

po_res = po_model.fit(cov_type='clustered', cluster_entity=True)
print(po_res.summary)
```

특징:

* **entity fixed effects**: 구별 평균적 차이 제거
* **time fixed effects**: 전 시점 공통 충격 제거
* **drop_absorbed=True**: 시간·개체 효과에 의해 중복되는 요인은 자동 제거
* **cluster_entity=True**: 구 단위로 군집 표준오차 계산

#### **- 잔차 진단 수행**

시간연속 시계열 패널의 특징으로 인해 잔차 진단은 필수다.

##### (a) ACF / PACF

```python
acf_vals = acf(residuals, nlags=24)
pacf_vals = pacf(residuals, nlags=24)
```

##### (b) 분포 검증 (Histogram, QQ-plot)

```python
plt.hist(residuals, bins=30)
sm.qqplot(residuals, line='45')
```

##### (c) 자기상관 검정 (Ljung-Box)

```python
acorr_ljungbox(residuals, lags=[12,24], return_df=True)
```

##### (d) 이분산성 검정 (Breusch-Pagan)

```python
bp_test = het_breuschpagan(residuals, sm.add_constant(panel_reg[factor_cols]))
```

---

### **(4) 시도 결과 및 한계**

#### **- 요인 설명력은 높았으나 회귀 형태에서는 제약 발생**

* DFM에서 공통 요인이 전체 변동의 상당 부분을 설명했으나
* fixed-effects 패널 회귀에서는 **시간효과(time FE)**가 모든 구에 동일한 충격을 제거한다.
* 결과적으로 DFM의 common factor와 time FE가 **중복되는 설명력**을 갖기 때문에 유효 회귀 계수가 제한되거나 제거됨.

#### **- 잔차에서 남는 자기상관**

* ACF·PACF에서 특정 시차에 구조적 패턴이 남아 있었음.
* 시계열 패널 자료는 일반 패널OLS 가정(독립성)에 적합하지 않음.

#### **- 이분산성 존재**

* Breusch-Pagan test에서 이분산 경향 확인.
* 이는 구별로 변동성 규모가 다른 시계열 패턴 때문.

#### **- 결론: 패널 회귀는 지원적 역할로 한정**

* 패널 회귀는 DFM 요인이 유의미하다는 “보조적 정성 검증”으로 활용 가능하지만
* **정량적 모형 선택으로는 부적합**하다는 판단.

따라서 이후 분석은
**공통 요인 기반 Dynamic Factor Model → 구별 패턴 분해 → 시계열 클러스터링(DTW)**
흐름으로 전환했다.

---

## **4.3. Dynamic Factor Model(DFM) 분석**

### **(1) 목적**

서울 25개구의 월별 실질 주택가격 변동에는 지역 고유 충격뿐 아니라 공통적인 거시적 충격(금리·유동성·전체 경기 흐름 등)이 동시에 작용한다.
따라서 패널 전체에 공통적으로 작용하는 **장기·중기 주기적 요인(common factor)**을 분리하고 그 변동성 기여도를 파악하기 위해 **Dynamic Factor Model(DFM)**을 적용하였다.

본 분석의 목적은 다음과 같다.

* 패널 전체가 공유하는 **공통 요인 추출**
* 개별 구의 변동 중 공통 요인으로 설명되는 **분산 비중 산출**
* 추출된 공통 요인의 **주기·변곡점(변화점) 분석**
* 지역별 고유(idiosyncratic) 충격과 구분하여 **집단적 시장 사이클 구조 파악**

DFM은 상태공간(State-Space) 구조로 구성되어 요인의 동적 구조(factor order)를 포함할 수 있으며, 단순 PCA보다 시간적 패턴을 더 잘 설명한다는 장점이 있다.

---

### **(2) 요인 수 결정 (`dfm_select_factors.py`)**

#### **- 데이터 입력**

DFM은 구별 표준화 실질가격을 wide matrix 형태(T×N)로 입력받는다.

```python
df_wide = pd.read_csv("data/panel_wide_matrix.csv", index_col=0)
df_wide.index = pd.to_datetime(df_wide.index)
```

#### **- Grid Search 기반 모형 선택 절차**

코드는 최대 요인 수 5개, factor order=1~2 범위에서 AIC/BIC 비교를 위한 그리드 탐색을 포함하고 있다.

핵심 구조는 다음과 같다.

```python
model = DynamicFactor(
    df_wide,
    k_factors=k,
    factor_order=p,
    error_order=0
)
res = model.fit(maxiter=300, disp=False)
results.append((k, p, res.aic, res.bic))
```

#### **- 최종 선택된 모형**

코드에서는 사전 탐색 결과에 기반해 다음 조합을 채택한다.

* **k = 1 (공통 요인 1개)**
* **p = 2 (factor AR order 2)**


이는 전체 주택시장 움직임이 단일 주기 구조로 충분히 설명된다는 경험적 결과에 부합한다.

---

### **(3) 공통 요인 추출 (`dfm_common_factor_cycle.py`)**

#### **- 최종 DFM 적합**

```python
final_model = DynamicFactor(
    df_wide,
    k_factors=best_k,
    factor_order=best_p,
    error_order=0
)
final_res = final_model.fit(maxiter=300, disp=False)
```

#### **1) 필터링된 공통 요인 시계열 저장**

```python
factors = pd.DataFrame(
    final_res.factors.filtered.T,
    index=df_wide.index,
    columns=[f"Factor{i+1}" for i in range(k)]
)
factors.to_csv("data/dfm_common_factors.csv")
```

#### **2) 개별 구 로딩(loadings) 추출**

```python
params = np.asarray(final_res.params)
loadings = params[:N * k].reshape(N, k)
```

#### **3) 전체 공통 구성 요소 재구성**

```python
common_part = factors.values @ loadings.T
```

#### **4) 고유(idiosyncratic) 구성 요소 분리**

```python
idiosync = df_wide.values - common_part
idiosync_df = pd.DataFrame(idiosync, index=df_wide.index, columns=df_wide.columns)
idiosync_df.to_csv("data/dfm_idiosyncratic_components.csv")
```

#### **5) 공통 요인의 사이클 분석 절차**

공통 요인 Factor1에 대해 다음 절차로 시장 사이클을 분석했다.

##### **(a) STL 분해**

```python
stl = STL(series, period=12)
res = stl.fit()
trend = res.trend
seasonal = res.seasonal
```

##### **(b) HP 필터 기반 cycle 분리**

```python
cycle, trend_hp = hpfilter(values, lamb=129600)
```

##### **(c) FFT 기반 주기 탐색**

실제 코드 추출:

```python
fft_result = np.fft.fft(cycle)
freqs = np.fft.fftfreq(N)

Pxx = np.abs(fft_result[1:N//2])**2
top_idx = np.argmax(Pxx) + 1
dominant_period = 1 / freqs[top_idx]
m = int(round(dominant_period))
```

이 과정을 통해 **m개월 단위의 주요 시장 주기**를 도출한다.

##### **(d) Welch 검증으로 주기 안정성 점검**

```python
freqs_w, psd_w = welch(values, nperseg=nperseg)
```

##### **(e) Ruptures 변화점 탐지**

```python
algo = rpt.Binseg(model="rbf").fit(values)
change_points = algo.predict(n_bkps=optimal_k)
change_dates = series.index[np.array(change_points[:-1]) - 1]
```

---

### **(4) 요인 해석**

추출된 **Factor1**은 다음 특징을 갖는 전국적 공통 움직임을 나타낸다.

#### **① 시장 전체의 방향과 주기**

* HP 필터와 FFT 분석 결과, 특정 개월 수(m개월) 단위의 반복적 상승·하락 구조가 관찰됨
* 웰치 스펙트럼 기반으로 주요 주기의 안정성을 추가 검증함
* 이는 금리 변화, 유동성, 경기 변동 등 거시충격이 서울 전체 주택시장에 미치는 **동조적 사이클**을 반영

#### **② STL Trend를 통한 장기 구조**

* STL trend는 비주기적 장기 상승·하락 흐름(예: 장기적 완만한 하락 → 반등 국면 등)을 비교적 매끄럽게 보여줌
* FFT 기반 cycle과 대비하여 장기·단기 구조를 분리하여 해석 가능

#### **③ 변화점 분석을 통한 국면 전환 탐지**

* Ruptures 변화점 탐지에서 도출된 날짜는 시장 회복 또는 급변 시점과 일관된 패턴
* 변화점은 **사이클의 국면 전환점(peak/trough 사이의 변곡구간)**으로 해석 가능

#### **④ 지역별 영향도 분리(Loadings)**

* 로딩 행렬(loadings)을 통해 각 구가 공통 요인에 얼마나 민감하게 반응하는지 확인
* 로딩이 큰 지역일수록 시장 전체의 국면 변화에 높은 민감도를 보임
* idiosyncratic component를 통해 지역별 고유 패턴을 명확히 분리할 수 있게 됨

---

## **4.4. DTW 기반 구별 패턴 클러스터링**

### **(1) 목적**

동일한 기간 동안 서울 25개구의 아파트 매매가격 시계열이 **어떤 구조적 패턴을 공유하는지** 확인하기 위해 계층적 클러스터링을 수행하였다.
특히 일반적인 유클리드 거리보다 **시계열 형태의 변화 양상(위상 차이, 시간축 이동 등)** 을 더 잘 반영하는 **Dynamic Time Warping(DTW)** 거리를 사용하여,
  - 원시 시계열(panel),
  - DFM 기반 특이 성분(idiosyncratic component),
  - HP-filter cycle

세 가지 관점에서 구별 유사도를 비교하였다.

이 분석을 통해,

* 구별 장기적 흐름(Trend)과
* 국지적 순환(Cycle),
* 지역 고유 충격(Idiosyncratic)에 대한
  **구조적 패턴 그룹**을 확인하는 것이 목적이다.

---

### **(2) 알고리즘**

#### **- Robust Scaling**

각 구의 시계열 수준 차이(평균, 분산)가 클러스터링에 과도하게 영향을 주지 않도록,
**중앙값(median)과 IQR 기반 스케일링**을 적용하였다.

[
x' = \frac{x - \tilde{x}}{\mathrm{IQR}(x)}
]

이 처리는 강남·송파처럼 가격 수준이 높은 지역과 금천·중랑처럼 낮은 지역 간의 스케일 차이를 제거하여 **형태(pattern)만 비교**할 수 있게 한다.

---

#### **- DTW Distance Matrix 생성**

`tslearn.metrics.cdist_dtw` 를 이용해
각 구 시계열 간 DTW 거리를 계산하였다.

* DTW는 시점이 조금씩 비틀린 시계열(예: 상승 시기 차이)을 유클리드 거리보다 잘 측정
* 계산된 거리행렬은 **정규화(max scaling)** 하여 클러스터링 안정성을 높임

거리행렬은

* Panel
* Idiosyncratic
* HP-cycle
  각각에 대해 독립적으로 생성하였다.

---

#### **- Hierarchical Clustering + Silhouette 기반 최적 k 선택**

클러스터 구조는 계층적 클러스터링(average linkage)을 사용하였다.

[
Z = \text{linkage}(D_{\text{DTW}},\ \mathrm{method}="average")
]

k=2~9까지 반복하며 실루엣 점수(silhouette score)를 계산하여
**최적 클러스터 개수(best k)** 를 선택하였다.

각 방법(panel, idio, hp)마다 서로 다른 k가 선택될 수 있으며,
이는 동일한 지역군이라도 보는 관점에 따라 **패턴의 해석이 달라짐**을 의미한다.

---

### **(3) 핵심 코드 (`dtw_clustering.py`)**

#### **- Robust scaling**

```python
def robust_scale_per_series(X):
    Xs = np.zeros_like(X, dtype=float)
    for i in range(X.shape[0]):
        row = X[i].astype(float)
        med = np.nanmedian(row)
        q25 = np.nanpercentile(row, 25)
        q75 = np.nanpercentile(row, 75)
        iqr = q75 - q25
        if (iqr == 0) or np.isnan(iqr):
            s = np.std(row)
            if s == 0 or np.isnan(s):
                s = 1.0
            iqr = s
        Xs[i] = (row - med) / iqr
    return Xs
```

---

#### **- DTW 거리행렬 계산**

```python
def compute_dtw_distance_matrix_from_df(df, desc="DTW"):
    X = df.T.values
    Xs = robust_scale_per_series(X)

    print(f"{desc}: computing DTW (tslearn.cdist_dtw) ...")
    D = cdist_dtw(Xs)        # DTW distance
    D = np.nan_to_num(D)
    
    maxv = D.max()
    if maxv > 0:
        D = D / maxv        # normalization
    return D
```

---

#### **- 계층 클러스터링 및 silhouette 평가**

```python
def hierarchical_labels_from_distance(D, k, method="average"):
    condensed = squareform(D, checks=True)
    Z = linkage(condensed, method=method)
    labels = fcluster(Z, k, criterion="maxclust")
    return labels, Z
```

Main loop:

```python
for mname, D in methods.items():
    best_sil = -999
    best_k = None
    for k in K_RANGE:
        labels, Z = hierarchical_labels_from_distance(D, k=k)
        sil = silhouette_precomputed_safe(D, labels)
        if sil > best_sil:
            best_sil = sil
            best_k = k
            best_labels = labels
            best_Z = Z

    df_k = pd.DataFrame(rows)
    df_k.to_csv(f"Output/{mname}_k_search.csv")
```
---

### **(4) 해석**

#### **- Panel 기준**

패널 원시 시계열 기준의 DTW 구조는
상승·하락의 **전체 흐름이 유사한 지역군**을 묶는다.

예)

* “강남–서초–송파–강서–마포–서대문” 군
* “노원–성북” 군
* “강동–동대문–동작” 군 등

이는 **가격 수준 + 장기 흐름**이 동시에 반영된 결과로,
주요 상권·직주근접 형태가 반영되었다.

---

#### **- Idiosyncratic 기준**

DFM 특이성분은 각 구의 **고유 충격·단기 모멘텀**을 반영하므로
클러스터는 더 “지역적 특성”에 가까워진다.

예)

* 강남·서초·송파·양천처럼 **동조적 단기 모멘텀**
* 노원·도봉·성북 등 **북부권 단기 변동 그룹**

이는 “트렌드 제거 후 남는 지역 고유 패턴”을 기준으로
구별 특성을 파악하는 데 유효하다.

---

#### **- HP-cycle 기준**

HP-filter cycle은 **순환 성분**(약 2~4년 템포)을 기준으로 묶기 때문에
단기 상승/하락의 타이밍이 비슷한 지역군을 잘 드러낸다.

예)

* 강남·서초·송파·양천
* 노원·도봉·성북·은평
* 강서·마포

이는 **경기 민감도·정책 반응도** 등을 반영한다.

---

#### **- 종합 해석**

세 기준에서 반복적으로 나타나는 핵심 패턴:

* **강남·서초·송파(양천 포함)**
  → 동조성이 가장 높은 대표적 고가권 클러스터
* **노원·성북·도봉(은평 포함)**
  → 북부권 특유의 동행성과 모멘텀 패턴
* **강서·마포**
  → 서부권 빠른 순환 구조
* 일부 구는 일관적으로 **단독 행동**을 보임
  (종로, 영등포, 강북, 금천 등)

이 결과는 **구간별, 가격대별 시장 구조적 동조성**을 보여주는 근거가 되며
이후 상위 분석(변화점 분석, 시계열 segmentation 등)에도 활용 가능하다.

