목차 만들고 클릭시 목차로 빠르게 이동하는 기능 본문에서 제목 누르면 최 상단 목차 테이블로 이동


# **Data Description**

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

# **Preprocessing**

## **1. 두 시트 결합 및 날짜 생성**

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

## **2. 주소 분해: 구/동 추출**

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

## **3. 수치형 변수 정제**

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

## **4. 분석용 컬럼 정리 및 저장**

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

## **5. 전용면적 및 노후도 버킷팅(Bucketing)**

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

## **6. IQR 기반 이상치 제거**

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

## **7. 월별 집계(Monthly Aggregation)**

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

## **8. 가중 지수 계산(Weighted Price Index)**

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

## **9. 실질 가격 변환(Real Price Index Conversion)**

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

# 📘 **Methodology**

본 연구의 방법론은 총 네 단계로 구성된다:
① 개별 구 단위의 가격 사이클 탐지,
② 패널 데이터 구조 구축 및 패널 회귀 시도,
③ Dynamic Factor Model(DFM)을 통한 공통요인 추출,
④ 시계열 간 유사도 기반 클러스터링(DTW)이다.

이 순서는 **단계적으로 더 높은 구조를 포착하는 방향**으로 설계되었다.
먼저 지역별 고유 패턴을 파악하고, 이후 이를 패널 구조로 통합해 설명가능성을 평가하며, 마지막으로 공통요인을 도출하고 클러스터링으로 패턴을 분류하는 방식이다.

---

## **1. 개별 구의 가격 사이클 및 변곡점 탐지**

### 1.1 목적

각 구의 아파트 가격은 독립적으로 변동하기보다 상승-하락의 주기를 갖는다.
이를 수학적으로 정의하기 위해, 각 구별 시계열에서

* 국소 최대값(peak)
* 국소 최소값(trough)
* 사이클 길이
* 사이클 강도
  를 탐지하였다.

## 1.2 주요 알고리즘

* SciPy의 `find_peaks` 이용
* prominence, distance, height 조건 조정
* peak / trough 간 간격으로 "사이클 길이" 계산
* 분기별 smoothing 적용해 잡음 제거

## 1.3 대표 코드

아래 코드는 `all_of_gu_price_cycle_detection.py`에서 구별 사이클을 계산하는 핵심 부분이다.

```
# 각 구별로 price cycle 탐지
for gu, series in gu_time_series.items():
    y = series.values

    # 피크 탐지
    peaks, _ = find_peaks(y, prominence=0.05, distance=3)
    troughs, _ = find_peaks(-y, prominence=0.05, distance=3)

    cycles = []
    for p, t in zip(peaks, troughs):
        cycle_length = abs(t - p)
        cycles.append(cycle_length)

    result[gu] = {
        "peaks": peaks.tolist(),
        "troughs": troughs.tolist(),
        "cycles": cycles,
    }
```

## 1.4 결과 활용

이 단계는 이후 DFM에서 구별 로딩(loadings) 비교 시 각 지역의 고유 변동성과 연결해 해석하는 데 사용된다.

---

# **2. 패널 데이터 구조 구축 및 패널 회귀 분석**

## 2.1 목적

서울 25개 구의 가격지수는 동일 시점에 공통 충격(금리, 경기 등)을 받을 수 있다.
이를 검증하기 위해 각 구를 **패널 데이터**로 구성하여 고정효과/랜덤효과 회귀를 수행하였다.

## 2.2 패널 데이터 구축

`prepare_panel_timeseries.py`에서는 다음 구조를 가진 데이터셋을 생성한다.

* entity: 구(25개)
* time: 월 단위(236개)
* value: 실질 가격의 표준화 지수(real_std)
* factor 후보 변수: 공통 요인 proxy

대표 코드:

```
panel_df = pd.DataFrame({
    "gu": all_series.index.get_level_values(0),
    "year_month": all_series.index.get_level_values(1),
    "real_std": all_series.values,
})

panel_df = panel_df.set_index(["gu", "year_month"])
panel_df = panel_df.sort_index()
```

## 2.3 패널 회귀 시도 (`panel_analysis.py`)

아래는 핵심 회귀 부분이다.

```
mod = PanelOLS(
    panel_df["real_std"],
    panel_df[["Factor1"]],
    entity_effects=True,
    time_effects=True
)

res = mod.fit(cov_type="clustered")
print(res.summary)
```

### 2.4 시도 결과 및 한계

회귀 결과는

* 계수 비유의미
* F-test p-value = 1
* Between R² 음수
  등으로 매우 불안정하였다.

이는
① 각 구의 변동성이 비선형적이고
② 단일 요인으로 설명이 어렵고
③ 패널 회귀 가정(독립성·정상성)을 위배하는
등의 구조적 한계로 판단된다.

**따라서 본 연구는 패널 회귀 대신 공통요인을 직접 추출하는 DFM 단계로 넘어간다.**

---

# **3. Dynamic Factor Model(DFM) 분석**

## 3.1 목적

25개 구의 가격 변동에서

* 모든 구가 공유하는 “서울 전체 공통 요인”
* 각 구 고유 idiosyncratic shock
  을 분리하여 구조적 패턴을 파악하는 것이 핵심 목표다.

## 3.2 요인 수 결정 (`dfm_select_factors.py`)

AIC/BIC 최소값 기준, cumulative variance 기준 등을 조합해 최적의 q개 요인을 선택한다.

대표 코드:

```
bic_scores = []
for q in range(1, 6):
    model = DynamicFactor(df_z, k_factors=q, factor_order=1)
    res = model.fit()
    bic_scores.append(res.bic)
```

## 3.3 공통 요인 추출 (`dfm_common_factor_cycle.py`)

아래는 공통 요인을 추정하는 주요 부분이다.

```
model = DynamicFactor(df_z, k_factors=1, factor_order=1)
res = model.fit()

common_factor = res.factors.filtered["factor.1"]
loadings = res.params.filter(like="factor_loading")
```

## 3.4 요인 해석

* Common factor는 **서울 전체의 경기 사이클을 반영**
* 각 구 로딩값은 “공통적인 충격에 대한 민감도”
* idiosyncratic variance는 지역 고유 충격의 비중

예:
“85.18%가 공통요인으로 설명됨” → 서울 시장이 고도로 동조화됨.

## 3.5 Cycle extraction from common factor

추출된 공통요인은 다시 peak-trough 분석에 사용되어 전체 사이클(버블-조정-반등)을 해석한다.

---

# **4. DTW 기반 구별 패턴 클러스터링**

## 4.1 목적

서울 25개 구는 공통 요인을 공유하더라도, **동일 타이밍에 동일한 패턴으로 움직이지 않는다.**
그래서 시계열 간 유사도를 비교하는 DTW를 사용해 **가격 흐름이 비슷한 지역끼리 군집화**하였다.

## 4.2 알고리즘

* Dynamic Time Warping (DTW)
* Hierarchical clustering
* Ward linkage
* distance matrix 기반 클러스터링

## 4.3 핵심 코드 (`dtw_clustering.py`)

```
for i, gu_i in enumerate(gus):
    for j, gu_j in enumerate(gus):
        dist = dtw(gu_series[gu_i], gu_series[gu_j]).distance
        D[i, j] = dist
```

클러스터링:

```
Z = linkage(D_condensed, method="ward")
clusters = fcluster(Z, k, criterion="maxclust")
```

## 4.4 해석

* 클러스터 구조를 통해 강남권·도심권·외곽권의 구조적 차이가 드러난다.
* 이는 DFM에서 나타난 factor loading과 결합해 지역별 특성을 더 입체적으로 해석하는 데 사용한다.

---

# 📌 Methodology 전체 요약 흐름

1. **구별 사이클 탐지**
   → 지역 고유의 변동 패턴을 먼저 파악
2. **패널 분석 시도**
   → 회귀로 설명 가능한가? → 한계 확인
3. **Dynamic Factor Model**
   → 공통 요인 추출 & 구조적 사이클 도출
4. **DTW 클러스터링**
   → 각 지역의 패턴을 그룹화하여 구조적 차이 파악

