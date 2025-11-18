1 노션에 적어둔 목표 구체화와 이전까지 진행한 것들 다시 읽기
    사이클(이유), 특이점(이유)

2 깃을 사용하면서 정리한다로 끝냄. 현재 브랜치가 어디인가. 
    merge 해봄
    vs code에서 git log 보는 방법 
    commit convention을 따르기로 하자. 이거 만들어둔 대화방 찾아 gpt? gemini? -> memo
    merge 넣고 커밋, commit을 vs code의 gui으로 하는 방법

3 어떻게 절차화 하는가(뼈대, 파편화, 순서)
https://chatgpt.com/c/68cdfdc5-9600-832e-a739-2283c1cbe469
구글 놑북 lm에 넣어서?

    내가 하고 싶은 것은 다음과 같아.
    끝 부분에 시각화까지 되어 있는지
    git 과 함께 작업을 한다고 할 때 브랜치는 몇개를 어떻게 만들면 좋을까? 그냥 보면서 이거 하려고 하는데 어떤 이름 브랜치? 혹은 쪼개서 더 많이 가져갈 필요 있어? 묻기
    이것과 관련해서 구체화 시키는 것을 Chat gpt와 같은 llm과 함께 하는 것 그리고 Google Notebook lm과 함께 하는 것 중 추천은? : notebookLM은 시험 볼 때 들어가고, 지금은 gpt로?

4 코드 진행을 클로드 코드 혹은 코파일럿과 함께 진행
    일단 코파일럿 + gpt, gemini로 진행하고 부족하거나 답이 안 보이면 클로드 코드 건들여 봐?
    클로드 코드 from scratch: https://codegear.tistory.com/146

**🦎 check point **

5 데이터 정제
    필요한 브랜치 만들기 (일단 현재 커밋, 푸시)
    현재 있는 것 내버려 두고 데이터를 복사해서 정제하기: 양이 너무 많아서 10년치씩 두개로 분리한 거임
    필요 없는 칼럼 삭제: NO, 번지, 본번, 부번, 동, 층, 매수자, 매도자, 해제사유발생일, 거래유형, 중개사소재지, 등기일자, 주택유형 column 삭제

    나머지 칼럼들에 대해서 gpt가 이야기해준 흐름대로 진행
        일단 순서는 다음과 같이 할거임: date, 구, 동, 전용면적(㎡), 거래금액(만원), 건축년도, 단지명, 도로명
        preprocess_apt_sale.py 만들어서 excel -> csv

## 목표와 절차

서울 국토교통부에서 2006년부터 2025년까지 아파트 매매 거래 데이터를 가지고 왔거든. 
“가격 사이클”과 “특이점(급등/급락)”을 동시에 잡으려면, 

**데이터 정제 → 지수화(믹스 보정) → 사이클 추출 → 특이점 탐지 → 사건 매핑 → 군집/요인 분석** 

서울시 전체를 묶어서 탑다운으로 한 번 보고, 그 다음에 구별로 묶어서 들어간다.

---

## 로드맵과 코드 틀

# 0) 데이터 조직 & 기본 전처리

**목표:** 월별·구별로 “믹스변화 영향이 최소화된 가격 시계열”을 만든다.
**해야 할 일**

1. **이상치 제거**: 동일 단지·동·층·전용면적 대비 3σ 바깥 가격 제거(또는 IQR).
2. **면적/준공연도 버킷팅**: 전용 **[<60, 60–85, 85–102, >102㎡]**·준공 **[<10년, 10–20, >20년]** 등으로 그룹.
3. **월별 구간화**: 거래일 → `YYYY-MM`로 묶고, **구×버킷별 중앙값** 산출.
4. **구별 대표지수**: 가중평균(가중치=해당 버킷 거래건수)으로 **월별 구 가격지수** 생성.

   * 보너스: 동일 주택의 재거래가 충분하면 **Repeat-Sales(케이스–쉴러형)** 지수도 병행.
5. **물가 보정**: CPI로 실질가 계산(필수).

---

# 1) “사이클”을 수학적으로 정의

**현실적 정의 2종**을 함께 본다.

* **(A) 저주파 추세 + 계절 + 잔차**: `STL/HP filter` 기반 분해 → **주기(연간·수년)** 유무 확인.
* **(B) 주파수/밴드 관점**: 파워스펙트럼(FFT/Welch)·밴드패스(Baxter–King, Christiano–Fitzgerald)로 **2–7년대 사이클** 존재 여부 확인.

---

# 2) 특이점(급등/급락·레짐 전환) 탐지

* **급등/급락 “점”**: `matrix profile (stumpy)`로 **discord/전형적이지 않은 subsequence** 탐지.
* **평균/추세 단절 “구간”**: `ruptures (Pelt/Binseg)`로 **변화점(수준·기울기·분산)** 검출.
* **이상치 스코어**: IsolationForest/LocalOutlierFactor로 월 수치의 비정상성도 병행.

---

# 3) 사건 매핑(해석 단계)

* **금리·대출규제·세제·공급이벤트(분양·입주 물량)·거시충격(코로나, 전쟁, 환율 급등)** 타임라인과 **변화점/급등점**을 매칭.
* **선행·동행 검사**: 사건 대비 ±k개월 윈도우의 수익률 분포/변화율을 비교, **효과량**과 **시차** 추정.

  * 통계: 이벤트 스터디(누적수익률) + 부트스트랩 CI.

---

# 4) 25개구를 “각각 vs 함께” 어떻게 볼까?

**정답은 “둘 다”**다. 이유와 방법:

1. **단일구(로컬) 해부**: 각 구에서 사이클/변화점을 엄밀히 잡아 **지역별 개성** 파악.
2. **공동(패널) 모델**: 25개구를 동시에 넣어 **공통 요인**과 **지역 고유 요인**을 분리.

   * **Dynamic Factor Model(DFM)** 또는 **State-Space(공통추세+개별잔차)**
   * **Hierarchical regression**으로 정책·금리·공급 변수의 **구별 상이한 민감도** 추정.
3. **동조성/유사도**: DTW 거리로 **사이클 동기화** 측정 → K-means/계층 군집으로 **권역 클러스터** 도출.

---

# 5) Python 스택 제안

* **데이터/시각화**: `pandas`, `numpy`, `matplotlib`(또는 `plotly`), `seaborn`
* **필터/분해**: `statsmodels`(STL, HP, CF/BK), `scipy.signal`(periodogram/Welch)
* **변화점/이상치**: `ruptures`, `stumpy`(matrix profile), `scikit-learn`(IF/LOF)
* **동조·클러스터**: `tslearn`(DTW, TimeSeriesKMeans)
* **요인/상태공간**: `statsmodels.tsa.statespace`(DynamicFactor, UnobservedComponents)
* **원인 탐색**: `linearmodels`(패널 회귀), `statsmodels`(ARDL/Granger)

---

# 6) 스타터 코드 스니펫

## 6.1 월별·구별 지수 만들기(믹스 보정)

```python
import pandas as pd
import numpy as np

df = pd.read_csv("seoul_apts_2006_2025.csv", parse_dates=["contract_date"])
# 필수 컬럼 예시: ['gu','danji','dong','floor','area_sqm','year_built','price','contract_date']

# 버킷팅
bins_area = [-np.inf, 60, 85, 102, np.inf]
labels_area = ["<60","60-85","85-102",">102"]
df["area_bucket"] = pd.cut(df["area_sqm"], bins=bins_area, labels=labels_area)

bins_age = [-np.inf, 10, 20, np.inf]
labels_age = ["<10y","10-20y",">20y"]
df["age_years"] = df["contract_date"].dt.year - df["year_built"]
df["age_bucket"] = pd.cut(df["age_years"], bins=bins_age, labels=labels_age)

# 월 키
df["ym"] = df["contract_date"].dt.to_period("M").dt.to_timestamp()

# 그룹별 중앙값
g = df.groupby(["gu","ym","area_bucket","age_bucket"], observed=True)["price"]
med = g.median().rename("price_med").reset_index()
cnt = g.size().rename("n").reset_index()

mix = med.merge(cnt, on=["gu","ym","area_bucket","age_bucket"], how="left")

# 구별 대표지수(거래수 가중 평균)
def weighted_median_like(x):
    # 가중 '중앙값 유사'로 단순화: 가중평균 사용(현업은 더 정교화 가능)
    return np.average(x["price_med"], weights=x["n"])

idx = (mix.groupby(["gu","ym"])
          .apply(weighted_median_like)
          .rename("gu_price_index")
          .reset_index())

# 물가보정(CPI series 준비 가정 cpi_df: columns ['ym','cpi'])
# idx = idx.merge(cpi_df, on="ym", how="left")
# idx["real_index"] = idx["gu_price_index"] / idx["cpi"] * 100
```

## 6.2 STL 분해 & 스펙트럼으로 주기 보기

```python
from statsmodels.tsa.seasonal import STL
from scipy.signal import welch

def stl_cycle(x, period=12):
    res = STL(x, period=period, robust=True).fit()
    return res.trend, res.seasonal, res.resid

gu = "Gangnam-gu"
series = idx[idx["gu"]==gu].set_index("ym")["gu_price_index"].asfreq("MS").interpolate()

trend, seas, resid = stl_cycle(series, period=12)

# 스펙트럼(연 주기 후보)
fs = 12 # monthly
freqs, psd = welch(series.values, fs=fs, nperseg=min(256, len(series)))
# 주기(개월) = fs / freq
periods_month = np.where(freqs>0, fs/freqs, np.nan)
```

## 6.3 변화점/급등점

```python
import ruptures as rpt
import stumpy

# 변화점: 수준+기울기 모델
model = "rbf"  # or "l2", "linear"
algo = rpt.Pelt(model=model).fit(series.values)
# penalty(람다) 튜닝 필요
bkps = algo.predict(pen=1e7)

# 급등/급락: matrix profile로 discord 찾기
m = 6  # subsequence window (개월)
mp = stumpy.stump(series.values, m)
discord_idx = np.argmax(mp[:,0])  # 최상위 discord 시작점
```

## 6.4 25개구 동조성/군집

```python
from tslearn.clustering import TimeSeriesKMeans
from tslearn.metrics import dtw
from tslearn.utils import to_time_series_dataset

wide = (idx.pivot(index="ym", columns="gu", values="gu_price_index")
          .asfreq("MS").interpolate(limit_direction="both"))
X = to_time_series_dataset([wide[c].values for c in wide.columns])

km = TimeSeriesKMeans(n_clusters=4, metric="dtw", random_state=42)
labels = km.fit_predict(X)
cluster_map = dict(zip(wide.columns, labels))
```

## 6.5 공통 요인 분리(DFM)

```python
from statsmodels.tsa.statespace.dynamic_factor import DynamicFactor

Y = wide.apply(np.log).diff().dropna()  # 수익률로 모델링 예시
mod = DynamicFactor(Y, k_factors=1, factor_order=1, error_var=True)
res = mod.fit(method='em', disp=False)
# 공통요인(res.fittedvalues, res.factors.filtered) + 구별 특이요인 분리
```

---

# 7) 결과 해석 순서(워크플로)

1. **구별 지수 플롯 + STL**: 계절성 유무, 장기 추세, 잔차 범위 파악.
2. **Welch/FFT**: **사이클 주기 후보(예: ~36–60개월)**를 숫자로 확인.
3. **밴드패스 필터**: 해당 주기 대역만 남겨 “사이클 성분” 시계열 시각화.
4. **변화점/급등점 표**: 각 구에서 발생 시점과 강도(전후 수익률 차).
5. **사건 매핑**: 금리/정책/공급 타임라인과 매칭, **선행/동행** 판정.
6. **동조성/군집**: 권역별로 비슷하게 움직이는지, 어느 구가 “선행”하는지(상관의 시차 최대).
7. **DFM/패널**: 공통 요인 vs 지역 특이 요인의 **기여율** 수치화.
8. **리포트 아웃**: 구별 카드(사이클 주기·변화점·주요 이벤트·민감도).

---

# 8) “각 구 따로 vs 한 번에”에 대한 결론

* **탑다운+보텀업 혼합**이 최적.

  * 보텀업(각 구 개별 분석)으로 **고유 패턴·지역 이슈**를 놓치지 않는다.
  * 탑다운(DFM/패널)으로 **서울 공통 사이클**과 **정책/금리의 공통 충격**을 정량화한다.
* 실무 팁: 먼저 **강남·서초·송파·마용성** 등 대표 구로 **분석 파이프라인을 안정화** → 25개구 확장 → 군집 결과로 **대표 구**만 깊게 모니터링하는 체계로 운영.

---

# 9) 체크리스트(현업 품질)

* [ ] **주소/단지 정규화**(동일 단지 키 생성)
* [ ] **거래취소/정정 레코드 반영**
* [ ] **물가·금리·주택공급·정책 변수를 외생 피처로 별도 테이블화**
* [ ] **신규 분양 대량 입주 시점(입주물량), 재건축 이슈 타임라인**
* [ ] **리샘플링/결측 보간 방식 고정**(forward-fill 금지, 선형/LOESS)
* [ ] **모델/하이퍼파라미터 고정 및 리포팅 자동화**(구별 YAML)

---

필요하면 **분석 노트북 템플릿**(데이터 로드 → 지수화 → STL/스펙트럼 → 변화점/discord → 군집 → DFM → 이벤트 스터디)을 바로 만들어줄게.
또, 사건 매핑용으로 **정책/금리/입주물량 타임라인**을 깔끔히 정리해 연결하는 표도 만들어주겠다.

다 끝나면 시각화 해서 보고서로 제출해야 함. 흐름을 논리적으로 따라서 작성한 글 하나와 인터렉티브하게 결과들을 가지고 놀 수 있게 하는 html(?) page 하나