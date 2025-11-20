import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.seasonal import STL
from statsmodels.tsa.filters.hp_filter import hpfilter
from scipy.signal import periodogram, welch
import stumpy
import ruptures as rpt
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor

import matplotlib.pyplot as plt
from matplotlib import rc

# 예: Windows에서 맑은 고딕
rc('font', family='Malgun Gothic')
plt.rcParams['axes.unicode_minus'] = False  # 마이너스 기호 정상 출력

# -----------------------------
# 1. 데이터 불러오기
# -----------------------------
df = pd.read_csv("data/weighted_index_real.csv", parse_dates=['year_month'])
df.set_index('year_month', inplace=True)

# -----------------------------
# # 구 리스트
# gu_list = df['구'].unique()

# # -----------------------------
# # 2. 구별 분석 루프
# # -----------------------------
# for gu in gu_list:
#   print(f"=== {gu} 분석 ===")
#   series = df[df['구']==gu]['real_price_index']
# -----------------------------
gu = df['구'].unique()[0]   # 첫 번째 구 선택
series = df[df['구'] == gu]['real_price_index']
# -----------------------------

print(f"=== {gu} 분석 ===")
series = df[df['구']==gu]['real_price_index']

# -----------------------------
# 2-1. STL 분해
# -----------------------------
stl = STL(series, period=12)
res = stl.fit()
trend = res.trend
seasonal = res.seasonal
resid = res.resid

# -----------------------------
# 2-2. HP filter
# -----------------------------
cycle, trend_hp = hpfilter(series.values, lamb=129600)

# -----------------------------
# 2-3. FFT / Welch 주파수 분석
# -----------------------------
freqs, psd = periodogram(series)
freqs_w, psd_w = welch(series.values, nperseg=24)

# -----------------------------
# 2-4. stumpy discord 탐지
# -----------------------------
m = 12  # 12개월 서브시퀀스
mp = stumpy.stump(series.values, m)
discord_idx = np.argmax(mp[:,0])
discord_date = series.index[discord_idx]

# -----------------------------
# 2-5. ruptures 변화점 탐지
# -----------------------------
algo = rpt.Pelt(model="rbf").fit(series.values)
change_points = algo.predict(pen=10)
change_dates = series.index[np.array(change_points)-1]  # 인덱스 매핑

# -----------------------------
# 2-6. 이상치 점검 (IsolationForest / LOF)
# -----------------------------
X = series.values.reshape(-1,1)

# IsolationForest
iforest = IsolationForest(contamination=0.05, random_state=42)
outliers_if = iforest.fit_predict(X)
outlier_dates_if = series.index[outliers_if==-1]

# LOF
lof = LocalOutlierFactor(n_neighbors=12)
outliers_lof = lof.fit_predict(X)
outlier_dates_lof = series.index[outliers_lof==-1]

# -----------------------------
# 2-7. 시각화
# -----------------------------
plt.figure(figsize=(14,6))
plt.plot(series, label='Original', color='black')
plt.plot(trend, label='STL Trend', color='blue', linewidth=2)
plt.plot(cycle, label='HP Cycle', color='orange', linewidth=2)
plt.scatter(discord_date, series.loc[discord_date], color='red', label='Discord', s=100, zorder=5)
plt.scatter(change_dates, series.loc[change_dates], color='purple', label='Change Points', marker='x', s=100, zorder=5)
plt.scatter(outlier_dates_if, series.loc[outlier_dates_if], color='green', label='IsolationForest', marker='o', s=50, alpha=0.7)
plt.scatter(outlier_dates_lof, series.loc[outlier_dates_lof], color='cyan', label='LOF', marker='^', s=50, alpha=0.7)
plt.title(f"{gu} 월별 실질가격 지수 분석")
plt.xlabel("Year-Month")
plt.ylabel("Real Price Index")
plt.legend()
plt.tight_layout()
plt.show()

# -----------------------------
# 2-8. FFT 시각화 (보조)
# -----------------------------
# (선택사항) 이해하기 쉽게 X축을 '주기(개월)'로 바꿔서 확인해보고 결정하세요
plt.figure(figsize=(10,4))
# 주파수가 0인 건 제외하고(무한대 방지), 역수(1/x)를 취해 '개월 수'로 변환
plt.semilogx(1/freqs_w[1:], psd_w[1:]) 
plt.xlabel("Period (Months)") # X축이 이제 '개월'입니다
plt.ylabel("Power")
plt.title("주기별 강도 확인 (몇 개월 주기가 쎈가?)")
plt.show()
