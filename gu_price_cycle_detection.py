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
from matplotlib import rc

# -----------------------------
# 0. 기본 설정
# -----------------------------
# Windows 맑은 고딕 설정 (Mac은 'AppleGothic' 등으로 변경)
rc('font', family='Malgun Gothic')
plt.rcParams['axes.unicode_minus'] = False

# 데이터 불러오기
df = pd.read_csv("data/weighted_index_real.csv", parse_dates=['year_month'])
df.set_index('year_month', inplace=True)

# 분석 대상 선택 (첫 번째 구)
gu = df['구'].unique()[0]
series = df[df['구'] == gu]['real_price_index']
values = series.values

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

print(f"=== [{gu}] 데이터 기반 파라미터 자동 탐색 및 분석 시작 ===")

# -----------------------------
# 1. 고정 파라미터 적용 (STL, HP Filter)
# -----------------------------
# STL (period=12 고정)
stl = STL(series, period=12)
res = stl.fit()
trend = res.trend
seasonal = res.seasonal

# HP Filter (lamb=129600 고정)
cycle, trend_hp = hpfilter(values, lamb=129600)

# -----------------------------
# 2. [자동 탐색] m (Stumpy Window) - FFT 기반
# -----------------------------
# 주파수 분석 (0번 주파수는 제외)
f, Pxx = periodogram(cycle)
top_idx = np.argmax(Pxx[1:]) + 1  # 가장 파워가 센 주파수 인덱스
dominant_period = 1 / f[top_idx]  # 주파수 -> 주기 변환

# m 결정: FFT 주기와 최소 12개월 중 큰 값 선택 (너무 짧은 노이즈 방지)
m = int(max(12, round(dominant_period)))
print(f" >> [FFT 분석 결과] 감지된 최강 주기: {dominant_period:.1f}개월 -> 결정된 m: {m}")

# -----------------------------
# 3. [자동 계산] nperseg (Welch) - m 연동
# -----------------------------
# m의 3배를 기본으로 하되, 데이터 길이보다는 작아야 함
nperseg = min(len(values), m * 3)
freqs_w, psd_w = welch(values, nperseg=nperseg)
print(f" >> [Welch 설정] nperseg: {nperseg} (m의 {nperseg/m:.1f}배)")

# -----------------------------
# 4. Stumpy (Matrix Profile) - 결정된 m 사용
# -----------------------------
mp = stumpy.stump(values, m)
discord_idx = np.argmax(mp[:, 0])
discord_date = series.index[discord_idx]
print(f" >> [Stumpy] Discord(특이 패턴) 발견일: {discord_date.strftime('%Y-%m')}")

# -----------------------------
# 5. [자동 탐색] Ruptures - Elbow Method (K-means 방식)
# -----------------------------

def find_elbow_point(costs):
    """
    오차 비용 곡선에서 가장 많이 꺾이는 지점(Elbow)을 찾는 함수
    (직선에서 가장 멀리 떨어진 점을 찾는 원리)
    """
    n_points = len(costs)
    all_coords = np.vstack((range(n_points), costs)).T
    first_point = all_coords[0]
    line_vec = all_coords[-1] - all_coords[0]
    line_vec_norm = line_vec / np.sqrt(np.sum(line_vec**2))
    
    vec_from_first = all_coords - first_point
    scalar_product = np.sum(vec_from_first * line_vec_norm, axis=1)
    vec_from_first_parallel = np.outer(scalar_product, line_vec_norm)
    vec_to_line = vec_from_first - vec_from_first_parallel
    dist_to_line = np.sqrt(np.sum(vec_to_line ** 2, axis=1))
    
    # 0부터 시작하므로 +1 해줘야 k(개수)가 맞습니다.
    return np.argmax(dist_to_line) + 1 

# 탐색할 변화점 개수 범위 (1개 ~ 15개)
n_bkps_candidates = range(1, 16)
costs = []
algo_binseg = rpt.Binseg(model="rbf").fit(values)

for k in n_bkps_candidates:
    bkps = algo_binseg.predict(n_bkps=k)
    cost_value = algo_binseg.cost.sum_of_costs(bkps)
    costs.append(cost_value)


# Elbow Method 실행
optimal_n_bkps = find_elbow_point(costs)

# 최적 k로 다시 예측
change_points = algo_binseg.predict(n_bkps=optimal_n_bkps)
# Ruptures 결과는 0-based index이므로 날짜 매핑 시점 확인
change_dates = series.index[np.array(change_points[:-1]) - 1] # 마지막 포인트(끝점) 제외

print(f" >> [Ruptures] Elbow Method로 찾은 최적 변화점 개수: {optimal_n_bkps}개")

# -----------------------------
# 6. 이상치 탐지 (고정 파라미터)
# -----------------------------
X = values.reshape(-1, 1)

# IsolationForest (contamination=0.05)
iforest = IsolationForest(contamination=0.05, random_state=42)
outliers_if = iforest.fit_predict(X)
outlier_dates_if = series.index[outliers_if == -1]

# LOF (n_neighbors=12)
lof = LocalOutlierFactor(n_neighbors=12)
outliers_lof = lof.fit_predict(X)
outlier_dates_lof = series.index[outliers_lof == -1]

# -----------------------------
# 7.1. 메인 그래프 시각화 (첫 번째 독립 창)
# -----------------------------
# 독립적인 Figure 객체 생성
plt.figure(figsize=(14, 8)) 

# 메인 그래프 내용
plt.plot(series, label='Original Price Index', color='black', alpha=0.6)
plt.plot(trend, label='STL Trend (Long-term)', color='blue', linewidth=2, alpha=0.8)
plt.plot(cycle + trend, label='HP Filter (Smoothed)', color='orange', linestyle='--', alpha=0.8)

# 이벤트 마커
plt.scatter(discord_date, series.loc[discord_date], color='red', label=f'Discord (Stumpy m={m})', s=150, zorder=5, marker='*')
plt.scatter(change_dates, series.loc[change_dates], color='purple', label=f'Change Points (k={optimal_n_bkps})', marker='v', s=100, zorder=5)
plt.scatter(outlier_dates_if, series.loc[outlier_dates_if], color='green', label='IsoForest (Top 5%)', marker='o', s=40, alpha=0.5)
plt.scatter(outlier_dates_lof, series.loc[outlier_dates_lof], color='cyan', label='LOF (Local)', marker='x', s=40, alpha=0.8)

plt.title(f"[{gu}] 부동산 실질가격 종합 분석 (Auto-Tuned Parameters)", fontsize=16)
plt.xlabel("Year-Month")
plt.ylabel("Real Price Index")
plt.legend(loc='upper left', bbox_to_anchor=(1, 1))
plt.grid(True, which='both', linestyle='--', alpha=0.5)
plt.tight_layout() # 간격 조정

plt.show()