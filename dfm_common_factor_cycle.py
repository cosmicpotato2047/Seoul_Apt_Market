import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.seasonal import STL
from statsmodels.tsa.filters.hp_filter import hpfilter
from scipy.signal import welch
import ruptures as rpt
from matplotlib import rc
import matplotlib.dates as mdates
import os
os.makedirs("figure", exist_ok=True)

# -----------------------------------------
# 0. 기본 설정
# -----------------------------------------
rc('font', family='Malgun Gothic')
plt.rcParams['axes.unicode_minus'] = False

# -----------------------------------------
# 1. 공통 요인 불러오기
# -----------------------------------------
df = pd.read_csv("data/dfm_common_factors.csv", parse_dates=['year_month'])
df.set_index('year_month', inplace=True)

series = df['Factor1']
values = series.values
N = len(series)

print("=== 공통 요인(Factor1) 기반 전체 시장 분석 시작 ===")

# -----------------------------------------
# 2. STL 분해
# -----------------------------------------
stl = STL(series, period=12)
res = stl.fit()
trend = res.trend
seasonal = res.seasonal

# -----------------------------------------
# 3. HP Filter
# -----------------------------------------
cycle, trend_hp = hpfilter(values, lamb=129600)

# -----------------------------------------
# 4. FFT 기반 주기 탐색
# -----------------------------------------
fft_result = np.fft.fft(cycle)
freqs = np.fft.fftfreq(N)

Pxx = np.abs(fft_result[1:N//2])**2
top_idx = np.argmax(Pxx) + 1

dominant_period = 1 / freqs[top_idx]
m = int(round(dominant_period))

print(f" >> FFT 기반 탐지 주기 = {dominant_period:.1f}개월 (m={m})")

phase_rad = np.angle(fft_result[top_idx])

# -----------------------------------------
# 5. Welch 기반 주기 안정성 점검
# -----------------------------------------
nperseg = min(N, m * 3)
freqs_w, psd_w = welch(values, nperseg=nperseg)

print(f" >> Welch nperseg = {nperseg}")

# -----------------------------------------
# 6. Ruptures 변화점 탐지
# -----------------------------------------
n_candidates = range(1, 10)
costs = []

algo = rpt.Binseg(model="rbf").fit(values)

for k in n_candidates:
    bkps = algo.predict(n_bkps=k)
    costs.append(algo.cost.sum_of_costs(bkps))

def find_elbow(costs):
    pts = np.vstack([range(len(costs)), costs]).T
    first, last = pts[0], pts[-1]
    line = last - first
    line_norm = line / np.sqrt(np.sum(line**2))
    vec = pts - first
    proj = np.outer(np.sum(vec * line_norm, axis=1), line_norm)
    dist = np.sqrt(np.sum((vec - proj)**2, axis=1))
    return np.argmax(dist) + 1

optimal_k = find_elbow(costs)
change_points = algo.predict(n_bkps=optimal_k)
change_dates = series.index[np.array(change_points[:-1]) - 1]

print(f" >> 변화점 개수 = {optimal_k}, 날짜 = {list(change_dates)}")

# -----------------------------------------
# 7. Figure 1 (Main)
# -----------------------------------------
fig, ax = plt.subplots(figsize=(12, 6))

x_data = np.arange(N)
cosine_wave = np.cos((2 * np.pi * x_data / m) + phase_rad) * cycle.std() * 0.5
m_cycle_line = trend_hp + cosine_wave

plt.plot(series, label='Common Factor (Factor1)', color='black')
plt.plot(trend, label='STL Trend', color='blue')
plt.plot(cycle + trend, label='HP Smoothed', color='orange')
plt.plot(series.index, m_cycle_line, label=f'Ideal Cycle (T={m} months)', color='red', linestyle=':')

plt.scatter(change_dates, series.loc[change_dates], color='purple', marker='v', label='Change Points', s=100)

ax.xaxis.set_major_locator(mdates.YearLocator(base=2))
ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
ax.xaxis.set_minor_locator(mdates.YearLocator())

plt.title("Common Factor (Factor1) - 시장 전체 분석")
plt.legend()
plt.grid(alpha=0.5)
plt.tight_layout()
fig.savefig("figure/common_factor_main.png", dpi=300)
plt.show()

# -----------------------------------------
# 8. Figure 2 (FFT)
# -----------------------------------------
fig2, ax2 = plt.subplots(figsize=(10, 5))
freqs_pos = freqs[1:N//2]
periods = 1 / freqs_pos

ax2.plot(periods, Pxx)
ax2.axvline(m, color='red', linestyle='--', label=f'm={m}')

ax2.set_xscale('log')
ax2.set_title("FFT Periodogram (Common Factor)")
ax2.set_xlabel("Period (Months)")
ax2.set_ylabel("Power")
ax2.legend()
ax2.grid(True, which='both', alpha=0.5)

plt.tight_layout()
fig2.savefig("figure/common_factor_fft.png", dpi=300)
plt.show()
