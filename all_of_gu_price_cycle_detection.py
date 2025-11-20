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
import matplotlib.dates as mdates
import os
os.makedirs("figure", exist_ok=True)


# -----------------------------
# 0. 기본 설정
# -----------------------------
# Windows 맑은 고딕 설정 (Mac은 'AppleGothic' 등으로 변경)
rc('font', family='Malgun Gothic')
plt.rcParams['axes.unicode_minus'] = False

# 데이터 불러오기
df = pd.read_csv("data/weighted_index_real.csv", parse_dates=['year_month'])
df.set_index('year_month', inplace=True)

# 구 리스트
gu_list = df['구'].unique()

for gu in gu_list:
    print(f"=== {gu} 분석 ===")
    series = df[df['구']==gu]['real_price_index']
    values = series.values

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
    # 2. [자동 탐색] m (Stumpy Window) - FFT 기반 (위상 추출 포함)
    # -----------------------------
    N = len(cycle)

    # 1. FFT를 직접 수행하여 주파수와 복소수 결과(위상 정보 포함)를 얻습니다.
    fft_result = np.fft.fft(cycle)
    freqs = np.fft.fftfreq(N)

    # 2. 가장 파워가 센 주파수 인덱스를 찾아 m(주기)을 결정합니다.
    Pxx = np.abs(fft_result[1:N//2])**2 # 파워 계산 (단측 스펙트럼)
    top_idx = np.argmax(Pxx) + 1 # 0번 인덱스 제외, 단측 스펙트럼이므로 +1

    dominant_period = 1 / freqs[top_idx] 
    m = int(max(12, round(dominant_period))) # m 결정: FFT 주기와 최소 12개월 중 큰 값 선택 (너무 짧은 노이즈 방지)
    print(f" >> [HP Cycle FFT 분석] 감지된 순환 주기: {dominant_period:.1f}개월 -> 결정된 m: {m}")

    # 3. m 주기에 해당하는 FFT 성분의 인덱스를 사용하여 위상(Phase)을 추출합니다.
    # (dominant_period를 결정한 top_idx_in_full_spectrum을 사용해야 합니다.)
    # top_idx는 1부터 N/2-1 범위의 인덱스이므로, 전체 FFT 결과 배열에서 해당 인덱스를 사용합니다.
    phase_rad = np.angle(fft_result[top_idx]) 
    print(f" >> [위상 분석] 초기 위상(Phase shift): {phase_rad:.2f} rad")

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

    change_dates_list = [date.strftime('%Y-%m') for date in change_dates]
    count_cp = len(change_dates_list)

    print(f" >> [Ruptures] 감지된 Change Points (총 {count_cp}개):")
    for date_str in change_dates_list:
        print(f"    - {date_str}")
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

    # 1. IsolationForest (IForest) 결과 출력
    iforest_dates_list = [date.strftime('%Y-%m') for date in outlier_dates_if]
    count_if = len(outlier_dates_if)
    print(f" >> [IsolationForest] 탐지된 이상치 (총 {count_if}개, {iforest.contamination*100:.0f}% 기준):")
    # 날짜를 줄 바꿈하여 출력
    for date_str in iforest_dates_list:
        print(f"    - {date_str}")

    # 2. Local Outlier Factor (LOF) 결과 출력
    lof_dates_list = [date.strftime('%Y-%m') for date in outlier_dates_lof]
    count_lof = len(outlier_dates_lof)
    print(f" >> [LOF] 탐지된 이상치 (총 {count_lof}개, 이웃 N={lof.n_neighbors} 기준):")
    # 날짜를 줄 바꿈하여 출력
    for date_str in lof_dates_list:
        print(f"    - {date_str}")

    # -----------------------------
    # Figure 1: 메인 분석 그래프
    # -----------------------------

    fig, ax = plt.subplots(figsize=(12, 6))

    # m 주기 코사인파 생성 (HP Filter 추세선을 중심으로 주기 패턴 시각화)
    # x축 데이터: 0부터 len(series)-1까지의 정수 배열
    x_data = np.arange(len(series))
    # [수정] 위상차 phase_rad를 코사인 함수의 인수에 더해줍니다.
    cosine_wave = np.cos((2 * np.pi * x_data / m) + phase_rad) * cycle.std() * 0.5
    # 코사인파를 HP Filter 추세선에 더하여 시각화
    m_cycle_line = trend_hp + cosine_wave

    # 메인 그래프 내용
    plt.plot(series, label='Original Price Index', color='black', alpha=0.6)
    plt.plot(trend, label='STL Trend (Long-term)', color='blue', linewidth=2, alpha=0.8)
    plt.plot(cycle + trend, label='HP Filter (Smoothed)', color='orange', linestyle='--', alpha=0.8)
    plt.plot(series.index, m_cycle_line, label=f'Ideal Cycle (T={m} months)', color='red', linestyle=':', linewidth=1.5, alpha=0.9) # m 주기 시각화

    # 이벤트 마커
    plt.scatter(discord_date, series.loc[discord_date], color='red', label=f'Discord (Stumpy m={m})', s=150, zorder=5, marker='*')
    plt.scatter(change_dates, series.loc[change_dates], color='purple', label=f'Change Points (k={optimal_n_bkps})', marker='v', s=100, zorder=5)
    plt.scatter(outlier_dates_if, series.loc[outlier_dates_if], color='green', label='IsoForest (Top 5%)', marker='o', s=40, alpha=0.5)
    plt.scatter(outlier_dates_lof, series.loc[outlier_dates_lof], color='cyan', label='LOF (Local)', marker='x', s=40, alpha=0.8)

    # Major tick: 2년 간격
    ax.xaxis.set_major_locator(mdates.YearLocator(base=2))
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))

    # Minor grid: 1년 간격
    ax.xaxis.set_minor_locator(mdates.YearLocator())

    # Grid 설정
    ax.grid(True, which='major', linestyle='--', alpha=0.7)
    ax.grid(True, which='minor', linestyle=':', alpha=0.3)

    plt.title(f"[{gu}] 부동산 실질가격 종합 분석", fontsize=16)
    plt.xlabel("Year")
    plt.ylabel("Real Price Index")
    plt.legend(loc='upper left', bbox_to_anchor=(1, 1))
    plt.grid(True, which='both', linestyle='--', alpha=0.5)
    plt.tight_layout() # 간격 조정
    fig.savefig(f"figure/{gu}_main_analysis.png", dpi=300)

    plt.show()

    # -----------------------------
    # Figure 2: Elbow Method (Ruptures)
    # -----------------------------
    fig2, ax2 = plt.subplots(figsize=(10, 5))

    ax2.plot(n_bkps_candidates, costs, marker='o', linestyle='-', color='gray')
    ax2.axvline(optimal_n_bkps, color='red', linestyle='--',
                label=f'Optimal k={optimal_n_bkps}')

    ax2.set_title("Elbow Method: Optimal Change Point Count")
    ax2.set_xlabel("Number of Breakpoints (k)")
    ax2.set_ylabel("Cost (Residual Error)")
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    fig2.savefig(f"figure/{gu}_ruptures_elbow.png", dpi=300)

    plt.show()

    # -----------------------------
    # Figure 3: FFT 분석 기반 보조 그래프
    # -----------------------------
    fig3, ax3 = plt.subplots(figsize=(10, 5))

    # 주파수(0 제외)
    freqs_pos = freqs[1:N//2]     # 양수 쪽 스펙트럼
    power = Pxx                   # 동일 길이

    # 주기(개월 단위)
    periods = 1 / freqs_pos

    ax3.plot(periods, power, alpha=0.8)

    # m 주기 강조
    ax3.axvline(m, color='red', linestyle='--',
                label=f'Detected Cycle = {m} months')

    ax3.set_title(f"HP Cycle FFT Analysis (Detected m = {m} months)")
    ax3.set_xlabel("Period (Months)")
    ax3.set_ylabel("Power Spectrum Density")
    ax3.set_xscale('log')      # 긴 주기 보기 좋음
    ax3.legend()
    ax3.grid(True, which='both', linestyle='--', alpha=0.5)

    plt.tight_layout()
    fig3.savefig(f"figure/{gu}_fft_periodogram.png", dpi=300)

    plt.show()
