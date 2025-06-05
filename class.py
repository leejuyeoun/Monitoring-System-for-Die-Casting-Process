# p_chart_analysis.py
# 슬통전자 불량률에 대한 p 관리도 시각화 코드

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# ================================
# 1. 데이터 준비
# ================================
n = np.array([61, 85, 75, 86, 64, 96, 87, 93, 67, 97, 77, 88, 90, 84, 65, 71, 69, 66, 98, 72])
x_i = np.array([4, 3, 2, 4, 2, 4, 5, 3, 6, 7, 6, 5, 8, 5, 5, 3, 3, 4, 5, 8])

day_i = np.arange(1, 21)  # 날짜

# ================================
# 2. 전체 불량률 계산 (관리 중심선)
# ================================
p_hat = np.sum(x_i) / np.sum(n)
print(f"관리 중심선 (p̄): {p_hat:.4f}")

# ================================
# 3. 관리 상한선(UCL)과 하한선(LCL) 계산
# ================================
ucl = p_hat + 3 * np.sqrt(p_hat * (1 - p_hat) / n)
lcl = p_hat - 3 * np.sqrt(p_hat * (1 - p_hat) / n)
lcl = np.clip(lcl, 0, 1)  # 음수 방지

# ================================
# 4. 불량률 계산 및 데이터프레임 구성
# ================================
p_i = x_i / n

df = pd.DataFrame({
    "Day": day_i,
    "Defective Rate": p_i,
    "UCL": ucl,
    "LCL": lcl,
    "Ave. Rate": [p_hat] * 20
})

# ================================
# 5. 시각화 (p 관리도 그리기)
# ================================
sns.set(style="whitegrid")
plt.figure(figsize=(12, 6))

sns.lineplot(x="Day", y="Defective Rate", data=df, marker="o", label="Defective Rate")
sns.lineplot(x="Day", y="UCL", data=df, color='red', label="UCL")
sns.lineplot(x="Day", y="LCL", data=df, color='red', label="LCL")
sns.lineplot(x="Day", y="Ave. Rate", data=df, color='black', linestyle='--', label="Center Line")

plt.fill_between(df["Day"], df["LCL"], df["UCL"], color='red', alpha=0.1)
plt.title('P 관리도 (불량률 관리도)')
plt.xlabel('날짜 (일차)')
plt.ylabel('불량률')
plt.ylim(0, max(df["UCL"].max(), df["Defective Rate"].max()) + 0.02)
plt.legend(loc="lower right")
plt.tight_layout()
plt.show()
