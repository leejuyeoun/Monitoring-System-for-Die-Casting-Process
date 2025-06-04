import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import IsolationForest
import shap
import matplotlib.pyplot as plt
from pathlib import Path
plt.rcParams['font.family'] ='Malgun Gothic' # 그래프 한글 설정


df_train_origin = pd.read_csv("../dashboard/data/df_final.csv")
df_test_origin = pd.read_csv('../dashboard/data/streaming_df.csv')

# 2. 필요 없는 열 제거
drop_cols = ['passorfail', 'registration_time', 'mold_code']  #mold_code 어떻게 할지 생각해야함!!!!!!!!
df_train = df_train_origin.drop(columns=[col for col in drop_cols if col in df_train_origin.columns])
df_test = df_test_origin.drop(columns=[col for col in drop_cols if col in df_test_origin.columns])

# 3. 수치형 변수만 사용
num_cols = df_train.select_dtypes(include=['int64', 'float64']).columns.tolist()
X_train = df_train[num_cols].to_numpy()
X_test = df_test[num_cols].to_numpy()

# 4. 수치형 변수 스케일링
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# 5. Isolation Forest 학습
IF = IsolationForest(contamination=0.05, random_state=42)
IF.fit(X_train)

df_train['is_anomaly'] = IF.predict(X_train)
print(df_train['is_anomaly'].value_counts(normalize=True))    # train데이터 정상/이상 비율


# 6. Ju Streaming Df에 이상치 판별 결과 컬럼 생성 (-1: 이상, 1: 정상)
df_test['is_anomaly'] = IF.predict(X_test)

# 7. 이상치 개수 등 확인
print(df_test['is_anomaly'].value_counts(normalize=True))     # test데이터 정상/이상 비율율
# df_test.to_excel('./df_test_-1.xlsx')

# 8. SHAP 분석 (첫 번째 이상치 행)
first_anom_idx = df_test.index[df_test['is_anomaly'] == -1][0]
X_row = X_test[first_anom_idx].reshape(1, -1)

explainer = shap.TreeExplainer(IF)
shap_values = explainer.shap_values(X_row)[0]

feature_names = num_cols

top5_idx = np.argsort(np.abs(shap_values))[::-1][:5]
top5_names = [feature_names[i] for i in top5_idx]
top5_vals = shap_values[top5_idx]

plt.figure(figsize=(8, 4))
plt.barh(top5_names[::-1], np.abs(top5_vals)[::-1], color='tomato')
plt.xlabel("SHAP Value (절댓값, 영향력)")
plt.title(f"첫 번째 이상치 행(index={first_anom_idx}) 변수 영향력 Top 5 (수치형만)")
plt.tight_layout()
plt.show()

print(f"이상치 행 인덱스: {first_anom_idx}")
print("Top5 변수와 SHAP 값:", dict(zip(top5_names, top5_vals)))


################################
# 그래프 10개 확인!!
# SHAP 분석 (이상치 상위 10개 행 각각 그래프)
################################
anomaly_indices = df_test.index[df_test['is_anomaly'] == -1][:10]  # 이상치 10개 인덱스

explainer = shap.TreeExplainer(IF)

for count, idx in enumerate(anomaly_indices, 1):
    X_row = X_test[idx].reshape(1, -1)
    shap_values = explainer.shap_values(X_row)[0]
    
    top5_idx = np.argsort(np.abs(shap_values))[::-1][:5]
    top5_names = [feature_names[i] for i in top5_idx]
    top5_vals = shap_values[top5_idx]
    
    plt.figure(figsize=(8, 4))
    plt.barh(top5_names[::-1], np.abs(top5_vals)[::-1], color='tomato')
    plt.xlabel("SHAP Value (절댓값, 영향력)")
    plt.title(f"이상치 {count}: 행 index={idx} 변수 영향력 Top 5")
    plt.tight_layout()
    plt.show()
    
    print(f"[{count}] 이상치 행 인덱스: {idx}")
    print("Top5 변수와 SHAP 값:", dict(zip(top5_names, top5_vals)))
    
    anomaly_preds = IF.predict(X_test)
    df_test_origin['is_anomaly'] = anomaly_preds
    
    df_test_origin.to_csv('../dashboard/data/streaming_df.csv', index=False)