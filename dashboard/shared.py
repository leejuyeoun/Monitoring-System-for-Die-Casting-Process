# ================================
# shared.py
# ✅ 실시간 스트리밍 기반 공정 모니터링 대시보드용 공통 모듈
# - 데이터 로딩
# - 센서 이름 정의
# - 실시간 시뮬레이션 클래스 정의
# ================================

import pandas as pd
from pathlib import Path
import joblib

# ================================
# 📁 데이터 로딩
# ================================
app_dir = Path(__file__).parent

# ✅ 정적 데이터 (누적 데이터 분석용)
try:
    static_df = pd.read_csv(app_dir / "./data/df_final.csv", encoding="utf-8")
except UnicodeDecodeError:
    static_df = pd.read_csv(app_dir / "./data/df_final.csv",  encoding="ISO-8859-1")

# ✅ 스트리밍 데이터 (실시간 시각화용)
try:
    streaming_df = pd.read_csv(app_dir / "./data/streaming_df.csv",  encoding="utf-8")
except UnicodeDecodeError:
    streaming_df = pd.read_csv(app_dir / "./data/streaming_df.csv",  encoding="cp949")


# ✅ 이상치 판단 기준 로드 (범위 기반)
try:
    spec_df_all = pd.read_csv(app_dir / "./data/iqr_bounds_by_mold_code.csv", encoding="utf-8")
except UnicodeDecodeError:
    spec_df_all = pd.read_csv(app_dir / "./data/iqr_bounds_by_mold_code.csv", encoding="cp949")

# ✅ 컬럼 정리
spec_df_all.columns = ["mold_code", "variable", "lower", "upper"]


# ✅ 센서 데이터의 사람이 읽기 쉬운 한글 이름과 단위 정의
# UI 카드나 그래프 라벨링 시 활용
sensor_labels = {
    "molten_temp": ("용탕온도", "°C"),
    "cast_pressure": ("주조압력", "bar"),
    "high_section_speed": ("고속구간속도", "mm/s")
    # 필요 시 더 추가
}
# 사용할 센서 컬럼 선택
selected_cols = [
    'mold_code',
    'registration_time',
    'molten_temp',           # 용탕 온도
    'cast_pressure',         # 주조 압력
    'high_section_speed',    # 고속 구간 속도
    'low_section_speed',     # 저속 구간 속도
    'biscuit_thickness',      # 비스킷 두께
    'passorfail',
    'is_anomaly',
    'anomaly_level',
    'top1',
    'top2',
    'top3',
    'physical_strength',
    'heating_furnace',
    'tryshot_signal',
    'lower_mold_temp2',
    'facility_operation_cycleTime',
    'upper_mold_temp2',
    'production_cycletime',
    'anomaly_score',
    'count',
    'Coolant_temperature',
    'sleeve_temperature',
    'molten_volume',
    'upper_mold_temp1',
    'EMS_operation_time',
    'lower_mold_temp1', 
    'working'

]
df_selected = streaming_df[selected_cols].reset_index(drop=True)

# ================================
# 🔧 실시간 스트리밍 클래스 정의
# ================================
class RealTimeStreamer:
    def __init__(self):
        self.full_data = df_selected.copy()
        self.current_index = 0

    def get_next_batch(self, batch_size=1):
        if self.current_index >= len(self.full_data):
            return None
        end_index = min(self.current_index + batch_size, len(self.full_data))
        batch = self.full_data.iloc[self.current_index:end_index].copy()
        self.current_index = end_index
        return batch

    def get_current_data(self):
        if self.current_index == 0:
            return pd.DataFrame()
        return self.full_data.iloc[:self.current_index].copy()

    def reset_stream(self):
        self.current_index = 0

    def get_stream_info(self):
        return {
            'total_rows': len(self.full_data),
            'current_index': self.current_index,
            'progress': (self.current_index / len(self.full_data)) * 100 if len(self.full_data) > 0 else 0
        }


class StreamAccumulator:
    def __init__(self, base_df: pd.DataFrame):
        # 누적에 사용할 기준 컬럼 (최초 static_df 기반)
        self.columns = list(base_df.columns)
        self.total_df = base_df.copy()

    def accumulate(self, new_data: pd.DataFrame):
        if not new_data.empty:
            try:
                available_cols = [col for col in self.columns if col in new_data.columns]
                new_data = new_data[available_cols].copy()
                self.total_df = pd.concat([self.total_df, new_data], ignore_index=True)
            except Exception as e:
                print(f"[⛔ accumulate 중 오류] {e}")

    def get_data(self):
        return self.total_df.copy()

    def reset(self):
        # 누적 데이터프레임을 초기 상태(static_df 기반)로 리셋
        self.total_df = static_df[self._common_columns()].copy()

    def _common_columns(self):
        # static_df와 streaming_df 간의 공통 컬럼을 리스트로 반환
        return sorted(set(static_df.columns).intersection(set(streaming_df.columns)))

import requests

def get_weather(lat=32.7767, lon=-96.7970):
    try:
        url = "https://api.open-meteo.com/v1/forecast"
        params = {
            "latitude": lat,
            "longitude": lon,
            "current_weather": True,
            "timezone": "America/Chicago"
        }
        response = requests.get(url, params=params, timeout=5)

        if response.status_code != 200:
            return f"🔌 오류 코드 [{response.status_code}] · 날씨 정보를 불러올 수 없습니다."

        data = response.json()
        weather = data["current_weather"]
        temp = round(weather["temperature"])
        windspeed = weather["windspeed"]
        time = weather["time"]

        # Open-Meteo는 날씨 설명 대신 weathercode 사용
        # → 아래는 간단한 날씨 코드 → 설명 및 이모지 매핑
        code_map = {
            0: ("☀️", "맑음"),
            1: ("🌤️", "부분 맑음"),
            2: ("⛅", "구름 많음"),
            3: ("☁️", "흐림"),
            45: ("🌫️", "박무"),
            48: ("🌫️", "박무"),
            51: ("🌦️", "가벼운 이슬비"),
            61: ("🌧️", "비"),
            71: ("❄️", "눈"),
            95: ("⛈️", "뇌우"),
        }
        code = weather["weathercode"]
        emoji, desc = code_map.get(code, ("🌡️", "정보 없음"))

        return f"텍사스 댈러스 | {emoji} {desc} | 외부온도 : {temp}℃  |  풍속 {windspeed}km/h"

    except Exception as e:
        return f"❌ 예외 발생: {str(e)}"