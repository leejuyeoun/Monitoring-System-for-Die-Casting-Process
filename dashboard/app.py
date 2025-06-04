# ================================
# 📦 1. Import
# ================================
from io import BytesIO
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from shiny import App, ui, render, reactive
import pandas as pd
import matplotlib.pyplot as plt
from shared import RealTimeStreamer, StreamAccumulator
from shared import sensor_labels, static_df, streaming_df, spec_df_all
import numpy as np
from datetime import datetime, timedelta
import matplotlib as mpl
import joblib
import warnings
from plotly.graph_objs import Figure, Scatter
import plotly.graph_objs as go
from shinywidgets import render_widget
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import os

warnings.filterwarnings('ignore')

plt.rcParams['font.family'] = 'Malgun Gothic'  # 윈도우
mpl.rcParams['axes.unicode_minus'] = False  # 마이너스 부호 깨짐 방지

selected_cols = [
    'molten_temp',           # 용탕 온도
    'cast_pressure',         # 주조 압력
    'high_section_speed',    # 고속 구간 속도
    'low_section_speed',     # 저속 구간 속도
    'biscuit_thickness'      # 비스킷 두께
]
df_selected = streaming_df[selected_cols].reset_index(drop=True)
STATIC_DIR = os.path.join(os.path.dirname(__file__), "www")

# ================================
# 🖼️ 2. UI 정의
# ================================

app_ui = ui.page_fluid(
            ui.output_ui("dynamic_ui")  # 전체 UI는 서버에서 조건에 따라 출력
        )

# ================================
# ⚙️ 3. 서버 로직
# ================================
def server(input, output, session):
    # 초기 상태
    streamer = reactive.Value(RealTimeStreamer())
    accumulator = reactive.value(StreamAccumulator(static_df))
    current_data = reactive.Value(pd.DataFrame())
    is_streaming = reactive.Value(False)


    prediction_table_logs = reactive.Value([])  # TAB 3. [B] 로그 테이블용
    latest_logged_time = reactive.Value(None)
    # 로그인 상태 저장
    login_status = reactive.Value(False)
    # ================================
    # 스트리밍 제어
    # ================================
    @reactive.effect
    @reactive.event(input.start)
    def on_start():
        is_streaming.set(True)

    @reactive.effect
    @reactive.event(input.pause)
    def on_pause():
        is_streaming.set(False)

    @reactive.effect
    @reactive.event(input.reset)
    def on_reset():
        streamer.get().reset_stream()
        current_data.set(pd.DataFrame())
        is_streaming.set(False)

    @reactive.effect
    def stream_data():
        try:
            if not is_streaming.get():
                return
            reactive.invalidate_later(1
                                      )
            s = streamer.get()
            next_batch = s.get_next_batch(1)
            if next_batch is not None:
                current_data.set(s.get_current_data())

                # ✅ 누적기록 클래스도 업데이트 (전체 컬럼)
                accum = accumulator.get()
                accum.accumulate(next_batch)  # 내부 상태 갱신
            else:
                is_streaming.set(False)
        except Exception as e:
            print("⛔ 오류 발생:", e)
            is_streaming.set(False)


    # ================================
    # TAB 1: 실시간 공정 모니터링	Process Monitoring
    # ================================

    # ▶ 데이터 스트리밍 진행률을 퍼센트로 표시합니다.
    @output
    @render.ui
    def stream_status():
        try:
            status = "🟢 스트리밍 중" if is_streaming.get() else "🔴 정지됨"
            return status
        except Exception as e:
            return f"에러: {str(e)}"
        
        
    # ================================
    # TAP 1 [A] - 스트리밍 표시
    # ================================
    @output
    @render.plot
    def stream_plot():
        try:
            df = current_data.get().tail(20)

            if df.empty:
                fig, ax = plt.subplots()
                ax.text(0.5, 0.5, "스트리밍을 시작하세요", ha='center', va='center', fontsize=14)
                ax.set_xticks([])
                ax.set_yticks([])
                return fig

            if "registration_time" not in df.columns:
                raise ValueError("'registration_time' 컬럼이 없습니다.")
            df["registration_time"] = pd.to_datetime(df["registration_time"])

            cols_to_plot = [col for col in selected_cols if col in df.columns][:4]
            if not cols_to_plot:
                raise ValueError("선택된 컬럼이 없습니다.")

            # ✅ 컬러 팔레트 (colorblind friendly)
            colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']

            fig, axs = plt.subplots(nrows=len(cols_to_plot), ncols=1,
                                    figsize=(10, 3.5 * len(cols_to_plot)), sharex=True)

            if len(cols_to_plot) == 1:
                axs = [axs]

            for i, col in enumerate(cols_to_plot):
                ax = axs[i]
                ax.plot(df["registration_time"], df[col],
                        label=col,
                        color=colors[i % len(colors)],
                        linewidth=2,
                        marker='o', markersize=5)

                
                #ax.set_ylabel(col, fontsize=11)
            # X축 라벨 및 시간 포맷 설정
            axs[-1].set_xlabel("월-일 시:분", fontsize=11)
            axs[-1].xaxis.set_major_formatter(mdates.DateFormatter('%m-%d %H:%M'))  # ← 요거 수정
            fig.autofmt_xdate()

            # fig.suptitle("실시간 센서 스트리밍", fontsize=16, fontweight='bold')
            fig.tight_layout(rect=[0, 0.03, 1, 0.95])  # suptitle 공간 확보

            return fig

        except Exception as e:
            print("⛔ stream_plot 오류:", e)
            fig, ax = plt.subplots()
            ax.text(0.5, 0.5, f"❌ 에러 발생:\n{str(e)}", ha='center', va='center',
                    fontsize=12, color='red')
            ax.set_xticks([])
            ax.set_yticks([])
            return fig

    # ================================
    # TAP 1 [B] - 실시간 값 
    # ================================
    @output
    @render.ui
    def real_time_values():
        try:
            df = current_data.get()
            if df.empty:
                return ui.div("데이터 없음", class_="text-muted")

            latest = df.iloc[-1] if len(df) > 0 else None
            prev = df.iloc[-2] if len(df) > 1 else latest

            # ✅ 그래프 색상과 매칭
            sensor_colors = {
                'molten_temp': '#1f77b4',
                'cast_pressure': '#ff7f0e',
                'upper_mold_temp1': '#2ca02c',
                'lower_mold_temp1': '#d62728',
                # 추가 센서 색상도 여기에
            }
            sensor_korean_labels = {
            'molten_temp': '용탕 온도 (℃)',
            'cast_pressure': '주조 압력 (bar)',
            'upper_mold_temp1': '상부 금형 온도1 (℃)',
            'lower_mold_temp1': '하부 금형 온도1 (℃)',
            'high_section_speed': '고속 구간 속도 (mm/s)',
            'low_section_speed': '저속 구간 속도 (mm/s)',
            'biscuit_thickness': '비스킷 두께 (mm)',
            # 필요시 계속 추가 가능
            }

            cards = []

            # ✅ [추가] mold_code 카드 삽입
            if 'mold_code' in df.columns:
                mold_code_val = latest['mold_code']
                cards.append(
                    ui.div(
                        ui.h6("Mold Code"),
                        ui.h4(str(mold_code_val), class_="fw-bold"),
                        class_="card p-3 mb-2 border border-info"
                    )
                )
            
            for col in sensor_labels:
                if col in df.columns:
                    current_val = latest[col]
                    prev_val = prev[col] if prev is not None else current_val
                    diff = current_val - prev_val
                    percent_change = (diff / prev_val * 100) if prev_val != 0 else 0

                    # 증감 화살표 및 색상
                    if diff > 0:
                        arrow = "⬆️"
                        color_class = "text-muted"
                    elif diff < 0:
                        arrow = "⬇️"
                        color_class = "text-muted"
                    else:
                        arrow = "➡️"
                        color_class = "text-muted"

                    # ================================
                    # 경고 테두리 적용 (스펙 범위 벗어났을 때)
                    # ================================
                    warning_class = ""
                    try:
                        mold_code_val = int(latest['mold_code'])  # mold_code가 문자열일 경우 int 변환 시도
                        spec_row = spec_df_all[
                            (spec_df_all["mold_code"] == mold_code_val) &
                            (spec_df_all["variable"] == col)
                        ]
                        if not spec_row.empty:
                            lower_bound = spec_row["lower"].values[0]
                            upper_bound = spec_row["upper"].values[0]

                            if current_val < lower_bound or current_val > upper_bound:
                                warning_class = "border border-danger"
                    except Exception as e:
                        print(f"[경고 테두리 판단 오류] {col}: {e}")
                        # 오류 발생 시 경고 미적용하고 통과

                    # 색상 적용
                    custom_color = sensor_colors.get(col, "#000000")

                    cards.append(
                        ui.div(
                            ui.h6(col.replace('_', ' ').title()),
                            ui.h4(
                                f"{current_val:.1f} {arrow} ({diff:+.1f}, {percent_change:+.1f}%)",
                                class_=color_class,
                                style=f"color: {custom_color}; font-weight: bold;"
                            ),
                            class_=f"card p-3 mb-2 {warning_class}"
                    
                        )
                    )

            return ui.div(*cards, class_="d-flex flex-column gap-2")

        except Exception as e:
            return ui.div(f"오류: {str(e)}", class_="text-danger")

    # ================================
    # TAP 1 [C] - 실시간 로그
    # ================================
    @output
    @render.ui
    def recent_data_table():
        try:
            df = current_data.get()
            if df.empty:
                return ui.HTML("<p class='text-muted'>데이터 없음</p>")

            df = df.tail(7).round(2).copy()
            rows = []

            # 헤더 행
            header_cells = [ui.tags.th(col) for col in df.columns]
            rows.append(ui.tags.tr(*header_cells))

            # 데이터 행
            for i, row in df.iterrows():
                is_latest = i == df.index[-1]
                style = "background-color: #fff7d1;" if is_latest else ""
                cells = [ui.tags.td(str(val)) for val in row]
                rows.append(ui.tags.tr(*cells, style=style))

            return ui.tags.table(
                {"class": "table table-sm table-striped table-bordered", "style": "font-size: 13px;"},
                *rows
            )

        except Exception as e:
            return ui.HTML(f"<p class='text-danger'>에러 발생: {str(e)}</p>")
    

    # ================================
    # TAP 1 [C] - 실시간 선택 다운로드 
    # ================================
    @output
    @render.ui
    def download_controls():
        return ui.div(
            ui.input_select("file_format", "다운로드 형식", {
                "csv": "CSV",
                "xlsx": "Excel",
                "pdf": "PDF"
            }, selected="csv"),
            ui.download_button("download_recent_data", "📥 최근 로그 다운로드")
        )
    # ================================
    # TAP 1 [C] - 실시간 선택 다운로드 로직  
    # ================================
    @output
    @render.download(filename=lambda: f"recent_log.{input.file_format()}")
    def download_recent_data():
        def writer():
            df = current_data.get().tail(1000).round(2)
            file_format = input.file_format()

            if df.empty:
                return

            if file_format == "csv":
                yield df.to_csv(index=False).encode("utf-8")

            elif file_format == "xlsx":
                buffer = BytesIO()
                with pd.ExcelWriter(buffer, engine="xlsxwriter") as writer:
                    df.to_excel(writer, sheet_name="RecentLog", index=False)
                yield buffer.getvalue()

            elif file_format == "pdf":
                buffer = BytesIO()
                with PdfPages(buffer) as pdf:
                    fig, ax = plt.subplots(figsize=(8.5, 4))
                    ax.axis("off")
                    table = ax.table(cellText=df.values, colLabels=df.columns, loc="center")
                    table.auto_set_font_size(False)
                    table.set_fontsize(10)
                    table.scale(1.2, 1.2)
                    pdf.savefig(fig, bbox_inches='tight')
                    plt.close(fig)
                yield buffer.getvalue()
        return writer()
    # ================================
    # TAP 1 [D] - 이상 불량 알림 
    # ================================
    @output
    @render.ui
    def anomaly_alerts():
        try:
            df = current_data.get()
            if df.empty:
                return ui.div("데이터 없음", class_="text-muted")

            # 최신 데이터로 이상/불량 상태 확인
            latest = df.iloc[-1] if len(df) > 0 else None
            
            # 이상 탐지 카드
            anomaly_status = "이상" if hasattr(latest, 'is_anomaly') and latest.get('is_anomaly', 0) == 1 else "정상"
            anomaly_score = latest.get('anomaly_score', 0) if latest is not None else 0
            anomaly_icon = "❌" if anomaly_status == "이상" else "✅"
            anomaly_class = "anomaly-card alert alert-danger" if anomaly_status == "이상" else "normal-card alert alert-success"
            reg_time = latest.get('registration_time')
            return ui.div(
                # 이상 탐지 카드
                ui.div(
                    ui.h6(f"{anomaly_icon} 이상 탐지"),
                    ui.p(f"상태: {anomaly_status}"),
                    ui.p(f"점수: {anomaly_score:.3f}"),
                    ui.p(f"시각: {reg_time}"),
                    ui.input_action_button("goto_2page", "이상탐지 확인하기", class_="btn btn-sm btn-outline-primary"),
                    class_=anomaly_class
                )
            )
            
        except Exception as e:
            return ui.div(f"오류: {str(e)}", class_="text-danger")
        
        
    @output
    @render.ui
    def current_prediction2():
        try:
            df = current_data.get()
            if df.empty:
                return ui.div("데이터 없음", class_="text-muted")

            # 최신 데이터 한 행
            latest = df.iloc[-1]

            if 'passorfail' not in latest:
                print("⚠️ 'passorfail' 컬럼이 존재하지 않음")
                return ui.div("예측값 없음", class_="text-muted")

            # 결합 확률은 이미 'passorfail' 컬럼에 예측값이 0~1로 들어온다고 가정
            prob = latest['passorfail']
            result = "불량" if prob >= 0.5 else "양품"
            icon = "❌" if result == "불량" else "✅"
            color_class = "alert alert-danger" if result == "불량" else "alert alert-success"

            reg_time = latest.get('registration_time')
            try:
                reg_time = pd.to_datetime(reg_time).strftime("%Y-%m-%d %H:%M:%S")
            except Exception as time_err:
                print(f"⚠️ 시간 파싱 오류: {time_err}")
                reg_time = "시간 정보 없음"

            return ui.div(
                ui.div(
                    ui.h6("🧾 판정 결과"),
                    ui.h4(f"{icon} {result}", class_="fw-bold"),
                    class_="mb-2"
                ),
                ui.div(
                    ui.h6("🕒 판정 시간"),
                    ui.p(reg_time),
                    ui.input_action_button("goto_3page", "불량탐지 확인하기", class_="btn btn-sm btn-outline-primary")
                ),
                class_=f"{color_class} p-3 rounded"
            )

        except Exception as e:
            print(f"⛔ current_prediction 오류 발생: {e}")
            return ui.div(f"오류: {str(e)}", class_="text-danger")
    @reactive.effect
    @reactive.event(input.goto_2page)
    def go_to_page_3():
        ui.update_navs("main_nav", "공정 이상 탐지	(Process Anomaly Detection)") 
    
    @reactive.effect
    @reactive.event(input.goto_3page)
    def go_to_page_3():
        ui.update_navs("main_nav", "품질 이상 판별	(Quality Defect Classification)") 

    # ================================
    # TAB 2: [A] 이상 예측
    # ================================
    @output
    @render.plot
    def anomaly_variable_count():
        try:
            df = accumulator.get().get_data()
            if df.empty:
                fig, ax = plt.subplots()
                ax.text(0.5, 0.5, "데이터 없음", ha='center', va='center')
                return fig

            # 이상 데이터만 필터링
            if 'is_anomaly' in df.columns:
                anomaly_df = df[df['is_anomaly'] == 1]
            else:
                # 임시로 상위 20% 데이터를 이상으로 간주
                threshold = df['anomaly_score'].quantile(0.8) if 'anomaly_score' in df.columns else 0.8
                anomaly_df = df[df.get('anomaly_score', 0) > threshold]

            if anomaly_df.empty:
                fig, ax = plt.subplots()
                ax.text(0.5, 0.5, "이상 데이터 없음", ha='center', va='center')
                return fig

            # SHAP 기준 변수별 영향도 집계 (시뮬레이션)
            variables = sensor_labels
            counts = {}
            
            for var in variables:
                # 각 이상 샘플에서 해당 변수가 가장 큰 영향을 준 횟수 계산
                # 실제로는 SHAP 값을 사용하지만, 여기서는 시뮬레이션
                counts[var] = np.random.randint(1, len(anomaly_df)//2)

            fig, ax = plt.subplots(figsize=(10, 6))
            bars = ax.bar(counts.keys(), counts.values(), color=['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4', '#FFEAA7'])
            ax.set_title("주요 변수의 이상 발생 횟수 (SHAP 기반)")
            ax.set_xlabel("변수명")
            ax.set_ylabel("이상 발생 횟수")
            
            # 막대 위에 값 표시
            for bar in bars:
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height + 0.1,
                       f'{int(height)}', ha='center', va='bottom')
            
            plt.xticks(rotation=45)
            plt.tight_layout()
            return fig
            
        except Exception as e:
            fig, ax = plt.subplots()
            ax.text(0.5, 0.5, f"오류: {str(e)}", ha='center', va='center')
            return fig

    @output
    @render.ui
    def anomaly_notifications():
        try:
            df = accumulator.get().get_data()
            if df.empty:
                return ui.div("데이터 없음", class_="text-muted")

            # 최근 이상 발생 건 조회
            if 'is_anomaly' in df.columns:
                recent_anomalies = df[df['is_anomaly'] == 1].tail(5)
            else:
                threshold = df['anomaly_score'].quantile(0.8) if 'anomaly_score' in df.columns else 0.8
                recent_anomalies = df[df.get('anomaly_score', 0) > threshold].tail(5)

            if recent_anomalies.empty:
                return ui.div("최근 이상 없음", class_="text-success")

            notifications = []
            risk_counts = {"위험": 0, "경고": 0, "주의": 0}
            
            for idx, row in recent_anomalies.iterrows():
                score = row.get('anomaly_score', 0)
                
                # 위험도 분류
                if score > 0.9:
                    risk_level = "위험"
                    icon = "🔴"
                    risk_counts["위험"] += 1
                elif score > 0.7:
                    risk_level = "경고"
                    icon = "🟡"
                    risk_counts["경고"] += 1
                else:
                    risk_level = "주의"
                    icon = "🟠"
                    risk_counts["주의"] += 1
                
                # 주요 원인 (시뮬레이션)
                main_cause = np.random.choice(sensor_labels)
                time_str = datetime.now().strftime('%H:%M:%S')
                
                notifications.append(
                    ui.div(
                        ui.p(f"{icon} [{risk_level}] {time_str}"),
                        ui.p(f"주요 원인: {main_cause}"),
                        ui.p(f"이상 점수: {score:.3f}"),
                        class_="border p-2 mb-2 rounded"
                    )
                )

            # 위험도별 누적 건수
            summary = ui.div(
                ui.h6("위험도별 누적 건수"),
                ui.p(f"🔴 위험: {risk_counts['위험']}건"),
                ui.p(f"🟡 경고: {risk_counts['경고']}건"),
                ui.p(f"🟠 주의: {risk_counts['주의']}건"),
                class_="bg-light p-2 mb-3 rounded"
            )

            return ui.div(summary, *notifications)
            
        except Exception as e:
            return ui.div(f"오류: {str(e)}", class_="text-danger")


    @output
    @render.plot
    def anomaly_time_analysis():
        try:
            df = accumulator.get().get_data()
            if df.empty or 'datetime' not in df.columns:
                fig, ax = plt.subplots()
                ax.text(0.5, 0.5, "시간 데이터 없음", ha='center', va='center')
                return fig

            time_unit = input.anomaly_time_unit()

            # datetime 컬럼 생성/변환
            if 'datetime' not in df.columns:
                if 'date' in df.columns and 'time' in df.columns:
                    df['datetime'] = pd.to_datetime(df['date'] + ' ' + df['time'], errors="coerce")
                else:
                    df['datetime'] = pd.date_range(start='2024-01-01', periods=len(df), freq='H')

            # 시간 단위별 그룹핑
            if time_unit == "1시간":
                df['time_group'] = df['datetime'].dt.floor('H')
            elif time_unit == "3시간":
                df['time_group'] = df['datetime'].dt.floor('3H')
            elif time_unit == "일":
                df['time_group'] = df['datetime'].dt.date
            elif time_unit == "주":
                df['time_group'] = df['datetime'].dt.to_period('W')
            elif time_unit == "월":
                df['time_group'] = df['datetime'].dt.to_period('M')

            # 이상 건수 집계
            if 'is_anomaly' in df.columns:
                anomaly_counts = df[df['is_anomaly'] == 1].groupby('time_group').size()
            else:
                threshold = df['anomaly_score'].quantile(0.8) if 'anomaly_score' in df.columns else 0.8
                anomaly_counts = df[df.get('anomaly_score', 0) > threshold].groupby('time_group').size()

            if anomaly_counts.empty:
                fig, ax = plt.subplots()
                ax.text(0.5, 0.5, "이상 데이터 없음", ha='center', va='center')
                return fig

            fig, ax = plt.subplots(figsize=(12, 6))
            ax.plot(range(len(anomaly_counts)), anomaly_counts.values, marker='o', linewidth=2, markersize=6)
            ax.set_title(f"시간에 따른 이상 발생량 ({time_unit} 단위)")
            ax.set_xlabel("시간")
            ax.set_ylabel("이상 건수")
            ax.grid(True, alpha=0.3)

            # x축 라벨 설정
            if len(anomaly_counts) > 10:
                step = len(anomaly_counts) // 10
                tick_positions = range(0, len(anomaly_counts), step)
                tick_labels = [str(anomaly_counts.index[i]) for i in tick_positions]
                ax.set_xticks(tick_positions)
                ax.set_xticklabels(tick_labels, rotation=45)

            plt.tight_layout()
            return fig

        except Exception as e:
            fig, ax = plt.subplots()
            ax.text(0.5, 0.5, f"에러 발생: {str(e)}", ha='center', va='center')
            return fig

    # ================================
    # TAB 3: 품질 분석
    # ================================
    @output
    @render.plot
    def defect_rate_plot():
        try:
            unit = input.grouping_unit()  # "일", "주", "월"

            #df_vis = static_df.copy()
            df_vis = accumulator.get().get_data()

            # 문자열 날짜를 datetime으로 변환
            df_vis['datetime'] = pd.to_datetime(df_vis['registration_time'], errors="coerce")

            # 그룹핑 기준 추가
            if unit == "일":
                df_vis['group'] = df_vis['datetime'].dt.strftime('%Y-%m-%d')
            elif unit == "주":
                df_vis['group'] = df_vis['datetime'].dt.to_period('W').astype(str)
            elif unit == "월":
                df_vis['group'] = df_vis['datetime'].dt.to_period('M').astype(str)

            # 각 그룹별 불량률 계산
            group_result = df_vis.groupby(['group', 'passorfail']).size().unstack(fill_value=0)
    
            selected_group = input.selected_group()
            if selected_group not in group_result.index:
                raise ValueError("선택한 그룹에 대한 데이터가 없습니다.")
            counts = group_result.loc[selected_group]
    
            # 시각화
            fig, ax = plt.subplots()
            labels = ['양품', '불량']
            sizes = [counts.get(0, 0), counts.get(1, 0)]
            colors = ['#4CAF50', '#F44336']
    
            wedges, _, _ = ax.pie(
                sizes, labels=labels, autopct='%1.1f%%', colors=colors, startangle=90
            )
            ax.axis('equal')
            ax.set_title(f"{selected_group} ({unit} 기준) 불량률")
            ax.legend(wedges, labels, title="예측 결과", loc="upper right", bbox_to_anchor=(1.1, 1))
    
            return fig
    
        except Exception as e:
            fig, ax = plt.subplots()
            ax.text(0.5, 0.5, f"에러: {str(e)}", ha='center', va='center')
            return fig
        
    @output
    @render.ui
    def group_choice():
        try:
            unit = input.grouping_unit()
            df_vis = accumulator.get().get_data()
            df_vis['datetime'] = pd.to_datetime(df_vis['registration_time'], errors="coerce")

            if unit == "일":
                df_vis['group'] = df_vis['datetime'].dt.strftime('%Y-%m-%d')
            elif unit == "주":
                df_vis['group'] = df_vis['datetime'].dt.to_period('W').astype(str)
            elif unit == "월":
                df_vis['group'] = df_vis['datetime'].dt.to_period('M').astype(str)

            unique_groups = sorted(df_vis['group'].dropna().unique())
            return ui.input_select("selected_group", "📆 조회할 기간 선택", choices=unique_groups, selected=unique_groups[-1] if unique_groups else None)
        except:
            return ui.input_select("selected_group", "📆 조회할 기간 선택", choices=["선택 불가"], selected=None)

    # ================================
    # TAP 3 [A] - 이상 불량 알림 
    # ================================
    @output
    @render.plot
    def defect_rate_plot():
        try:
            # 기간 선택
            start_date, end_date = input.date_range()

            df_vis = accumulator.get().get_data()
            df_vis = df_vis.loc[:, ~df_vis.columns.duplicated()]  # 중복 열 제거
            df_vis['datetime'] = pd.to_datetime(df_vis['registration_time'], errors="coerce")

            # 날짜 필터링
            mask = (df_vis['datetime'].dt.date >= pd.to_datetime(start_date).date()) & \
                (df_vis['datetime'].dt.date <= pd.to_datetime(end_date).date())
            df_filtered = df_vis.loc[mask]

            if df_filtered.empty:
                raise ValueError("선택한 기간 내 데이터가 없습니다.")

            # ✅ 몰드코드 + 불량 여부별 카운트
            grouped = df_filtered.groupby(['mold_code', 'passorfail']).size().unstack(fill_value=0)
            grouped.columns = ['양품', '불량'] if 0 in grouped.columns else ['불량']
            grouped = grouped.reset_index()

            # ✅ 시각화 (stacked bar chart)
            import numpy as np
            mold_codes = grouped['mold_code']
            x = np.arange(len(mold_codes))
            width = 0.6

            fig, ax = plt.subplots(figsize=(10, 5))
            ax.bar(x, grouped.get('양품', [0]*len(grouped)), width, label='양품', color='#4CAF50')
            ax.bar(x, grouped.get('불량', [0]*len(grouped)), width,
                bottom=grouped.get('양품', [0]*len(grouped)), label='불량', color='#F44336')

            ax.set_xlabel('몰드 코드')
            ax.set_ylabel('개수')
            ax.set_title(f"{start_date} ~ {end_date} 몰드코드별 누적 예측 결과")
            ax.set_xticks(x)
            ax.set_xticklabels(mold_codes, rotation=45, ha='right')
            ax.legend()

            fig.tight_layout()
            return fig

        except Exception as e:
            print(f"[defect_rate_plot] 에러: {e}")
            fig, ax = plt.subplots()
            ax.text(0.5, 0.5, f"에러 발생: {str(e)}", ha='center', va='center')
            return fig


    # ================================
    # TAP 3 [B] - 이상 불량 알림
    # ================================
    @output
    @render.ui
    def current_prediction():
        try:
            df = current_data.get()
            if df.empty:
                print("⚠️ current_data가 비어 있음")
                return ui.div("데이터 없음", class_="text-muted")

            # 최신 데이터 한 행
            latest = df.iloc[-1]

            if 'passorfail' not in latest:
                print("⚠️ 'passorfail' 컬럼이 존재하지 않음")
                return ui.div("예측값 없음", class_="text-muted")

            # 결합 확률은 이미 'passorfail' 컬럼에 예측값이 0~1로 들어온다고 가정
            prob = latest['passorfail']
            result = "불량" if prob >= 0.5 else "양품"
            icon = "❌" if result == "불량" else "✅"
            color_class = "alert alert-danger" if result == "불량" else "alert alert-success"

            reg_time = latest.get('registration_time')
            try:
                reg_time = pd.to_datetime(reg_time).strftime("%Y-%m-%d %H:%M:%S")
            except Exception as time_err:
                print(f"⚠️ 시간 파싱 오류: {time_err}")
                reg_time = "시간 정보 없음"

            return ui.div(
                ui.div(
                    ui.h6("🧾 판정 결과"),
                    ui.h4(f"{icon} {result}", class_="fw-bold"),
                    class_="mb-2"
                ),
                ui.div(
                    ui.h6("🕒 판정 시간"),
                    ui.p(reg_time)
                ),
                class_=f"{color_class} p-3 rounded"
            )

        except Exception as e:
            print(f"⛔ current_prediction 오류 발생: {e}")
            return ui.div(f"오류: {str(e)}", class_="text-danger")

    @reactive.effect
    @reactive.event(current_data)
    def log_prediction_from_current_row():
        df = current_data.get()
        if df.empty or 'passorfail' not in df.columns:
            return

        row = df.iloc[-1]
        prob = row.get('passorfail', None)

        if pd.isna(prob):
            return

        result = "불량" if prob >= 0.5 else "양품"
        reg_time = row.get('registration_time')
        try:
            reg_time = pd.to_datetime(reg_time).strftime("%Y-%m-%d %H:%M:%S")
        except:
            reg_time = str(reg_time)

        logs = prediction_table_logs.get()
        logs.append({
            "판정 시간": reg_time,
            "결과": result
        })
        prediction_table_logs.set(logs[-20:])  # 최신 20개만 유지

    @output
    @render.ui
    def prediction_log_table():
        logs = prediction_table_logs.get()
        if not logs:
            return ui.div("예측 로그 없음", class_="text-muted")

        headers = ["판정 시간", "결과"]
        table_rows = [ui.tags.tr(*[ui.tags.th(h) for h in headers])]

        for log in reversed(logs):  # 최신이 위에
            table_rows.append(
                ui.tags.tr(
                    ui.tags.td(log["판정 시간"]),
                    ui.tags.td(log["결과"]),
                )
            )

        return ui.div(
        ui.tags.table(
            {"class": "table table-sm table-bordered table-striped mb-0"},
            *table_rows
        ),
        style="max-height: 200px; overflow-y: auto;"  # 스크롤 설정
    )
# ================================
    # TAP 3 [C] - 이상 불량 알림 
# ================================ 
    @output
    @render.plot
    def fail_rate_by_time():
        try:
            df = accumulator.get().get_data()
            if df.empty or 'passorfail' not in df.columns:
                raise ValueError("데이터 없음")

            # datetime 생성
            if 'datetime' not in df.columns:
                df['datetime'] = pd.to_datetime(df['registration_time'], errors='coerce')

            # 시간 단위 선택
            unit = input.fail_time_unit()
            if unit == "1시간":
                df['time_group'] = df['datetime'].dt.floor('H')
            elif unit == "3시간":
                df['time_group'] = df['datetime'].dt.floor('3H')
            elif unit == "일":
                df['time_group'] = df['datetime'].dt.date
            elif unit == "주":
                df['time_group'] = df['datetime'].dt.to_period('W')
            elif unit == "월":
                df['time_group'] = df['datetime'].dt.to_period('M')

            # 불량률 계산
            total_counts = df.groupby('time_group').size()
            fail_counts = df[df['passorfail'] == 1].groupby('time_group').size()
            rate = (fail_counts / total_counts).fillna(0)

            # ⛔ 기존 코드에서는 전체 rate 사용
            # ✅ 수정: 가장 최근 20개만 사용
            rate = rate.sort_index().iloc[-20:]  # 최근 시간 기준 정렬 후 20개 선택

            labels = rate.index.astype(str)
            values = rate.values

            fig, ax = plt.subplots(figsize=(12, 6))
            ax.plot(labels, values, marker='o', linestyle='-')
            ax.set_title(f"시간 단위별 불량률 분석 ({unit}) - 최근 20개")
            ax.set_xlabel("시간 단위")
            ax.set_ylabel("불량률")
            ax.set_ylim(0, 1)
            ax.grid(True, alpha=0.3)
            plt.xticks(rotation=45)
            plt.tight_layout()
            return fig

        except Exception as e:
            fig, ax = plt.subplots()
            ax.text(0.5, 0.5, f"에러 발생: {str(e)}", ha='center', va='center')
            return fig


# ================================
    # TAP 0  - 로그인 기능
# ================================ 
    # 로그인 버튼 처리
    @reactive.effect
    @reactive.event(input.login_button)
    def login():
        if input.username() == "admin" and input.password() == "1234":
            login_status.set(True)
        else:
            ui.notification_show("❌ 로그인 실패", duration=3)

    # 로그아웃 버튼 처리
    @reactive.effect
    @reactive.event(input.logout_button)
    def logout():
        login_status.set(False)

    # 전체 UI 렌더링
    @output
    @render.ui
    def dynamic_ui():
        if not login_status.get():
            # 로그인 화면 반환
            return ui.card(
                ui.div(
                    ui.tags.img(
                        src="./logo2.png",
                        style="max-width: 300px; margin-bottom: 20px;"
                    ),
                    class_="text-center"
                ),
                ui.card_header("LS 기가 팩토리 로그인"),
                ui.input_text("username", "아이디"),
                ui.input_password("password", "비밀번호"),
                ui.input_action_button("login_button", "로그인"),
                ui.p("ID: admin / PW: 1234", class_="text-muted")
            )
        else:
            return ui.page_fluid(
                        ui.tags.head(
                            ui.tags.link(rel="stylesheet", href="./style.css")
                        ),
                        
                        ui.page_navbar(
                            # ================================
                            # TAB 1: 실시간 공정 모니터링	Process Monitoring
                            # ================================
                            ui.nav_panel("실시간 공정 모니터링	(Process Monitoring)",
                                ui.layout_columns(
                                    # [A] 실시간 그래프
                                    ui.card(
                                        ui.row(
                                            ui.column(12,
                                                ui.div(
                                                    ui.input_action_button("start", "▶ 시작", class_="btn-success me-2"),
                                                    ui.input_action_button("pause", "⏸ 일시정지", class_="btn-warning me-2"),
                                                    ui.input_action_button("reset", "🔄 리셋", class_="btn-secondary me-2"),
                                                    ui.output_ui("stream_status"),
                                                )
                                            )
                                        ),
                                        ui.card_header("[A] 실시간 센서 스트리밍"),
                                        ui.output_plot("stream_plot", height="400px")
                                    ),
                                    # [B] 실시간 값
                                    ui.card(
                                        ui.card_header("[B] 실시간 값"),
                                        ui.output_ui("real_time_values"),
                                    ),
                                    col_widths=[8, 4]
                                ),
                                ui.layout_columns(
                                    # [C] 실시간 로그
                                    ui.card(
                                        ui.card_header("[C] 실시간 로그"),
                                        ui.div(
                                            ui.h5("실시간 로그 (최근 7건)"),
                                            ui.output_table("recent_data_table"),
                                            ui.output_ui("download_controls")  # 형식 선택 + 다운로드 버튼
                                        )
                                    ),
                                    # [D] 이상 불량 알림 탭
                                    ui.card(
                                        ui.card_header("[D] 이상 불량 알림"),
                                        ui.output_ui("anomaly_alerts"),
                                        ui.output_ui("current_prediction2"),
                                    ),
                                    col_widths=[6, 6]
                                )    
                            ),
                            
                            # ================================
                            # TAB 2: 이상 예측
                            # ================================
                            ui.nav_panel("공정 이상 탐지	(Process Anomaly Detection)",
                                ui.layout_columns(
                                    # TAB 2 [A] 주요 변수의 이상 발생 횟수
                                    ui.card(
                                        ui.card_header("[A] 주요 변수의 이상 발생 횟수"),
                                        ui.output_plot("anomaly_variable_count", height="300px")
                                    ),
                                    # TAB 2 [B] 이상 탐지 알림
                                    ui.card(
                                        ui.card_header("[B] 이상 탐지 알림"),
                                        ui.output_ui("anomaly_notifications")
                                    ),
                                    col_widths=[6, 6]
                                ),
                                ui.layout_columns(
                                    #TAB 2 [C] 시간에 따른 이상 분석
                                    ui.card(
                                        ui.card_header("[C] 시간에 따른 이상 분석"),
                                        ui.div(
                                            ui.input_select(
                                                "anomaly_time_unit", 
                                                "시간 단위 선택", 
                                                choices=["1시간", "3시간", "일", "주", "월"], 
                                                selected="일"
                                            ),
                                            class_="mb-3"
                                        ),
                                        ui.output_plot("anomaly_time_analysis", height="300px")
                                    ),
                                    # [D] SHAP 해석, 변수 기여도 분석
                                    ui.card(
                                        ui.card_header("[D] SHAP 변수 기여도 분석"),
                                        ui.output_table("shap_analysis_table")
                                    ),
                                    col_widths=[6, 6]
                                )
                            ),
                            # ================================
                            # TAB 3: 품질
                            # ================================
                    
                                ui.nav_panel("품질 이상 판별	(Quality Defect Classification)",
                                    # TAB 3 [A] 
                                    ui.layout_columns(
                                        ui.card(
                                            ui.card_header("[A]"),
                                            ui.input_date_range(
                                                "date_range", 
                                                "📅 기간 선택", 
                                                start="2019-02-21",  # 데이터 시작일
                                                end="2019-03-12",    # 데이터 종료일 # 기본값
                                            ),
                                            ui.output_plot("defect_rate_plot", height="300px"),
                    
                                        ),
                                        # TAB 3 [B]
                                        ui.card(
                                            ui.card_header("[B]"),
                                            ui.output_ui("current_prediction"),
                                            ui.output_ui("prediction_log_table")
                                        )
                                    ),
                                    # TAB 3 [C]
                                    ui.layout_columns(
                                        ui.card(
                                            ui.card_header("[C]"),
                                            ui.input_select(
                                                "fail_time_unit", 
                                                "시간 단위 선택", 
                                                choices=["1시간", "3시간", "일", "주", "월"], 
                                                selected="일"
                                            ),
                                            ui.output_plot("fail_rate_by_time", height="350px")
                                        ),
                                        ui.card(
                                            ui.card_header("[D]"),
                                        )
                                    )
                                ),
                                ui.nav_spacer(),  # 선택
                            ui.nav_panel("🔓 로그아웃",  # ✅ 여기 추가!
                                ui.layout_column_wrap(
                                    ui.h4("로그아웃 하시겠습니까?"),
                                    ui.input_action_button("logout_button", "로그아웃", class_="btn btn-danger")
                                )
                            ),
                                id="main_nav",
                                title = "LS 기가 펙토리"
                            )
                        )
            
            
# ================================
# 🚀 4. 앱 실행
# ================================
app = App(app_ui, server, static_assets=STATIC_DIR)
