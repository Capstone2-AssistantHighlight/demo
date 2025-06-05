# graph_viewer.py
import sys
import os
import json
from datetime import timedelta
import math

from PyQt5.QtWidgets import (
    QApplication, QWidget, QVBoxLayout, QMessageBox
)
import matplotlib
matplotlib.rcParams['font.family'] = 'Malgun Gothic'
matplotlib.rcParams['axes.unicode_minus'] = False

matplotlib.use("Qt5Agg")
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
from matplotlib.patches import Patch
from moviepy.editor import VideoFileClip

# 감정별 색상 매핑
COLOR_MAP = {
    "슬픔":   "#1565C0",   # 딥 블루
    "중립":   "#616161",   # 다크 그레이
    "분노":   "#C62828",   # 딥 레드
    "행복":   "#FFD600",   # 브라이트 옐로우
    "놀람":   "#E64A19"    # 딥 오렌지
}

def timestamp_to_seconds(ts: str) -> float:
    # "HH-MM-SS" 또는 "HH-MM-SS.s" 형태를 초(float)로 변환
    parts = ts.split("-")
    if len(parts) != 3:
        return 0.0
    hh, mm, ss = parts
    try:
        seconds = int(hh) * 3600 + int(mm) * 60 + float(ss)
    except ValueError:
        # 소수점 있는 경우
        seconds = int(hh) * 3600 + int(mm) * 60 + float(ss)
    return seconds

def seconds_to_timestamp(sec: float) -> str:
    # 초(float) → "HH-MM-SS.s" 문자열로 변환
    if sec < 0:
        sec = 0.0
    hh = int(sec // 3600)
    mm = int((sec % 3600) // 60)
    ss = sec - (hh * 3600 + mm * 60)
    return f"{hh:02d}-{mm:02d}-{ss:05.2f}"  # 소수 둘째 자리까지

class GraphViewer(QWidget):
    def __init__(self, video_path, speed_json, chat_emo_json, audio_emo_json):
        super().__init__()
        self.video_path = video_path
        self.speed_json = speed_json
        self.chat_emo_json = chat_emo_json
        self.audio_emo_json = audio_emo_json

        self.load_data()
        self.initUI()

    def load_data(self):
        # 1) JSON 파일 읽기
        with open(self.speed_json, "r", encoding="utf-8") as f:
            self.speed_data = json.load(f)
        with open(self.chat_emo_json, "r", encoding="utf-8") as f:
            self.chat_emo_data = json.load(f)
        with open(self.audio_emo_json, "r", encoding="utf-8") as f:
            raw_audio_list = json.load(f)

        # 2) audio_emotion.json(리스트) → filename: emotion 맵으로 변환
        self.audio_emo_map = {}
        for entry in raw_audio_list:
            fname = entry.get("filename")
            emo   = entry.get("emotion", "중립")
            if fname is not None:
                self.audio_emo_map[fname] = emo

        # 3) 전체 rank 리스트를 모아서 “상위 30%” cutoff 구하기
        all_ranks = [e["rank"] for e in self.speed_data if "rank" in e]
        if not all_ranks:
            cutoff_rank = 0
        else:
            sorted_ranks = sorted(all_ranks)  # 예: [1, 2, 3, 4, 5, ...]
            n = len(sorted_ranks)
            # 상위 30% 개수 (소수점 올림)
            top_count = math.ceil(n * 0.3)
            # 인덱스는 0부터 시작하므로, cutoff 인덱스 = top_count - 1
            cutoff_index = max(top_count - 1, 0)
            cutoff_rank = sorted_ranks[cutoff_index]

        # 4) rank가 cutoff_rank 이하인(상위 30%) 윈도우만 self.points에 추가
        self.points = []
        for entry in self.speed_data:
            rank = entry.get("rank", None)
            # rank 정보가 없거나, rank가 cutoff_rank보다 크면(하위 70%) 건너뜀
            if rank is None or rank > cutoff_rank:
                continue

            start_ts   = entry["start_time"]   # ex. "00-37-30"
            end_ts     = entry["end_time"]     # ex. "00-37-34"
            line_count = entry["line_count"]   # 채팅 속도 (lines per 5s)

            # 해당 윈도우 중앙 시간(초) 계산
            start_sec  = timestamp_to_seconds(start_ts)
            end_sec    = timestamp_to_seconds(end_ts)
            center_sec = start_sec + (end_sec - start_sec) / 2.0

            # chat emotion 찾기 (chat_emo_data도 리스트 형태)
            chat_emo = "중립"
            for ce in self.chat_emo_data:
                if ce.get("start_time") == start_ts:
                    chat_emo = ce.get("emotion", "중립")
                    break

            # audio emotion 찾기 (audio_emo_map에서 start_ts 기준)
            audio_emo = self.audio_emo_map.get(start_ts, "중립")

            # self.points에 담기: (중앙 초, 속도, 채팅창 감정, 오디오 감정, start_ts, end_ts)
            self.points.append((center_sec, line_count, chat_emo, audio_emo, start_ts, end_ts))

        # 5) 시각화 시 “시간 순서”대로 그리기 위해 정렬해 둠
        self.points.sort(key=lambda x: x[0])


    def initUI(self):
        self.setWindowTitle("속도 + 감정 그래프 (원 안에 원 형태)")
        self.setGeometry(100, 100, 1800, 500)

        layout = QVBoxLayout()
        self.setLayout(layout)

        # Matplotlib Figure / Canvas
        self.fig = Figure(figsize=(9, 5))
        self.canvas = FigureCanvas(self.fig)
        layout.addWidget(self.canvas)

        self.ax = self.fig.add_subplot(111)
        self.ax.set_title("Chat Speed + Emotion")
        self.ax.set_xlabel("Time (seconds)")
        self.ax.set_ylabel("Chat Speed (lines per 5s)")
        # 격자는 너무 진하면 복잡해 보이므로 꺼두거나 연하게
        self.ax.grid(False)

        legend_handles = []
        for emo, color in COLOR_MAP.items():
            # Patch 객체에 해당 감정 이름과 색상을 넣어서 레전드 항목으로 사용
            legend_handles.append(Patch(facecolor=color, edgecolor="white", label=emo))
        # loc="upper left", bbox_to_anchor으로 좌측 상단에 고정
        self.ax.legend(
            handles=legend_handles,
            title="Emotion",
            loc="upper left",
            bbox_to_anchor=(0.01, 0.99),
            fontsize=9,
            title_fontsize=10,
            frameon=True,
            framealpha=0.5
        )

        # (선택) “채팅 속도 추세”는 그리지 않고, 원만 표시할 것이므로 plot 코드는 생략했습니다.
        # 만약 그리기를 원하면, 아래 줄을 참고:
        # xs_sorted = [pt[0] for pt in self.points]
        # ys_sorted = [pt[1] for pt in self.points]
        # self.ax.plot(xs_sorted, ys_sorted, color="gray", linewidth=0.5, alpha=0.3)

        # 0) 정렬된 순서대로 선 연결
        #if len(xs_sorted) > 0:
        #    self.ax.plot(xs_sorted, ys_sorted, color="gray", linewidth=1, alpha=0.4, label="Chat Speed Trend")
        #    self.ax.legend(loc="upper left", fontsize=8)

        # 2) 겹쳐진 원의 크기 설정 (마커 면적 단위: points^2)
        #    outer_size: 큰 원 (스트리머 감정) 크기
        #    inner_size: 작은 원 (채팅창 감정) 크기
        outer_size = 100   # 예: 80 포인트^2
        inner_size = 50   # 예: 30 포인트^2

        # ② 각 포인트마다 “큰 원(스트리머)” → “작은 원(채팅창)” 순으로 겹쳐서 그리기
        for cx, ly, chat_e, audio_e, st, et in self.points:
            # 큰 원: 스트리머 감정
            self.ax.scatter(
                cx, ly,
                c=COLOR_MAP.get(audio_e, "black"),
                s=outer_size,
                marker="o",
                edgecolors="white",
                linewidths=0.5,
                alpha=0.7,
                zorder=1
            )
            # 작은 원: 채팅창 감정
            self.ax.scatter(
                cx, ly,
                c=COLOR_MAP.get(chat_e, "black"),
                s=inner_size,
                marker="o",
                edgecolors="white",
                linewidths=0.5,
                alpha=0.8,
                zorder=2
            )

        # 4) 클릭 이벤트 연결 (필요 시 30초 클립 생성 등 구현)
        self.canvas.mpl_connect("button_press_event", self.on_click)

        self.canvas.draw()

    def on_click(self, event):
        if event.inaxes != self.ax:
            return

        # 클릭한 x 좌표(초)와 가장 가까운 포인트 인덱스 찾기 (단순 절댓값 비교)
        click_x = event.xdata
        distances = [abs(click_x - pt[0]) for pt in self.points]
        idx = int(min(range(len(distances)), key=lambda i: distances[i]))

        # 선택된 포인트 정보
        center_sec, line_count, chat_e, audio_e, start_ts, end_ts = self.points[idx]

        # 클립 구간 계산: 중심 ± 15초
        clip_start = max(center_sec - 15.0, 0.0)
        clip_end   = center_sec + 15.0

        # 실제 영상 길이와 비교
        try:
            video = VideoFileClip(self.video_path)
            total_dur = video.duration
            if clip_end > total_dur:
                clip_end = total_dur
        except Exception as e:
            QMessageBox.critical(self, "오류", f"비디오를 열 수 없습니다:\n{str(e)}")
            return

        # 저장할 클립 경로
        base_name = os.path.splitext(os.path.basename(self.video_path))[0]
        out_dir = f"{base_name}_graph_clips"
        os.makedirs(out_dir, exist_ok=True)
        clip_filename = f"{base_name}_{seconds_to_timestamp(clip_start)}_to_{seconds_to_timestamp(clip_end)}.mp4"
        clip_path = os.path.join(out_dir, clip_filename)

        # 이미 파일이 있으면 재생만, 없으면 생성 후 재생
        if not os.path.exists(clip_path):
            try:
                subclip = video.subclip(clip_start, clip_end)
                subclip.write_videofile(clip_path, codec="libx264", audio_codec="aac", logger=None)
            except Exception as e:
                QMessageBox.critical(self, "오류", f"클립 생성 실패:\n{str(e)}")
                video.close()
                return

        video.close()

        # 클립 재생 (Windows: os.startfile / macOS: open / Linux: xdg-open 등 사용 가능)
        try:
            os.startfile(clip_path)
        except AttributeError:
            # Windows가 아닐 경우
            if sys.platform.startswith("darwin"):
                os.system(f"open \"{clip_path}\"")
            else:
                os.system(f"xdg-open \"{clip_path}\"")

if __name__ == "__main__":
    if len(sys.argv) < 5:
        print("사용법: python graph_viewer.py <video.mp4> <speed_json> <chat_emo_json> <audio_emo_json>")
        sys.exit(1)

    video_path     = sys.argv[1]
    speed_json     = sys.argv[2]
    chat_emo_json  = sys.argv[3]
    audio_emo_json = sys.argv[4]

    app = QApplication(sys.argv)
    gv = GraphViewer(video_path, speed_json, chat_emo_json, audio_emo_json)
    gv.show()
    sys.exit(app.exec_())
