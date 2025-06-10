# graph_viewer.py
import sys
import os
import json
import math
import mplcursors
import subprocess

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

        # Matplotlib Figure / Canvas (다크 테마 적용)
        self.fig = Figure(figsize=(9, 5), facecolor="#121212")
        self.canvas = FigureCanvas(self.fig)
        layout.addWidget(self.canvas)

        self.ax = self.fig.add_subplot(111, facecolor="#121212")
        self.ax.set_title("Chat Speed + Emotion", fontsize=14, color="white", weight="bold")
        self.ax.set_xlabel("Time (seconds)", fontsize=12, color="white")
        self.ax.set_ylabel("Chat Speed (lines per 5s)", fontsize=12, color="white")
        self.ax.tick_params(axis='both', labelsize=10, colors='white')
        self.ax.grid(True, linestyle='--', linewidth=0.5, color="#444444")

        legend_handles = []
        for emo, color in COLOR_MAP.items():
            legend_handles.append(Patch(facecolor=color, edgecolor="white", label=emo))
        legend = self.ax.legend(
            handles=legend_handles,
            title="Emotion",
            loc="upper left",
            bbox_to_anchor=(0.01, 0.99),
            fontsize=9,
            title_fontsize=10,
            frameon=True,
            framealpha=0.5,
            facecolor="#222222",
            edgecolor="white"
        )

        legend.get_title().set_color("white")
        for text in legend.get_texts():
            text.set_color("white")

        outer_size = 100
        inner_size = 50

        scatter_artists = []
        for cx, ly, chat_e, audio_e, st, et in self.points:
            artist_outer = self.ax.scatter(
                cx, ly,
                c=COLOR_MAP.get(audio_e, "black"),
                s=outer_size,
                marker="o",
                edgecolors="#222222",
                linewidths=0.8,
                alpha=0.5,
                zorder=1
            )
            artist_inner = self.ax.scatter(
                cx, ly,
                c=COLOR_MAP.get(chat_e, "black"),
                s=inner_size,
                marker="o",
                edgecolors="#222222",
                linewidths=0.8,
                alpha=0.8,
                zorder=2
            )
            scatter_artists.append((artist_inner, st, chat_e, audio_e))

        cursor = mplcursors.cursor([tpl[0] for tpl in scatter_artists], hover=True)

        @cursor.connect("add")
        def on_add(sel):
            picked_artist = sel.artist
            for art, start_ts, chat_e, audio_e in scatter_artists:
                if art == picked_artist:
                    hh, mm, ss = start_ts.split("-")
                    time_str = f"{int(hh):02d}:{int(mm):02d}:{int(float(ss)):02d}"
                    text = f"{time_str}\n채팅: {chat_e}\n스트리머: {audio_e}"
                    sel.annotation.set_text(text)
                    sel.annotation.get_bbox_patch().set_alpha(0.7)
                    sel.annotation.get_bbox_patch().set_facecolor("#eeeeee")
                    sel.annotation.get_bbox_patch().set_edgecolor("#222222")
                    sel.annotation.get_bbox_patch().set_linewidth(0.5)
                    break

        # 하단 로고 추가
        self.ax.text(
            0.5, -0.13,  # ← y좌표 -0.15 → 0.01로 수정
            "ScenePulsE",
            fontsize=22,
            color='white',
            ha='center',
            va='bottom',
            fontweight='bold',
            fontname='Arial',
            transform=self.ax.transAxes
        )

        self.canvas.mpl_connect("button_press_event", self.on_click)
        self.canvas.draw()

    def on_click(self, event):
        if event.inaxes != self.ax:
            return

        click_x = event.xdata
        distances = [abs(click_x - pt[0]) for pt in self.points]
        idx = int(min(range(len(distances)), key=lambda i: distances[i]))

        center_sec, line_count, chat_e, audio_e, start_ts, end_ts = self.points[idx]

        clip_start = max(center_sec - 15.0, 0.0)
        clip_end = center_sec + 15.0
        duration = clip_end - clip_start

        # ffplay로 바로 재생 (–autoexit: 재생 후 창 자동 종료)
        try:
            subprocess.Popen([
                "ffplay",
                "-ss", str(clip_start),
                "-t", str(duration),
                "-autoexit",
                self.video_path
            ], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        except Exception as e:
            QMessageBox.critical(self, "오류", f"FFplay 실행 실패:\n{str(e)}")
        """
        try:
            video = VideoFileClip(self.video_path)
            total_dur = video.duration
            if clip_end > total_dur:
                clip_end = total_dur
        except Exception as e:
            QMessageBox.critical(self, "오류", f"비디오를 열 수 없습니다:\n{str(e)}")
            return

        base_name = os.path.splitext(os.path.basename(self.video_path))[0]
        out_dir = f"{base_name}_graph_clips"
        os.makedirs(out_dir, exist_ok=True)
        clip_filename = f"{base_name}_{seconds_to_timestamp(clip_start)}_to_{seconds_to_timestamp(clip_end)}.mp4"
        clip_path = os.path.join(out_dir, clip_filename)

        if not os.path.exists(clip_path):
            try:
                subclip = video.subclip(clip_start, clip_end)
                subclip.write_videofile(clip_path, codec="libx264", audio_codec="aac", logger=None)
            except Exception as e:
                QMessageBox.critical(self, "오류", f"클립 생성 실패:\n{str(e)}")
                video.close()
                return

        video.close()

        try:
            os.startfile(clip_path)
        except AttributeError:
            if sys.platform.startswith("darwin"):
                os.system(f"open \"{clip_path}\"")
            else:
                os.system(f"xdg-open \"{clip_path}\"") 
        """

if __name__ == "__main__":
    #python graph_viewer.py capston2.mp4 capston2_output/speed_rankings.json capston2_output/emotion_by_line.json capston2_output/audio_emotion.json
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

    gv.show()
    sys.exit(app.exec_())
