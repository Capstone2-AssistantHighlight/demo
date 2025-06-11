import sys
import os
import json
import subprocess
from PyQt5.QtWidgets import (
    QApplication, QWidget, QLabel, QVBoxLayout,
    QHBoxLayout, QScrollArea, QFrame, QComboBox, QMessageBox
)
from PyQt5.QtGui import QPixmap, QFont, QCursor
from PyQt5.QtCore import Qt

def timestamp_to_seconds(ts: str) -> float:
    hh, mm, ss = ts.split('-')
    return int(hh) * 3600 + int(mm) * 60 + float(ss)

EMOJI_MAP = {
    "행복": "😄", "슬픔": "😢", "분노": "😡", "중립": "😐", "놀람": "😲"
}

class HighlightCard(QFrame):
    def __init__(self, image_path, start_time, end_time, chat_emotion, chat_emotion_percent, audio_emotion, chat_speed, video_path):
        super().__init__()
        self.video_path = video_path
        self.setStyleSheet("background-color: #1e1e1e; border-radius: 10px;")
        self.setFixedHeight(220)

        # 썸네일
        img_label = QLabel()
        img_label.setPixmap(QPixmap(image_path).scaled(300, 180, Qt.KeepAspectRatio))
        img_label.setCursor(QCursor(Qt.PointingHandCursor))

        # 클릭 시 FFplay로 해당 구간 재생
        def play_segment(event):
            start_sec = timestamp_to_seconds(start_time)
            end_sec   = timestamp_to_seconds(end_time)
            duration  = end_sec - start_sec
            cmd = [
                "ffplay",
                "-ss", str(start_sec),
                "-t",  str(duration),
                "-autoexit",
                self.video_path
            ]
            try:
                subprocess.Popen(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            except FileNotFoundError:
                QMessageBox.critical(self, "오류", "ffplay 실행 파일을 찾을 수 없습니다.\nFFmpeg가 설치되어 있고 PATH에 등록되었는지 확인하세요.")
            except Exception as e:
                QMessageBox.critical(self, "오류", f"FFplay 실행 실패:\n{e}")

        img_label.mousePressEvent = play_segment

        info_label = QLabel(
            f"⏱️ {start_time} ~ {end_time}\n"
            f"💬 채팅감정: {EMOJI_MAP.get(chat_emotion,'')} {chat_emotion} ({chat_emotion_percent}%)\n"
            f"⚡ 채팅속도: {chat_speed}줄/5초\n"
            f"🎙️ 음성감정: {EMOJI_MAP.get(audio_emotion, '')} {audio_emotion}"
        )
        info_label.setStyleSheet("color: white;")
        info_label.setFont(QFont("Arial", 15))
        info_label.setWordWrap(True)

        layout = QHBoxLayout(self)
        layout.addWidget(img_label)
        layout.addWidget(info_label)
        layout.setContentsMargins(10, 10, 10, 10)

class HighlightOutputWindow(QWidget):
    def __init__(self, video_path, json_path):
        super().__init__()
        self.video_path = video_path
        self.json_path  = json_path

        self.setWindowTitle("High-Mate - 하이라이트 결과")
        self.setGeometry(100, 100, 800, 600)
        self.setStyleSheet("background-color: black;")

        # 드롭다운 & 레이아웃 설정
        top_bar = QHBoxLayout()
        top_bar.addStretch()
        self.combo = QComboBox()
        self.combo.setFixedWidth(160)
        self.combo.setStyleSheet("background-color: gray; color: black; font-size: 14px;")
        self.combo.addItem("전체")
        self.combo.addItems(["행복", "슬픔", "놀람", "중립", "감정 퍼센트"])
        self.combo.currentTextChanged.connect(self.update_display)

        main_layout = QVBoxLayout(self)
        main_layout.addLayout(top_bar)
        top_bar.addWidget(self.combo)

        self.scroll_area = QScrollArea()
        self.scroll_area.setWidgetResizable(True)
        main_layout.addWidget(self.scroll_area)

        self.logo = QLabel("High-Mate")
        self.logo.setFont(QFont("Arial", 20, QFont.Bold))
        self.logo.setStyleSheet("color: white;")
        self.logo.setAlignment(Qt.AlignCenter)
        main_layout.addWidget(self.logo)

        # 데이터 로드 & 초기 표시
        self.load_data()
        self.update_display("전체")

    def load_data(self):
        with open(self.json_path, "r", encoding="utf-8") as f:
            self.highlights = json.load(f)

    def update_display(self, key):
        items = self.highlights

        if key == "감정 퍼센트":
            items = sorted(items, key=lambda x: x.get("chat_emotion_percent", 0), reverse=True)

        container = QWidget()
        layout = QVBoxLayout(container)
        layout.setAlignment(Qt.AlignTop)

        for item in items:
            if key == "전체" or key == "감정 퍼센트" or item["chat_emotion"] == key:
                card = HighlightCard(
                    image_path=item["screenshot"],
                    start_time=item["start_time"],
                    end_time=item["end_time"],
                    chat_emotion=item["chat_emotion"],
                    chat_emotion_percent=item["chat_emotion_percent"],
                    audio_emotion=item["audio_emotion"],
                    chat_speed=item["chat_speed"],
                    video_path=self.video_path
                )
                layout.addWidget(card)

        self.scroll_area.setWidget(container)

if __name__ == '__main__':
    if len(sys.argv) < 3:
        print("사용법: python highlight_viewer.py <video_path> <highlight_json>")
        sys.exit(1)

    video_path = sys.argv[1]
    json_path  = sys.argv[2]

    if not os.path.exists(video_path):
        print(f"비디오 파일을 찾을 수 없습니다: {video_path}")
        sys.exit(1)
    if not os.path.exists(json_path):
        print(f"하이라이트 JSON 파일을 찾을 수 없습니다: {json_path}")
        sys.exit(1)

    app = QApplication(sys.argv)
    window = HighlightOutputWindow(video_path, json_path)
    window.show()
    sys.exit(app.exec_())
