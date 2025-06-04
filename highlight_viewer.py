import sys
import os
import json
from PyQt5.QtWidgets import (
    QApplication, QWidget, QLabel, QVBoxLayout, QHBoxLayout,
    QScrollArea, QFrame, QComboBox
)
from PyQt5.QtGui import QPixmap, QFont, QCursor
from PyQt5.QtCore import Qt

EMOJI_MAP = {
    "행복": "😄", "슬픔": "😢", "분노": "😡", "중립": "😐"
}

class HighlightCard(QFrame):
    def __init__(self, image_path, start_time, end_time, chat_emotion, audio_emotion, chat_speed, clip_path):
        super().__init__()
        self.setStyleSheet("background-color: #1e1e1e; border-radius: 10px;")
        self.setFixedHeight(220)

        #썸네일
        img_label = QLabel()
        img_label.setPixmap(QPixmap(image_path).scaled(300, 180, Qt.KeepAspectRatio))
        img_label.setCursor(QCursor(Qt.PointingHandCursor))
        img_label.mousePressEvent = lambda event: os.startfile(clip_path)

        info_label = QLabel(
            f"⏱️ {start_time} ~ {end_time}\n"
            f"💬 채팅감정: {EMOJI_MAP.get(chat_emotion, '')} {chat_emotion}\n"
            f"⚡ 채팅속도: {chat_speed}줄/5초\n"
            f"🎙️ 음성감정: {EMOJI_MAP.get(audio_emotion, '')} {audio_emotion}"
        )
        info_label.setStyleSheet("color: white;")
        info_label.setFont(QFont("Arial", 15))
        info_label.setWordWrap(True)

        layout = QHBoxLayout()
        layout.addWidget(img_label)
        layout.addWidget(info_label)
        layout.setContentsMargins(10, 10, 10, 10)
        self.setLayout(layout)

class HighlightOutputWindow(QWidget):
    """
    최종 highlight_result.json을 읽어,
    감정 필터링(“전체/행복/슬픔/분노/중립”) 드롭다운을 제공한다.
    채팅속도도 함께 표시한다.
    """
    def __init__(self, json_path):
        super().__init__()
        self.json_path = json_path
        self.setWindowTitle("ScenePulsE - 하이라이트 결과")
        self.setGeometry(100, 100, 800, 600)
        self.setStyleSheet("background-color: black;")

        self.main_layout = QVBoxLayout()
        self.setLayout(self.main_layout)

        # 상단 드롭다운 메뉴 오른쪽 정렬
        top_bar = QHBoxLayout()
        top_bar.addStretch()

        self.combo = QComboBox()
        self.combo.setFixedWidth(160)
        self.combo.setStyleSheet("background-color: gray; color: black; font-size: 14px;")
        self.combo.addItem("전체")
        self.combo.addItems(["행복", "슬픔", "분노", "중립"])
        self.combo.currentTextChanged.connect(self.update_display)
        top_bar.addWidget(self.combo)

        self.main_layout.addLayout(top_bar)

        # 스크롤 가능한 영역
        self.scroll_area = QScrollArea()
        self.scroll_area.setWidgetResizable(True)
        self.main_layout.addWidget(self.scroll_area)

        # 하단 로고
        self.logo = QLabel("ScenePulsE")
        self.logo.setFont(QFont("Arial", 20, QFont.Bold))
        self.logo.setStyleSheet("color: white;")
        self.logo.setAlignment(Qt.AlignCenter)
        self.main_layout.addWidget(self.logo)

        # 데이터 로드 및 초기 표시
        self.load_data()
        self.update_display("전체")

    def load_data(self):
        with open(self.json_path, "r", encoding="utf-8") as f:
            self.highlights = json.load(f)

    def update_display(self, selected_emotion):
        """
        드롭다운에서 감정을 선택했을 때 호출.
        '전체'를 선택하면 모든 카드를, 아니면 해당 대분류(chat_emotion)을 갖는 카드만 보여준다.
        """
        content_widget = QWidget()
        content_layout = QVBoxLayout(content_widget)
        content_layout.setAlignment(Qt.AlignTop)  # 항상 위부터 정렬됨

        for item in self.highlights:
            if selected_emotion == "전체" or item["chat_emotion"] == selected_emotion:
                card = HighlightCard(
                    image_path=item["screenshot"],
                    start_time=item["start_time"],
                    end_time=item["end_time"],
                    chat_emotion=item["chat_emotion"],
                    audio_emotion=item["audio_emotion"],
                    chat_speed=item["chat_speed"],
                    clip_path=item["clip"]
                )
                content_layout.addWidget(card)

        self.scroll_area.setWidget(content_widget)


if __name__ == '__main__':
    if len(sys.argv) != 2:
        print("사용법: python highlight_viewer.py <highlight_result.json 경로>")
        sys.exit(1)

    json_path = sys.argv[1]
    if not os.path.exists(json_path):
        print(f"{json_path} 파일이 존재하지 않습니다.")
        sys.exit(1)

    app = QApplication(sys.argv)
    window = HighlightOutputWindow(json_path)
    window.show()
    sys.exit(app.exec_())