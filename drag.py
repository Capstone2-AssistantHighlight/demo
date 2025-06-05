# drag.py

import cv2
import sys
import os
import subprocess
from PyQt5.QtWidgets import (
    QApplication, QWidget, QLabel, QPushButton,
    QVBoxLayout, QMessageBox, QProgressBar
)
from PyQt5.QtGui import QPixmap, QFont, QImage, QPainter, QColor
from PyQt5.QtCore import Qt, QRect, QThread, pyqtSignal

from full_pipeline import full_pipeline  # full_pipeline 함수 임포트


class FrameExtractionThread(QThread):
    """
    별도 스레드: 동영상에서 드래그 영역 기준으로 매초 이미지를 crop → 저장하며
    진행률(퍼센트)을 progress 시그널로 메인에 전달함.
    """
    progress = pyqtSignal(int)
    finished = pyqtSignal()

    def __init__(self, video_path, crop_coords, frames_dir):
        super().__init__()
        self.video_path = video_path
        self.crop_x, self.crop_y, self.crop_w, self.crop_h = crop_coords
        self.frames_dir = frames_dir

    def run(self):
        cap = cv2.VideoCapture(self.video_path)
        if not cap.isOpened():
            self.finished.emit()
            return

        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        duration_sec = int(total_frames / fps)

        for sec in range(duration_sec + 1):
            cap.set(cv2.CAP_PROP_POS_MSEC, sec * 1000)
            ret, frame = cap.read()
            if not ret:
                continue

            crop_frame = frame[self.crop_y:self.crop_y + self.crop_h,
                               self.crop_x:self.crop_x + self.crop_w]

            h = sec // 3600
            m = (sec % 3600) // 60
            s = sec % 60
            timestamp = f"{h:02d}-{m:02d}-{s:02d}"
            output_path = os.path.join(self.frames_dir, f"{timestamp}.png")
            cv2.imwrite(output_path, crop_frame)

            percent = int((sec / max(1, duration_sec)) * 100)
            self.progress.emit(percent)

        cap.release()
        self.progress.emit(100)
        self.finished.emit()


class FullPipelineThread(QThread):
    """
    full_pipeline(frames_dir, wav_dir)를 백그라운드로 실행하는 QThread.
    완료되면 finished 시그널을 보냄.
    """
    finished = pyqtSignal()

    def __init__(self, frames_dir, wav_dir):
        super().__init__()
        self.frames_dir = frames_dir
        self.wav_dir    = wav_dir

    def run(self):
        full_pipeline(self.frames_dir, self.wav_dir)
        self.finished.emit()


class DraggableImageLabel(QLabel):
    def __init__(self, pixmap, parent=None):
        super().__init__(parent)
        self.setPixmap(pixmap)
        self.start_pos = None
        self.end_pos = None
        self.selection_rect = None

    def mousePressEvent(self, event):
        if event.button() == Qt.LeftButton:
            self.start_pos = event.pos()

    def mouseMoveEvent(self, event):
        if self.start_pos:
            self.end_pos = event.pos()
            self.selection_rect = QRect(self.start_pos, self.end_pos)
            self.update()

    def mouseReleaseEvent(self, event):
        if event.button() == Qt.LeftButton and self.start_pos:
            self.end_pos = event.pos()
            self.selection_rect = QRect(self.start_pos, self.end_pos)
            self.start_pos = None
            self.end_pos = None
            self.update()

    def paintEvent(self, event):
        super().paintEvent(event)
        if self.selection_rect:
            painter = QPainter(self)
            painter.setPen(QColor(255, 0, 0))
            painter.drawRect(self.selection_rect)


class ChatDragWindow(QWidget):
    def __init__(self, video_path, wav_dir):
        super().__init__()
        self.video_path = video_path
        self.wav_dir = wav_dir
        self.frame = None
        self.pixmap = None
        self.thread = None
        self.pipeline_thread = None
        self.initUI()

    def initUI(self):
        self.setWindowTitle('ScenePulsE - Drag Chat Window')
        self.setGeometry(100, 100, 900, 750)
        self.setStyleSheet("background-color: black;")

        # 안내 문구
        self.label = QLabel('댓글 창을 드래그 해주세요', self)
        self.label.setStyleSheet("color: white;")
        self.label.setFont(QFont('Arial', 16))
        self.label.setAlignment(Qt.AlignLeft | Qt.AlignTop)

        # 비디오 중간 프레임 가져오기
        self.frame, self.pixmap = self.capture_middle_frame(self.video_path)
        if self.pixmap:
            self.image_label = DraggableImageLabel(
                self.pixmap.scaled(900, 600, Qt.KeepAspectRatio), self
            )
            self.image_label.setAlignment(Qt.AlignCenter)
        else:
            self.image_label = QLabel('영상 불러오기 실패', self)
            self.image_label.setStyleSheet("color: red;")

        # 진행률 바 (숨김 상태로 시작)
        self.progress_bar = QProgressBar(self)
        self.progress_bar.setValue(0)
        self.progress_bar.setFixedHeight(25)
        self.progress_bar.setVisible(False)
        self.progress_bar.setStyleSheet("QProgressBar { color: white; }")

        # 1) “채팅창 추출” 버튼: 드래그 영역 확정 → 프레임 추출 스레드 실행
        self.select_btn = QPushButton('채팅창 추출', self)
        self.select_btn.setStyleSheet("color: red; font-weight: bold; font-size: 20px;")
        self.select_btn.clicked.connect(self.on_select_clicked)

        # 2) “분석 시작” 버튼: 프레임 저장 완료 후 활성화 → full_pipeline 실행
        self.next_btn = QPushButton('분석 시작', self)
        self.next_btn.setStyleSheet("color: white; font-weight: bold; font-size: 20px;")
        self.next_btn.setVisible(False)
        self.next_btn.clicked.connect(self.go_to_full_pipeline)

        # 3) “그래프 보기” 버튼: full_pipeline 완료 후 활성화 → graph_viewer 실행
        self.graph_btn = QPushButton('그래프 보기', self)
        self.graph_btn.setStyleSheet("color: white; font-weight: bold; font-size: 20px;")
        self.graph_btn.setVisible(False)
        self.graph_btn.clicked.connect(self.open_graph_viewer)

        # 4) “하이라이트 보기” 버튼: full_pipeline 완료 후 활성화 → highlight.py 실행
        self.highlight_btn = QPushButton('하이라이트 보기', self)
        self.highlight_btn.setStyleSheet("color: white; font-weight: bold; font-size: 20px;")
        self.highlight_btn.setVisible(False)
        self.highlight_btn.clicked.connect(self.open_highlight_viewer)

        # 로고
        self.logo_label = QLabel('ScenePulsE', self)
        self.logo_label.setStyleSheet("color: white;")
        self.logo_label.setFont(QFont('Arial', 18, QFont.Bold))
        self.logo_label.setAlignment(Qt.AlignCenter)

        # 레이아웃 구성
        layout = QVBoxLayout()
        layout.addWidget(self.label, alignment=Qt.AlignLeft | Qt.AlignTop)
        layout.addWidget(self.image_label, alignment=Qt.AlignHCenter)
        layout.addWidget(self.progress_bar)
        layout.addWidget(self.select_btn, alignment=Qt.AlignHCenter)
        layout.addWidget(self.next_btn, alignment=Qt.AlignHCenter)
        layout.addWidget(self.graph_btn, alignment=Qt.AlignHCenter)
        layout.addWidget(self.highlight_btn, alignment=Qt.AlignHCenter)
        layout.addWidget(self.logo_label, alignment=Qt.AlignCenter)
        self.setLayout(layout)

    def capture_middle_frame(self, video_path):
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            print("Error: Cannot open video")
            return None, None

        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        middle_frame_num = total_frames // 2
        cap.set(cv2.CAP_PROP_POS_FRAMES, middle_frame_num)
        ret, frame = cap.read()
        cap.release()

        if ret:
            height, width, channel = frame.shape
            bytes_per_line = 3 * width
            qimg = QImage(
                frame.data, width, height,
                bytes_per_line, QImage.Format_RGB888
            ).rgbSwapped()
            return frame, QPixmap.fromImage(qimg)
        else:
            return None, None

    def on_select_clicked(self):
        """
        1) 드래그 영역이 없으면 경고창 출력
        2) 있으면, crop 좌표 계산 → FrameExtractionThread 실행
        3) 프로그래스바 보이기, 스레드의 progress 시그널 연결
        """
        if not self.image_label.selection_rect:
            QMessageBox.warning(self, "경고", "드래그 영역을 먼저 선택해주세요!")
            return

        rect = self.image_label.selection_rect
        display_size = self.image_label.size()
        orig_h, orig_w, _ = self.frame.shape

        x_ratio = orig_w / display_size.width()
        y_ratio = orig_h / display_size.height()

        crop_x = int(rect.x() * x_ratio)
        crop_y = int(rect.y() * y_ratio)
        crop_w = int(rect.width() * x_ratio)
        crop_h = int(rect.height() * y_ratio)

        base_name = os.path.splitext(os.path.basename(self.video_path))[0]
        frames_dir = f"{base_name}_frames"
        os.makedirs(frames_dir, exist_ok=True)

        self.progress_bar.setValue(0)
        self.progress_bar.setVisible(True)
        self.select_btn.setVisible(False)   # 채팅창 추출 버튼 숨기기

        # FrameExtractionThread 실행
        self.thread = FrameExtractionThread(
            video_path=self.video_path,
            crop_coords=(crop_x, crop_y, crop_w, crop_h),
            frames_dir=frames_dir
        )
        self.thread.progress.connect(self.progress_bar.setValue)
        self.thread.finished.connect(self.on_extraction_done)
        self.thread.start()

    def on_extraction_done(self):
        """
        FrameExtractionThread가 완료된 후 호출
        → 프로그래스바 숨기고, “분석 시작” 버튼 보이기
        """
        self.progress_bar.setVisible(False)
        QMessageBox.information(
            self, "완료",
            f"{os.path.splitext(os.path.basename(self.video_path))[0]}_frames에 프레임이 저장되었습니다!"
        )
        # “분석 시작” 버튼만 보이도록 설정
        self.next_btn.setVisible(True)
        self.graph_btn.setVisible(False)
        self.highlight_btn.setVisible(False)

    def go_to_full_pipeline(self):
        """
        '분석 시작' 버튼을 누르면, full_pipeline을 별도 스레드에서 실행.
        완료 시점에 on_pipeline_done 호출
        """
        self.next_btn.setVisible(False)     # 분석 시작 버튼 숨기기
        self.graph_btn.setVisible(False)
        self.highlight_btn.setVisible(False)

        base_name = os.path.splitext(os.path.basename(self.video_path))[0]
        frames_dir = f"{base_name}_frames"
        wav_dir    = self.wav_dir

        self.pipeline_thread = FullPipelineThread(frames_dir, wav_dir)
        self.pipeline_thread.finished.connect(self.on_pipeline_done)
        self.pipeline_thread.start()

    def on_pipeline_done(self):
        """
        FullPipelineThread의 run()이 끝나면 호출.
        → “그래프 보기” 및 “하이라이트 보기” 버튼 보이기
        """
        QMessageBox.information(self, "완료", "분석이 모두 완료되었습니다!")
        self.graph_btn.setVisible(True)
        self.highlight_btn.setVisible(True)

    def open_graph_viewer(self):
        """
        full_pipeline이 이미 완료되었거나, speed/emotion/audio JSON이 준비되었는지 확인 후
        graph_viewer.py를 실행해서 새로운 창 띄우기
        """
        base_name = os.path.splitext(os.path.basename(self.video_path))[0]
        output_dir = f"{base_name}_output"
        speed_json    = os.path.join(output_dir, "speed_rankings.json")
        chat_emo_json = os.path.join(output_dir, "emotion_by_line.json")
        audio_emo_json= os.path.join(output_dir, "audio_emotion.json")

        for p in (speed_json, chat_emo_json, audio_emo_json):
            if not os.path.exists(p):
                QMessageBox.warning(self, "경고", f"{p} 파일을 찾을 수 없습니다.\n먼저 '분석 시작'을 눌러주세요.")
                return

        subprocess.Popen([
            sys.executable, "graph_viewer.py",
            self.video_path,
            speed_json,
            chat_emo_json,
            audio_emo_json
        ])

    def open_highlight_viewer(self):
        """
        full_pipeline이 완료된 뒤에, highlight.py를 실행하고
        (highlight.py 내부에서 highlight_result.json 생성 → highlight_viewer 호출)
        """
        base_name = os.path.splitext(os.path.basename(self.video_path))[0]
        output_dir = f"{base_name}_output"
        emotion_json = os.path.join(output_dir, "emotion_by_line.json")
        speed_json   = os.path.join(output_dir, "speed_rankings.json")
        audio_json   = os.path.join(output_dir, "audio_emotion.json")
        highlight_output = os.path.join(output_dir, "highlight_output")

        for p in (emotion_json, speed_json, audio_json):
            if not os.path.exists(p):
                QMessageBox.warning(self, "경고", f"{p} 파일을 찾을 수 없습니다.\n먼저 '분석 시작'을 눌러주세요.")
                return

        subprocess.Popen([
            sys.executable, "highlight.py",
            self.video_path,
            emotion_json,
            speed_json,
            audio_json,
            highlight_output
        ])


if __name__ == '__main__':
    app = QApplication(sys.argv)
    video_path = sys.argv[1] if len(sys.argv) > 1 else ''
    wav_dir    = sys.argv[2] if len(sys.argv) > 2 else ''
    window = ChatDragWindow(video_path, wav_dir)
    window.show()
    sys.exit(app.exec_())
