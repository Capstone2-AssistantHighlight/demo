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
        """
        video_path: 원본 비디오 경로 (예: '123.mp4')
        crop_coords: (x, y, w, h) 형태의 튜플 (픽셀 단위)
        frames_dir: 잘린 이미지를 저장할 폴더 경로 (예: '123_frames')
        """
        super().__init__()
        self.video_path = video_path
        self.crop_x, self.crop_y, self.crop_w, self.crop_h = crop_coords
        self.frames_dir = frames_dir

    def run(self):
        cap = cv2.VideoCapture(self.video_path)
        if not cap.isOpened():
            # 비디오를 열 수 없으면 곧바로 종료 시그널
            self.finished.emit()
            return

        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        duration_sec = int(total_frames / fps)

        # 매 초마다 이미지 저장: 총 duration_sec + 1번 반복
        for sec in range(duration_sec + 1):
            cap.set(cv2.CAP_PROP_POS_MSEC, sec * 1000)
            ret, frame = cap.read()
            if not ret:
                continue

            # crop 영역: (y:y+h, x:x+w)
            crop_frame = frame[self.crop_y:self.crop_y + self.crop_h,
                               self.crop_x:self.crop_x + self.crop_w]

            h = sec // 3600
            m = (sec % 3600) // 60
            s = sec % 60
            timestamp = f"{h:02d}-{m:02d}-{s:02d}"
            output_path = os.path.join(self.frames_dir, f"{timestamp}.png")
            cv2.imwrite(output_path, crop_frame)

            # 진행률 계산: (sec / duration_sec) * 100
            percent = int((sec / max(1, duration_sec)) * 100)
            self.progress.emit(percent)

        cap.release()
        # 완료 시 100%를 보장
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
        # 실제로 시간이 오래 걸리는 full_pipeline을 여기에 옮겨서 수행
        full_pipeline(self.frames_dir, self.wav_dir)
        # 작업이 모두 끝나면 시그널 방출
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
        self.wav_dir = wav_dir  # main.py에서 넘겨준 <영상이름>_wav 폴더
        self.frame = None
        self.pixmap = None
        self.thread = None
        self.initUI()

    def initUI(self):
        self.setWindowTitle('ScenePulsE - Drag Chat Window')
        self.setGeometry(100, 100, 900, 700)
        self.setStyleSheet("background-color: black;")

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

        # ① 진행률 바 (숨김 상태로 시작)
        self.progress_bar = QProgressBar(self)
        self.progress_bar.setValue(0)
        self.progress_bar.setFixedHeight(25)
        self.progress_bar.setVisible(False)
        self.progress_bar.setStyleSheet("QProgressBar { color: white; }")

        # “채팅창 추출” 버튼: 드래그 영역 확정 → 프레임 추출 스레드 실행
        self.select_btn = QPushButton('채팅창 추출', self)
        self.select_btn.setStyleSheet("color: red; font-weight: bold; font-size: 20px;")
        self.select_btn.clicked.connect(self.on_select_clicked)

        # “분석 시작” 버튼: 프레임 저장 완료 후 활성화
        self.next_btn = QPushButton('분석 시작', self)
        self.next_btn.setStyleSheet("color: white; font-weight: bold; font-size: 20px;")
        self.next_btn.setEnabled(False)
        self.next_btn.clicked.connect(self.go_to_highlight)

        self.logo_label = QLabel('ScenePulsE', self)
        self.logo_label.setStyleSheet("color: white;")
        self.logo_label.setFont(QFont('Arial', 18, QFont.Bold))
        self.logo_label.setAlignment(Qt.AlignCenter)

        layout = QVBoxLayout()
        layout.addWidget(self.label, alignment=Qt.AlignLeft | Qt.AlignTop)
        layout.addWidget(self.image_label, alignment=Qt.AlignHCenter)
        layout.addWidget(self.progress_bar)  # ProgressBar 추가
        layout.addWidget(self.select_btn, alignment=Qt.AlignHCenter)
        layout.addWidget(self.next_btn, alignment=Qt.AlignHCenter)
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

        # 드래그 영역 비율 → 실제 픽셀 좌표로 변환
        rect = self.image_label.selection_rect
        display_size = self.image_label.size()
        orig_h, orig_w, _ = self.frame.shape

        x_ratio = orig_w / display_size.width()
        y_ratio = orig_h / display_size.height()

        crop_x = int(rect.x() * x_ratio)
        crop_y = int(rect.y() * y_ratio)
        crop_w = int(rect.width() * x_ratio)
        crop_h = int(rect.height() * y_ratio)

        # 이미지 저장할 폴더: <base_name>_frames
        base_name = os.path.splitext(os.path.basename(self.video_path))[0]
        frames_dir = f"{base_name}_frames"
        os.makedirs(frames_dir, exist_ok=True)

        # ProgressBar 보이기, 초기화
        self.progress_bar.setValue(0)
        self.progress_bar.setVisible(True)
        self.select_btn.setEnabled(False)  # 중복 클릭 방지

        # 스레드 생성 및 실행
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
        → 프로그래스바 숨기고, “RUN FULL PIPELINE” 버튼 활성화
        """
        self.progress_bar.setVisible(False)
        QMessageBox.information(self, "완료", f"{os.path.splitext(os.path.basename(self.video_path))[0]}_frames에 프레임이 저장되었습니다!")
        self.next_btn.setEnabled(True)

    def go_to_highlight(self):
        """
        1) '분석 시작' 버튼을 누르면, full_pipeline을 별도 스레드에서 실행.
        2) 완료 시점에 highlight.py를 호출하고, 현재 창을 닫음.
        """
        # ① 버튼 중복 클릭 방지
        self.next_btn.setEnabled(False)

        base_name = os.path.splitext(os.path.basename(self.video_path))[0]
        frames_dir = f"{base_name}_frames"
        wav_dir    = self.wav_dir

        # ② pipeline 스레드 생성 및 실행
        self.pipeline_thread = FullPipelineThread(frames_dir, wav_dir)
        self.pipeline_thread.finished.connect(self.on_pipeline_done)

        self.pipeline_thread.start()

    def on_pipeline_done(self):
        """
        FullPipelineThread의 run()이 끝난 직후 호출됨.
        → highlight.py 실행, 현재 창 닫기
        """

        base_name = os.path.splitext(os.path.basename(self.video_path))[0]
        output_folder = f"{base_name}_output/highlight_output"
        os.makedirs(output_folder, exist_ok=True)

        # highlight.py 호출
        subprocess.Popen([
            sys.executable, "highlight.py",
            self.video_path,
            f"{base_name}_output/emotion_by_line.json",
            f"{base_name}_output/speed_rankings.json",
            f"{base_name}_output/audio_emotion.json",
            output_folder
        ])

        # 창 닫기
        self.close()


if __name__ == '__main__':
    app = QApplication(sys.argv)
    # drag.py를 직접 실행할 때에는 video_path, wav_dir 두 개 인자를 전달해야 함
    # 예: python drag.py 123.mp4 123_wav
    video_path = sys.argv[1] if len(sys.argv) > 1 else ''
    wav_dir    = sys.argv[2] if len(sys.argv) > 2 else ''
    window = ChatDragWindow(video_path, wav_dir)
    window.show()
    sys.exit(app.exec_())
