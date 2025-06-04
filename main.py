import sys
import os
from PyQt5.QtWidgets import QApplication, QWidget, QLabel, QPushButton, QFileDialog, QVBoxLayout, QMessageBox, QProgressBar
from PyQt5.QtGui import QFont
from PyQt5.QtCore import Qt, QThread, pyqtSignal
import moviepy.editor as mp
import subprocess
from drag import ChatDragWindow


class AudioExtractionThread(QThread):
    progress = pyqtSignal(int)
    done = pyqtSignal(str)
    error = pyqtSignal(str)

    def __init__(self, video_path):
        super().__init__()
        self.video_path = video_path

    def run(self):
        try:
            video = mp.VideoFileClip(self.video_path)
            base_name = os.path.splitext(os.path.basename(self.video_path))[0]

            wav_dir = f"{base_name}_wav"
            os.makedirs(wav_dir, exist_ok=True)

            duration = int(video.duration)
            # 4) 5초 간격으로 잘라서 WAV 파일 생성
            for start in range(0, duration, 5):
                end = min(start + 5, duration)
                h = start // 3600
                m = (start % 3600) // 60
                s = start % 60
                timestamp = f"{h:02d}-{m:02d}-{s:02d}"
                wav_path = os.path.join(wav_dir, f"{timestamp}.wav")

                # 오디오 서브클립만 추출
                audio_clip = video.audio.subclip(start, end)
                audio_clip.write_audiofile(wav_path, logger=None)

                # 진행률 신호: 전체 길이에 대한 비율로 계산
                self.progress.emit(int((start / duration) * 100))

            # 마지막에는 100%로 설정
            self.progress.emit(100)

            # 5) 작업 완료 시, base_name(영상 이름)을 전달
            self.done.emit(wav_dir)
        except Exception as e:
            self.error.emit(str(e))


class MainWindow(QWidget):
    def __init__(self):
        super().__init__()
        self.video_path = None
        self.wav_dir = None
        self.initUI()

    def initUI(self):
        self.setWindowTitle('ScenePulsE - 영상 업로드')
        self.setGeometry(100, 100, 600, 450)
        self.setStyleSheet("background-color: black;")

        # 상단 안내 문구
        self.label = QLabel('영상을 삽입해주세요', self)
        self.label.setStyleSheet("color: white;")
        self.label.setFont(QFont('Arial', 16))
        self.label.setAlignment(Qt.AlignLeft | Qt.AlignTop)

        # 영상 선택 버튼
        self.upload_btn = QPushButton(self)
        self.upload_btn.setFixedSize(800, 450)
        self.upload_btn.setText("+")
        self.upload_btn.setFont(QFont('Arial', 150))
        self.upload_btn.setStyleSheet("background-color: white; color: black; border-radius: 50px;")
        self.upload_btn.clicked.connect(self.openFileNameDialog)

        # 진행 상태를 보여줄 ProgressBar
        self.progress_bar = QProgressBar(self)
        self.progress_bar.setValue(0)
        self.progress_bar.setStyleSheet("color: white; text-align: center;")
        self.progress_bar.setTextVisible(True)

        # NEXT 버튼 (프레임 드래그 윈도우로 넘어가기)
        self.next_btn = QPushButton('NEXT', self)
        self.next_btn.setStyleSheet("color: white; font-weight: bold; font-size: 30px;")
        self.next_btn.clicked.connect(self.openDragWindow)
        # NEXT 버튼은 오디오 분할 완료 시 활성화
        self.next_btn.setEnabled(False)

        # 하단 로고
        self.logo_label = QLabel('ScenePulsE', self)
        self.logo_label.setStyleSheet("color: white;")
        self.logo_label.setFont(QFont('Arial', 18, QFont.Bold))
        self.logo_label.setAlignment(Qt.AlignCenter)

        layout = QVBoxLayout()
        layout.addWidget(self.label, alignment=Qt.AlignLeft | Qt.AlignTop)
        layout.addStretch()
        layout.addWidget(self.upload_btn, alignment=Qt.AlignHCenter)
        layout.addWidget(self.progress_bar)
        layout.addStretch()
        layout.addWidget(self.next_btn, alignment=Qt.AlignHCenter)
        layout.addWidget(self.logo_label, alignment=Qt.AlignHCenter)

        self.setLayout(layout)

    def openFileNameDialog(self):
        fname, _ = QFileDialog.getOpenFileName(self, 'Open file', '', 'Video files (*.mp4 *.avi *.mov)')
        if fname:
            self.video_path = fname
            print(f"Selected file: {self.video_path}")
            self.start_audio_extraction()

    def start_audio_extraction(self):
        self.progress_bar.setValue(0)
        self.thread = AudioExtractionThread(self.video_path)
        self.thread.progress.connect(self.progress_bar.setValue)
        self.thread.done.connect(self.on_extraction_done)
        self.thread.error.connect(self.on_extraction_error)
        self.thread.start()

    def on_extraction_done(self, wav_dir: str):
        self.wav_dir = wav_dir
        self.show_message("Success", f"오디오 분할 완료: '{wav_dir}' 폴더 생성됨")
        # NEXT 버튼 활성화
        self.next_btn.setEnabled(True)

    def on_extraction_error(self, error):
        self.show_message("Error", f"오디오 추출 실패 다시 비디오를 넣어주세요", error=True)

    def show_message(self, title, text, error=False):
        msg = QMessageBox(self)
        msg.setWindowTitle(title)
        msg.setText(text)
        msg.setStyleSheet("QLabel { color: white; } QPushButton { color: black; }")
        msg.setStandardButtons(QMessageBox.Ok)
        if error:
            msg.setIcon(QMessageBox.Critical)
        else:
            msg.setIcon(QMessageBox.Information)
        msg.exec_()

    def openDragWindow(self):
        # NEXT 버튼을 누르면 원래 있던 채팅 영역 드래그용 윈도우로 넘어갑니다.
        if self.video_path:

            self.hide()
            self.drag_window = ChatDragWindow(self.video_path, self.wav_dir)
            self.drag_window.show()
        else:
            self.show_message("경고", "먼저 영상을 업로드해주세요!", error=True)

if __name__ == '__main__':
    app = QApplication(sys.argv)
    main = MainWindow()
    main.show()
    sys.exit(app.exec_())