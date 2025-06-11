import os
import json
import sys
import subprocess
from datetime import datetime
from moviepy.editor import VideoFileClip
from PyQt5.QtWidgets import QApplication, QLabel, QWidget, QVBoxLayout, QProgressBar
from PyQt5.QtCore import Qt, QThread, pyqtSignal

# ---------- 1. 유틸 함수 ----------

def timestamp_to_seconds(ts: str) -> int:
    """
    "HH-MM-SS" 형식의 스트링(ts)을 초 단위 정수로 반환.
    """
    ts = ts.replace("-", ":")
    t = datetime.strptime(ts, "%H:%M:%S")
    return t.hour * 3600 + t.minute * 60 + t.second

def seconds_to_timestamp(sec: int) -> str:
    """
    초(sec)를 "HH-MM-SS" 형식의 스트링으로 변환.
    """
    hh = sec // 3600
    mm = (sec % 3600) // 60
    ss = sec % 60
    return f"{hh:02d}-{mm:02d}-{ss:02d}"

def merge_jsons(emotion_path: str, speed_path: str, audio_path: str, output_path: str):
    """
    1) emotion_by_line.json, speed_rankings.json, audio_emotion.json을 모두 읽어들인다.
    2) speed_rankings에서 rank가 전체 중절값(중간 랭크) 이하인 항목들만 추출.
    3) 각 채팅속도 구간의 start_time을 기준으로
       - 채팅창 감정(emotion_by_line)과 매핑
       - 이전 5초 구간에 해당하는 파일명(예: "00-01-10")을 audio_emotion.json에서 찾아 음성 감정 추출
       - chat_speed는 원래 speed_data의 'line_count' 값을 사용
       - 최종 하이라이트 구간을 “음성시작(prev_start) ~ 채팅끝(chat_end)” (10초)으로 지정
    4) 결과를 output_path에 JSON 형태로 저장
    """

    with open(emotion_path, 'r', encoding='utf-8') as f1:
        emotion_data = json.load(f1)
    with open(speed_path, 'r', encoding='utf-8') as f2:
        speed_data = json.load(f2)
    with open(audio_path, 'r', encoding='utf-8') as f3:
        audio_data = json.load(f3)

    # 2) 전체 rank 범위, 중간 랭크 계산
    all_ranks = [entry["rank"] for entry in speed_data]
    if not all_ranks:
        print("speed_rankings.json에 데이터가 없습니다.")
        return
    max_rank = max(all_ranks)
    median_rank = max_rank // 2  # 예: max_rank=10이면 median_rank=5

    # 3) rank <= median_rank에 해당하는 부분만 고른 뒤, rank 오름차순으로 정렬
    selected_speed = [s for s in speed_data if s["rank"] <= median_rank]
    selected_speed.sort(key=lambda x: x["rank"])


    merged = []
    for s in selected_speed:
        chat_start = s["start_time"]  # 예: "00-01-15"
        chat_end = s["end_time"]  # 예: "00-01-19"
        chat_speed = s["line_count"]
        chat_rank = s["rank"]
        # 3-1) 채팅창 감정 매핑
        matching_emotion = next(
            (e for e in emotion_data if e["start_time"] == chat_start),
            None
        )
        chat_emotion = matching_emotion["emotion"] if matching_emotion else "중립"

        raw_percent = matching_emotion.get("emotion_percent", 0) if matching_emotion else 0
        chat_emotion_percent = round(raw_percent, 2)

        # 3-2) 이전 5초 구간 계산
        start_sec = timestamp_to_seconds(chat_start)
        prev_start_sec = start_sec - 5
        if prev_start_sec < 0:
            # 이전 구간이 없으면, 채팅구간만 5초로 처리
            prev_start_ts = chat_start
        else:
            prev_start_ts = seconds_to_timestamp(prev_start_sec)

        # 3-3) audio_emotion.json에서 prev_start_ts에 대응하는 감정 찾기
        matching_audio = next(
            (a for a in audio_data if a["filename"] == prev_start_ts),
            None
        )
        audio_emotion = matching_audio["emotion"] if matching_audio else "중립"

        # 3-4) 최종 하이라이트 구간: prev_start_ts ~ chat_end (10초)
        highlight_entry = {
            "start_time": prev_start_ts,
            "end_time": chat_end,
            "chat_emotion": chat_emotion,
            "chat_emotion_percent": chat_emotion_percent,
            "audio_emotion": audio_emotion,
            "chat_speed": chat_speed,
            "chat_rank": chat_rank
            # 예시: chat_speed나 그 외 메타데이터를 추가하려면 여기에 넣는다.
        }
        merged.append(highlight_entry)

    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(merged, f, indent=2, ensure_ascii=False)


# ---------- 2. 클립 생성 + 썸네일 추출 ----------

def generate_clips_and_thumbnails(video_path, json_input_path, output_dir="highlight_output"):
    """
    1) 전달받은 JSON(merge_jsons 결과)을 읽어,
       각 entry별로 비디오에서 해당 구간(start_time~end_time)을 잘라낸 뒤,
       클립(.mp4)과 썸네일(.png)을 output_dir에 생성.
    2) 각 하이라이트 항목에 clip, screenshot 경로와 기존 메타데이터(채팅감정, 음성감정)를 추가하여
       최종 highlight_result.json을 반환.
    """
    os.makedirs(output_dir, exist_ok=True)
    video = VideoFileClip(video_path)

    with open(json_input_path, "r", encoding="utf-8") as f:
        highlights = json.load(f)

    enriched = []
    for idx, item in enumerate(highlights, start=1):
        start_sec = timestamp_to_seconds(item["start_time"])
        end_sec = timestamp_to_seconds(item["end_time"])
        img_path = os.path.join(output_dir, f"highlight{idx}.png")

        with video.subclip(start_sec, end_sec) as clip:
            clip.save_frame(img_path, t=0.5)

        item["screenshot"] = img_path
        enriched.append(item)

    final_path = os.path.join(output_dir, "highlight_result.json")
    with open(final_path, "w", encoding="utf-8") as f:
        json.dump(enriched, f, indent=2, ensure_ascii=False)
    return final_path

# ---------- 3. QThread + 로딩 ----------

class HighlightWorker(QThread):
    """
    백그라운드에서 하이라이트 JSON 병합 및 클립 생성 작업을 수행하고,
    완료되면 finished 시그널로 결과 JSON 경로를 전달한다.
    """

    finished = pyqtSignal(str)

    def __init__(self, video_path, emotion_path, speed_path, audio_path, merged_json_path, output_dir):
        super().__init__()
        self.video_path = video_path
        self.emotion_path = emotion_path
        self.speed_path = speed_path
        self.audio_path = audio_path
        self.merged_json_path = merged_json_path
        self.output_dir = output_dir

    def run(self):
        # 1) 세 JSON 병합
        merge_jsons(self.emotion_path, self.speed_path, self.audio_path, self.merged_json_path)
        # 2) 클립 생성 + 썸네일
        output_json = generate_clips_and_thumbnails(self.video_path, self.merged_json_path, self.output_dir)
        self.finished.emit(output_json)

class LoadingUI(QWidget):
    """
    하이라이트 생성 작업 중에 “로딩 중” 표시를 보여주는 간단한 윈도우.
    """
    def __init__(self, video_path, emotion_path, speed_path,audio_path, merged_json_path, output_dir):
        super().__init__()
        self.video_path = video_path

        self.setWindowTitle("High-Mate - 하이라이트 생성 중")
        self.setGeometry(400, 300, 400, 150)
        self.setStyleSheet("background-color: black;")

        layout = QVBoxLayout()
        label = QLabel("⏳ 하이라이트 생성 중입니다...\n잠시만 기다려 주세요.")
        label.setStyleSheet("color: white; font-size: 30px;")
        label.setAlignment(Qt.AlignCenter)
        self.progress = QProgressBar()
        self.progress.setRange(0, 0)

        layout.addWidget(label)
        layout.addWidget(self.progress)
        self.setLayout(layout)

        self.worker = HighlightWorker(video_path, emotion_path, speed_path, audio_path, merged_json_path, output_dir)
        self.worker.finished.connect(self.launch_viewer)
        self.worker.start()

    def launch_viewer(self, json_path):
        """
        하이라이트 JSON이 완성되면, 현재 로딩 창을 닫고
        highlight_viewer.py를 실행하도록 한다.
        """
        self.close()
        subprocess.run([sys.executable,"highlight_viewer.py",self.video_path,json_path])

# ---------- 4. json병합, 하이라이트 클립생성 실행 ----------

if __name__ == "__main__":
    """
    사용법 예시:
        python highlight.py \
            <video_path> \
            <base_name>_output/emotion_by_line.json \
            <base_name>_output/speed_rankings.json \
            <base_name>_output/audio_emotion.json \
            <base_name>_output/highlight_output
    """

    if len(sys.argv) != 6:
        print("사용법: python highlight.py <video_path> <emotion_json> <speed_json> <audio_json> <output_dir>")
        sys.exit(1)

    video_path       = sys.argv[1]
    emotion_path     = sys.argv[2]  # ex) 123_output/emotion_by_line.json
    speed_path       = sys.argv[3]  # ex) 123_output/speed_rankings.json
    audio_path       = sys.argv[4]  # ex) 123_output/audio_emotion.json
    highlight_output = sys.argv[5]  # ex) 123_output/highlight_output

    # 필수 파일/폴더 존재 여부 검사
    if not os.path.exists(video_path):
        print(f"Video 파일이 없습니다: {video_path}")
        sys.exit(1)
    for p in (emotion_path, speed_path, audio_path):
        if not os.path.exists(p):
            print(f"필요한 JSON 파일이 없습니다: {p}")
            sys.exit(1)

    # 중간 병합 JSON 경로
    merged_json_path = os.path.join(os.path.dirname(highlight_output), "merged_highlight.json")
    os.makedirs(os.path.dirname(highlight_output), exist_ok=True)

    # PyQt 로딩 창 실행
    app = QApplication(sys.argv)
    loader = LoadingUI(video_path, emotion_path, speed_path, audio_path, merged_json_path, highlight_output)
    loader.show()
    sys.exit(app.exec_())