import warnings
warnings.filterwarnings("ignore", category=FutureWarning)
import os, sys
import torch
import numpy as np
import librosa

import opensmile
from transformers import Wav2Vec2Processor, Wav2Vec2Model

device = torch.device("cuda" if torch.cuda.is_available() else "cpu") 
print(f"Using device : {device}")
# ----- 오디오 처리 함수 -----
def extract_wav2vec_embedding(waveform, model, processor):
    inputs = processor(waveform, sampling_rate=16000, return_tensors="pt")
    input_values = inputs.input_values.to(model.device)
    with torch.no_grad():
        outputs = model(input_values, output_hidden_states=True)
        hidden_states = outputs.hidden_states
    return torch.stack([layer.squeeze(0) for layer in hidden_states], dim=0)[:6]  # (6, T, D)

def extract_egemaps_feature(y, sr=16000):
    smile = opensmile.Smile(
        feature_set=opensmile.FeatureSet.eGeMAPSv02,
        feature_level=opensmile.FeatureLevel.LowLevelDescriptors
    )
    df = smile.process_signal(y, sampling_rate=sr)
    features = df.to_numpy().astype(np.float32)
    if features.shape[0] % 2 == 1:
        features = features[:-1]
    if features.shape[0] >= 2:
        features = features.reshape(-1, 2, features.shape[1]).mean(axis=1)
    return torch.tensor(features, dtype=torch.float32).to("cuda")


import torch
import torch.nn as nn

class FusionModel(nn.Module):
    def __init__(self, w2v_dim=768, os_dim=25, hidden_dim=128, num_layers=6, num_classes=4, dropout=0.2):
        super(FusionModel, self).__init__()

        # 1. 각 w2v 레이어에 대해 Group Conv1D (6 x (768 -> 128))
        self.w2v_layer_convs = nn.ModuleList([
            nn.Sequential(
                nn.Conv1d(in_channels=w2v_dim, out_channels=hidden_dim, kernel_size=1),
                nn.BatchNorm1d(hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout)
            ) for _ in range(num_layers)
        ])

        # 2. w2v 레이어 통합 후 128으로 축소 (6 x 128 -> 128)
        self.w2v_reduce = nn.Sequential(
            nn.Conv1d(in_channels=num_layers * hidden_dim, out_channels=hidden_dim, kernel_size=1),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout)
        )

        # 3. os (25 -> 128)
        self.os_proj = nn.Sequential(
            nn.Conv1d(in_channels=os_dim, out_channels=hidden_dim, kernel_size=1),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout)
        )

        # 4. 최종 합치고 Conv1D → classifier
        self.final_conv = nn.Sequential(
            nn.Conv1d(in_channels=2 * hidden_dim, out_channels=hidden_dim, kernel_size=1),
            nn.ReLU()
        )

        self.classifier = nn.Linear(hidden_dim, num_classes)

    def forward(self, w2v_layers, os_input):
        # w2v_layers: (B, 6, T, 768) → 각 레이어 conv 후 concat → (B, 6*128, T)
        conv_outputs = []
        for i in range(w2v_layers.shape[1]):
            x = w2v_layers[:, i].permute(0, 2, 1)  # (B, 768, T)
            x = self.w2v_layer_convs[i](x)         # (B, 128, T)
            conv_outputs.append(x)

        w2v_concat = torch.cat(conv_outputs, dim=1)  # (B, 6*128, T)
        w2v_feat = self.w2v_reduce(w2v_concat)       # (B, 128, T)

        os_feat = self.os_proj(os_input.permute(0, 2, 1))  # (B, 128, T)

        merged = torch.cat([w2v_feat, os_feat], dim=1)     # (B, 256, T)
        merged = self.final_conv(merged)                   # (B, 128, T)

        pooled = merged.mean(dim=2)  # 평균 풀링 (B, 128)
        return self.classifier(pooled)  # (B, num_classes)



# ----- 모델 및 프로세서 불러오기 -----
# (wav2vec2, FusionModel)
wav2vec_model = Wav2Vec2Model.from_pretrained("Kkonjeong/wav2vec2-base-korean", output_hidden_states=True).to("cuda")
processor = Wav2Vec2Processor.from_pretrained("Kkonjeong/wav2vec2-base-korean")

# FusionModel 정의 (당신의 모델 코드 그대로 복사/사용)
# ... (여기에 FusionModel 정의 넣기)

# 모델 로드
model = FusionModel(
    w2v_dim=768,
    os_dim=25,
    hidden_dim=128,
    num_classes=4,
    dropout=0.2
).to("cuda")

if getattr(sys, 'frozen', False):
    # PyInstaller로 패킹된 실행파일인 경우
    base_path = sys._MEIPASS
else:
    # 개발 중일 때
    base_path = os.path.abspath(".")

model_path = os.path.join(base_path, "epoch20_final.pth")
checkpoint = torch.load(model_path, map_location="cuda")

model.load_state_dict(checkpoint["model_state_dict"])
model.eval()


# ----- 오디오 예측 -----
def predict_emotion(wav_path):
    y, sr = librosa.load(wav_path, sr=16000, mono=True)
    
    # Wav2Vec2 특징 추출
    w2v = extract_wav2vec_embedding(y, wav2vec_model, processor)  # (6, T, 768)

    # OpenSMILE 특징 추출
    os_ = extract_egemaps_feature(y, sr=16000)  # (T, 25)

    # Padding
    T = max(w2v.shape[1], os_.shape[0])
    w2v_tensor = torch.zeros(1, 6, T, 768).to("cuda")
    os_tensor = torch.zeros(1, T, 25).to("cuda")
    w2v_tensor[0, :, :w2v.shape[1], :] = w2v
    os_tensor[0, :os_.shape[0], :] = os_

    # 예측
    with torch.no_grad():
        logits = model(w2v_tensor, os_tensor)
        probs = torch.softmax(logits, dim=1).squeeze(0).cpu().numpy()

    label_map = {
        "angry": "분노",
        "happiness": "행복",
        "sadness": "슬픔",
        "neutral": "중립"
    }
    emotion_list = ['angry', 'happiness', 'sadness', 'neutral']
    pred_label = np.argmax(probs)
    pred_emotion_en = emotion_list[pred_label]
    pred_emotion_kr = label_map[pred_emotion_en]

    return pred_emotion_kr, probs

import os
import glob
import json

# frontend 에서 이를 바탕으로 할 꺼임
def process_folder(input_dir, output_json_path):
    """
    input_dir: .wav 파일들이 들어 있는 디렉토리 경로
    output_json_path: 결과를 저장할 JSON 파일 경로
    """
    results = {}  # { filename: { "filename": filename, "emotion": emotion } }

    # input_dir 내 .wav 파일 모두 찾기
    pattern = os.path.join(input_dir, "*.wav")
    wav_list = sorted(glob.glob(pattern))

    for wav_path in wav_list:
        filename = os.path.splitext(os.path.basename(wav_path))[0]

        # predict_emotion 함수 호출
        emotion, probs = predict_emotion(wav_path)

        # 결과 딕셔너리에 저장 (기존처럼)
        results[filename] = {
            "filename": filename,
            "emotion": emotion
        }
        #print(f"[Processed] {filename}.wav → {emotion}")

    # 딕셔너리 values()만 뽑아서 리스트 형태로 변환
    audio_list = list(results.values())

    # JSON 파일로 저장
    with open(output_json_path, "w", encoding="utf-8") as fout:
        json.dump(audio_list, fout, ensure_ascii=False, indent=2)

    print(f"\n결과를 '{output_json_path}'에 저장했습니다.")

