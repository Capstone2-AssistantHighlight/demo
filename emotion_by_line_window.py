from collections import Counter
import torch, json
from transformers import BertForSequenceClassification, AutoTokenizer

# 1) 모델·토크나이저 로드
device    = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model_dir = "./kobert_emotion1_final"
tokenizer = AutoTokenizer.from_pretrained(model_dir, trust_remote_code=True)
model     = BertForSequenceClassification.from_pretrained(model_dir, trust_remote_code=True)
model.to(device).eval()

id2label = {0:"분노", 1:"슬픔", 2:"중립", 3:"행복", 4:"공포"}

# 2) pipeline_results.json 로드
with open("pipeline_results.json", encoding="utf-8") as f:
    records = json.load(f)

# 3) 윈도우별로 라인 텍스트 묶기 (기존 window_texts 대신)
from collections import defaultdict
def parse_timestamp(ts):
    h,m,s = map(int, ts.split('-'))
    return h*3600 + m*60 + s

def format_window(start, w=5):
    end = start + w - 1
    fmt = lambda x: f"{x//3600:02d}-{(x%3600)//60:02d}-{x%60:02d}"
    return f"{fmt(start)}~{fmt(end)}"

window_lines = defaultdict(list)
for r in records:
    sec = parse_timestamp(r["capture"])
    win = format_window(sec, 5)
    window_lines[win].append(r["text"])

# 4) 윈도우별 감정 집계
emotion_results = []
for win, texts in window_lines.items():
    preds = []
    for text in texts:
        inputs = tokenizer(
            text,
            padding="max_length",
            truncation=True,
            max_length=128,
            return_tensors="pt"
        ).to(device)
        with torch.no_grad():
            logits = model(**inputs).logits
            pred  = torch.argmax(logits, dim=-1).item()
        preds.append(id2label[pred])

    total = len(preds)
    cnt   = Counter(preds)
    emo, cnt_max = cnt.most_common(1)[0]

    # 50% 이상인지 체크
    if cnt_max / total >= 0.5:
        selected = emo
    else:
        selected = "중립"

    emotion_results.append({
        "window":  win,
        "emotion": selected,
        "counts":  dict(cnt)   # 필요 없으면 이 줄은 제거하세요
    })

# 5) 저장
with open("emotion_by_line_results.json", "w", encoding="utf-8") as f:
    json.dump(emotion_results, f, ensure_ascii=False, indent=2)