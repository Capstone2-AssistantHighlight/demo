import argparse
import glob
import json
import os
from concurrent.futures import ThreadPoolExecutor

from collections import defaultdict, Counter

import torch

from kobert_tokenizer import KoBERTTokenizer
from transformers import BertModel
from model import KoBERTClassifier

# -------------------------------------------------------------------
# 1) Detection+Recognition 파이프라인
# -------------------------------------------------------------------
from image_text_thread_pipeline import run_pipeline
# run_pipeline(capture_dir, output_json_path) 형태

# SER_converted.py 안의 process_folder 함수를 가져옵니다.
from SER_converted import process_folder
# -------------------------------------------------------------------
# 2) 속도 랭킹 함수 (lines_per_window 기준)
# -------------------------------------------------------------------
def compute_speed_ranking(pipeline_json, out_path, window_size=5):
    from collections import defaultdict

    def parse_timestamp(ts):
        h, m, s = map(int, ts.split('-'))
        return h*3600 + m*60 + s

    def fmt_window(start):
        end = start + window_size - 1
        def f(x): return f"{x//3600:02d}-{(x%3600)//60:02d}-{x%60:02d}"
        return f"{f(start)}~{f(end)}"

    def fmt_time(sec):
        h = sec // 3600
        m = (sec % 3600) // 60
        s = sec % 60
        return f"{h:02d}-{m:02d}-{s:02d}"

    # load
    with open(pipeline_json, encoding='utf-8') as f:
        records = json.load(f)

    # per-second line counts
    per_sec = defaultdict(int)
    for r in records:
        per_sec[ parse_timestamp(r['capture']) ] += 1

    # window sums
    all_secs = sorted(per_sec)
    if not all_secs:
        return
    t0, t1 = all_secs[0], all_secs[-1]
    stats = []
    t = t0
    while t <= t1:
        cnt = sum(per_sec.get(s, 0) for s in range(t, t+window_size))
        stats.append({
            'start_time': fmt_time(t),
            'end_time': fmt_time(t + window_size - 1),
            'line_count': cnt
        })
        t += window_size

    # rank & save
    stats.sort(key=lambda x: x['line_count'], reverse=True)
    for i, w in enumerate(stats, 1):
        w['rank'] = i

    with open(out_path, 'w', encoding='utf-8') as f:
        json.dump(stats, f, ensure_ascii=False, indent=2)
    print(f"윈도우별 속도 결과가 '{out_path}' 에 저장되었습니다.")

# -------------------------------------------------------------------
# 3) 감정 분석 함수 (라인별, 50% 보유 시 대표 감정, else 중립)
# -------------------------------------------------------------------

def load_trained_kobert(model_dir: str, device: torch.device):

    tokenizer = KoBERTTokenizer.from_pretrained(model_dir)
    bert_model = BertModel.from_pretrained(model_dir, use_safetensors=True, return_dict=False)

    num_classes = 7
    dr_rate = 0.5
    classifier = KoBERTClassifier(bert_model=bert_model, dr_rate=dr_rate, num_classes=num_classes)

    #state_dict = torch.load(f"{model_dir}/best_kobert_epoch2.pt", map_location=device)

    classifier_head_path = os.path.join(model_dir, "classifier_head.pt")
    classifier.classifier.load_state_dict(torch.load(classifier_head_path, map_location=device))

    classifier.to(device)
    classifier.eval()

    id2label = {
        0: "공포",
        1: "놀람",
        2: "분노",
        3: "슬픔",
        4: "중립",
        5: "행복",
        6: "혐오"
    }

    return tokenizer, classifier, id2label

def predict_text_list(texts: list, tokenizer, model, id2label, device: torch.device):
    encodings = tokenizer(
        texts,
        padding='max_length',
        truncation=True,
        max_length=128,
        return_tensors='pt'
    ).to(device)

    input_ids = encodings['input_ids']
    attention_mask = encodings['attention_mask']
    token_type_ids = encodings['token_type_ids']

    # valid_length 계산 (각 시퀀스별 실제 토큰 개수)
    valid_length = attention_mask.sum(dim=1)

    with torch.no_grad():
        logits = model(input_ids, valid_length, token_type_ids)
        pred_indices = logits.argmax(dim=-1).tolist()

    return [id2label[i] for i in pred_indices]


def predict_text_list_with_heuristic(texts: list, tokenizer, model, id2label: dict, device: torch.device) -> list:
    final_labels = []
    for txt in texts:
        # 1) 휴리스틱 검사
        heuristic_label = override_emotion_by_heuristic(txt)
        if heuristic_label is not None:
            final_labels.append(heuristic_label)
            continue  # 모델 예측 없이 강제 적용 :contentReference[oaicite:13]{index=13}

        # 2) 모델 예측: input_ids, valid_length, token_type_ids 준비
        encodings = tokenizer(
            [txt],
            padding='max_length',
            truncation=True,
            max_length=128,
            return_tensors='pt'
        ).to(device)

        input_ids = encodings['input_ids']
        attention_mask = encodings['attention_mask']
        token_type_ids = encodings['token_type_ids']
        valid_length = attention_mask.sum(dim=1)  # :contentReference[oaicite:14]{index=14}

        with torch.no_grad():
            logits = model(input_ids, valid_length, token_type_ids)
            # 3) logits에서 top2 인덱스 추출
            top2 = logits.topk(2, dim=-1).indices.squeeze(0).tolist()

        # 4) 만약 top1이 '공포' 또는 '혐오'라면 두 번째로 높은 label 사용
        top1_idx = top2[0]
        if id2label[top1_idx] in ("공포", "혐오", "분노"):  # fear, disgust
            # 두 번째 인덱스를 사용
            second_idx = top2[1]
            final_labels.append(id2label[second_idx])
        else:
            final_labels.append(id2label[top1_idx])

    return final_labels

def override_emotion_by_heuristic(txt: str) -> str:
    # 1) 'ㅋ'가 세 번 이상이면 '행복'
    if txt.count('ㅋ') >= 3:
        return "행복"  # happy :contentReference[oaicite:8]{index=8}
    # 2) '?'로 끝나면 '놀람'
    if txt.endswith('?'):
        return "놀람"  # surprise :contentReference[oaicite:9]{index=9}
    # 3) 'ㄷ'이 두 번 이상이면 '놀람'
    if txt.count('ㄷ') >= 2:
        return "놀람"  # surprise :contentReference[oaicite:10]{index=10}
    # 4) 'ㅠ'가 두 번 이상이면 '슬픔'
    if txt.count('ㅠ') >= 2:
        return "슬픔"  # sadness :contentReference[oaicite:11]{index=11}
    # 어느 조건에도 해당하지 않으면 None 반환 (모델 예측 사용)
    return None

def compute_emotion_by_line(pipeline_json, out_path, kobert_dir, window_size=5):
    # 1) load pipeline results
    with open(pipeline_json, encoding='utf-8') as f:
        records = json.load(f)

    # 2) load trained KoBERT model & tokenizer
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer, model, id2label = load_trained_kobert(kobert_dir, device)

    # 3) 개별 라인 감정 예측 (텍스트가 많으면 배치 단위로 나누어 처리 가능)
    #print("=== 개별 라인 감정 예측 시작 ===")
    #texts = [r['text'] for r in records]
    #labels = predict_text_list_with_heuristic(texts, tokenizer, model, id2label, device)
    #for entry, lbl in zip(records, labels):
    #    print(f"{entry['capture']} | {entry.get('line', '-')} | \"{entry['text']}\" ➔ {lbl}")
    #print("=== 개별 라인 감정 예측 종료 ===\n")

    # 4) 윈도우별 묶음 및 대표 감정 예측
    def parse_ts(ts):
        h, m, s = map(int, ts.split('-')); return h*3600 + m*60 + s
    def fmt_time(sec):
        h = sec // 3600; m = (sec % 3600)//60; s = sec % 60
        return f"{h:02d}-{m:02d}-{s:02d}"

    buckets = defaultdict(list)
    for entry in records:
        sec = parse_ts(entry['capture'])
        win_start = (sec // window_size) * window_size
        bucket_key = win_start
        buckets[bucket_key].append(entry['text'])

    results = []
    for win_start, text_list in buckets.items():
        # 윈도우 내 텍스트 전체 배치 예측
        preds = predict_text_list_with_heuristic(text_list, tokenizer, model, id2label, device)
        cnt = Counter(preds)
        top_emotion, top_count = cnt.most_common(1)[0]
        rep = top_emotion if top_count / len(preds) >= 0.4 else "중립"
        start_str = fmt_time(win_start)
        end_str = fmt_time(win_start + window_size - 1)
        percent = round(top_count / len(preds) * 100, 2)
        results.append({
            "start_time": start_str,
            "end_time": end_str,
            "emotion": rep,
            "emotion_percent": percent
        })

    # 5) 결과 저장
    with open(out_path, 'w', encoding='utf-8') as fout:
        json.dump(results, fout, ensure_ascii=False, indent=2)
    print(f"윈도우별 감정 결과가 '{out_path}' 에 저장되었습니다.")
# -------------------------------------------------------------------
# 4) Orchestrator: capture_dir 만 인자로 받아, 3단계 실행
# -------------------------------------------------------------------
def full_pipeline(capture_dir, wav_dir):

    base_name = os.path.basename(capture_dir).replace("_frames", "")
    output_dir = f"{base_name}_output"
    os.makedirs(output_dir, exist_ok=True)

    # 1) 텍스트 파이프라인
    pipeline_json = os.path.join(output_dir, 'pipeline_result1.json')
    run_pipeline(capture_dir, pipeline_json)
    # 2) speed & emotion 병렬 실행
    with ThreadPoolExecutor(max_workers=3) as exe:
        # 2-1) 속도 랭킹
        speed_future = exe.submit(
            compute_speed_ranking,
            pipeline_json,
            os.path.join(output_dir, "speed_rankings.json")
        )

        # 2-2) 텍스트 단위 감정 분석 (KoBERT)
        emo_text_future = exe.submit(
            compute_emotion_by_line,
            pipeline_json,
            os.path.join(output_dir, "emotion_by_line.json"),
            'best_full_kobert_epoch2'  # KoBERT 모델 디렉토리
        )

        # 2-3) 오디오 기반 감정 분석
        #    process_folder(input_dir, output_json_path)
        audio_future = exe.submit(
            process_folder,
            wav_dir,
            os.path.join(output_dir, "audio_emotion.json")
        )
        import traceback

        try:
            speed_future.result()
            emo_text_future.result()
            audio_future.result()

        except Exception as e:
            print("스레드 실행 중 예외 발생:", e)
            traceback.print_exc()

    print("✅ All done. See outputs/")

def main():
    p = argparse.ArgumentParser()
    p.add_argument('--capture_dir', help='이미지 폴더 경로')
    args = p.parse_args()
    output_dir = f"{args.capture_dir}_output"
    cap_dir = args.capture_dir
    os.makedirs(output_dir, exist_ok=True)

    # 1) 텍스트 파이프라인
    pipeline_json = os.path.join(output_dir,'pipeline_result1.json')
    run_pipeline(cap_dir, pipeline_json)

    # 2) speed & emotion 병렬 실행
    with ThreadPoolExecutor(max_workers=2) as exe:
        future1 = exe.submit(
            compute_speed_ranking,
            pipeline_json,
            os.path.join(output_dir,'speed_rankings.json')
        )
        future2 = exe.submit(
            compute_emotion_by_line,
            pipeline_json,
            os.path.join(output_dir,'emotion_by_line.json'),
            'best_full_kobert_epoch2'
        )
        import traceback

        try:
            future1.result()
            future2.result()
        except Exception as e:
            print("스레드 실행 중 예외 발생:", e)
            traceback.print_exc()

    print("✅ All done. See outputs/")

if __name__=='__main__':
    main()
