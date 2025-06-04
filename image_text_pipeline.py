import os
import glob
import json
import sys
import torch
import yaml
import numpy as np
from tqdm import tqdm
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from types import SimpleNamespace

# 로컬 YOLOv5 및 DTRB 리포지토리 경로 추가
base_dir = os.path.dirname(os.path.abspath(__file__))
yolov5_dir = 'yolov5'
dtrb_dir = 'dtrb'
sys.path.insert(0, yolov5_dir)
sys.path.insert(0, dtrb_dir)
# forked repo 체크 비활성화
import torch
torch.hub._validate_not_a_forked_repo = lambda a, b, c: True

import cv2
from PIL import Image
# DTRB import (로컬) from deep-text-recognition-benchmark-master
from dtrb.utils import CTCLabelConverter, AttnLabelConverter
from dtrb.dataset import AlignCollate
from dtrb.model import Model

# -- (기존 import 생략) --
def load_charset(path_or_str):
    """
       path_or_str: 파일 경로, 문자열, 리스트 허용
       파일 경로일 경우 파일에서 읽어 문자 리스트 반환
       문자열일 경우 각 문자로 분할하여 리스트 반환
       리스트/튜플일 경우 그대로 반환
       """
    if isinstance(path_or_str, (list, tuple)):
        return list(path_or_str)
    try:
        if os.path.isfile(path_or_str):
            with open(path_or_str, 'r', encoding='utf-8') as f:
                chars = f.read().strip()
            return list(chars)
    except Exception:
        pass
    return list(str(path_or_str))

def load_recognition_model(args, device):
    # 1) 체크포인트 불러오기 (prefix stripping 전)
    raw_state = torch.load(args.rec_weights, map_location=device)
    # 2) 첫 번째 conv weight key 찾기
    ckpt_key = next(k for k in raw_state if 'conv0_1.weight' in k)
    ckpt_in_ch = raw_state[ckpt_key].shape[1]  # 1 혹은 3
    args.input_channel = ckpt_in_ch

    # 3) 라벨 컨버터 초기화
    if 'CTC' in args.Prediction:
        converter = CTCLabelConverter(args.character)
    else:
        converter = AttnLabelConverter(args.character)
    args.num_class = len(converter.character)

    # 4) 모델 생성
    rec_model = Model(args)

    # 5) module. prefix 제거 & 로드
    from collections import OrderedDict
    new_state = OrderedDict()
    for k, v in raw_state.items():
        name = k[7:] if k.startswith('module.') else k
        new_state[name] = v
    rec_model.load_state_dict(new_state)

    # 6) DataParallel wrapping
    rec_model = torch.nn.DataParallel(rec_model).to(device).eval()
    # rec_model.half()

    # 7) Transform 세팅
    transform = AlignCollate(imgH=args.imgH, imgW=args.imgW, keep_ratio_with_pad=args.PAD)
    return rec_model, converter, transform

def inject_anchors_from_yaml(model, yolov5_dir, cfg_name, device):
    """
    model      : torch.hub.load(..., autoshape=False) 으로 받은 DetectionModel
    yolov5_dir : yolov5 코드 폴더 경로
    cfg_name   : 'yolov5s' 등 (models/ 밑에 있는 yaml 이름, 확장자 제외)
    device     : torch.device
    """
    # 1) yaml 읽기 (UTF-8로)
    cfg_path = os.path.join(yolov5_dir, 'models', f'{cfg_name}.yaml')
    with open(cfg_path, 'r', encoding='utf-8') as f:
        cfg = yaml.safe_load(f)

    # 2) anchors 배열 꺼내기: 리스트 of [xa,ya,xb,yb,...] 형태
    arr = np.array(cfg['anchors'], dtype=np.float32)  # shape (nl, na*2)
    # 3) Detect 레이어 내부에서 앵커를 담당하는 모듈 찾기
    detect_layer = None
    for m in model.modules():
        if hasattr(m, 'anchors') and hasattr(m, 'anchor_grid'):
            detect_layer = m
            break
    if detect_layer is None:
        print("⚠️ Detect 레이어를 찾을 수 없습니다. 앵커 변경 생략.")
        return

    # 4) tensor 로 변환하여 anchors 속성만 교체
    nl = detect_layer.nl  # detection 레이어 수
    na = detect_layer.na  # 앵커 per 레이어
    new_anchors = arr.reshape(nl, na, 2)
    anchors_t = torch.from_numpy(new_anchors).to(device)
    detect_layer.anchors = anchors_t

    print(f"🔧 {cfg_name}.yaml 앵커({nl}×{na})만 주입 완료")
    # (anchor_grid 버퍼는 건드리지 않아도 다음 forward 시 _make_grid로 재생성됩니다)

def group_chars_into_lines(records, y_thresh=20):
    """
    records: [{'bbox':[x1,y1,x2,y2], 'text':str, 'capture':id}, …]
    y_thresh: 같은 줄로 볼 최대 y 중심 차이(px)
    """
    # 1) 각 문자 레코드에 center_y 추가
    items = [
        {'rec': r,
         'cy': (r['bbox'][1] + r['bbox'][3]) / 2}
        for r in records
    ]
    # 2) y 기준 오름차순 정렬
    items.sort(key=lambda x: x['cy'])

    lines = []
    for it in items:
        placed = False
        for line in lines:
            # 기존 라인의 평균 center_y 와 비교
            if abs(it['cy'] - line['mean_cy']) <= y_thresh:
                line['recs'].append(it['rec'])
                # 평균 y 업데이트
                cy_list = [(r['bbox'][1] + r['bbox'][3]) / 2 for r in line['recs']]
                line['mean_cy'] = sum(cy_list) / len(cy_list)
                placed = True
                break
        if not placed:
            lines.append({
                'mean_cy': it['cy'],
                'recs': [it['rec']]
            })
    # 3) 각 줄별로 x 오름차순 정렬된 레코드 리스트만 반환
    grouped = []
    for line in lines:
        recs = sorted(line['recs'], key=lambda r: r['bbox'][0])
        grouped.append(recs)
    return grouped

from difflib import SequenceMatcher

def remove_redundant_lines(records, thresh=0.9):
    # 1) capture 이름 순 정렬 (HH-MM-SS 형식이라 사전순 정렬로 OK)
    caps = sorted({r['capture'] for r in records})
    cleaned = []
    prev_texts = None

    for cap in caps:
        # 2) 한 캡처 안의 라인들 L0, L1… 순서대로 뽑기
        lines = sorted(
            [r for r in records if r['capture']==cap],
            key=lambda r: int(r['line'][1:])
        )
        if prev_texts is None:
            # 첫 캡처는 그대로
            cleaned.extend(lines)
        else:
            # 3) 뒤에서부터 돌면서 이전 캡처 텍스트와 유사도 검사
            drop_idx = None
            for i in range(len(lines)-1, -1, -1):
                for pt in prev_texts:
                    ratio = SequenceMatcher(None, lines[i]['text'], pt).ratio()
                    if ratio >= thresh:
                        drop_idx = i
                        break
                if drop_idx is not None:
                    break

            # 4) 중복된 라인과 그 앞 라인 모두 버리고, 그 뒤 라인만 추가
            if drop_idx is None:
                cleaned.extend(lines)
            else:
                cleaned.extend(lines[drop_idx+1:])

        # 5) 이번 캡처의 텍스트 리스트를 prev_texts로 갱신
        prev_texts = [r['text'] for r in lines]

    return cleaned

def run_pipeline(capture_dir, output_path):
    # -- default args 세팅 (기존과 동일) --
    defaults = {
        'rgb': False,
        'char_det_weights': 'yolov5/runs/train/detect_text/weights/best.pt',
        'rec_weights': 'dtrb/saved_models/text_nn_rec/best_accuracy.pth',
        'character': 'dtrb/charset.txt', 'input_channel': 1, 'output_channel':512, 'hidden_size':256,
        'imgH': 32, 'imgW': 100, 'batch_max_length': 25,
        'PAD': True, 'char_conf': 0.25,
        'workers': 4, 'device': 'cuda:0', 'batch_size': 8,
        'Prediction': 'Attn', 'Transformation':'None', 'FeatureExtraction':'ResNet', 'SequenceModeling':'None', 'num_fiducial':20,
    }
    defaults['character'] = ''.join(load_charset(defaults['character']))
    args = SimpleNamespace(**{**defaults,
                             'capture_dir': capture_dir,
                             'output': output_path})

    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')

    # 문자 검출 모델 한 개만 로드
    text_det = torch.hub.load('yolov5', 'custom',
                              path=args.char_det_weights,
                              source='local').to(device).eval()
    text_det.half()

    rec_model, converter, transform = load_recognition_model(args, device)

    image_paths = sorted(glob.glob(os.path.join(capture_dir, '*.*')))
    all_outputs = []

    pbar = tqdm(total=len(image_paths), desc="Images processed")
    from torch.amp import autocast

    for img_path in image_paths:
        img_bgr = cv2.imread(img_path)
        if img_bgr is None:
            pbar.update(1)
            continue
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

        # 1) 문자 검출
        with autocast(device_type='cuda'):
            results = text_det([img_rgb])
        dets = results.xyxy[0].cpu().numpy()  # [ [x1,y1,x2,y2,conf,cls], … ]

        # 2) 각 문자에 대해 인식
        records = []
        capture_id = os.path.splitext(os.path.basename(img_path))[0]
        for x1, y1, x2, y2, conf, _ in dets:
            if conf < args.char_conf:
                continue

            crop = Image.fromarray(
                img_rgb[int(y1):int(y2), int(x1):int(x2)]
            )
            crop = crop.convert('RGB' if args.rgb else 'L')

            # recognition
            tensor, _ = transform([(crop, '')])
            image_batch = tensor.to(device).float()
            bs = image_batch.size(0)
            text_input = torch.LongTensor(
                bs, args.batch_max_length + 1
            ).fill_(0).to(device)

            with autocast(device_type='cuda'):
                if 'CTC' in args.Prediction:
                    preds = rec_model(image_batch, text_input)
                    sizes = torch.IntTensor([preds.size(1)] * bs)
                    _, idx = preds.max(2)
                    texts = converter.decode(idx, sizes)
                    text = texts[0]
                else:
                    preds = rec_model(image_batch, text_input, is_train=False)
                    _, idx = preds.max(2)
                    texts = converter.decode(idx,
                                             torch.IntTensor([args.batch_max_length] * bs))
                    text = texts[0].split('[s]')[0]

            records.append({
                'capture': capture_id,
                'bbox': [int(x1), int(y1), int(x2), int(y2)],
                'text': text
            })
        #bbox 위치 디버그
        #print("▶ raw records:", records)
        # 3) y축 기준으로 라인 그룹핑 → 줄별 텍스트 조합
        lines = group_chars_into_lines(records, y_thresh=30)
        #print("▶ grouped lines:", lines)
        for i, line_recs in enumerate(lines):
            #x1_min = min(r['bbox'][0] for r in line_recs)
            #y1_min = min(r['bbox'][1] for r in line_recs)
            #x2_max = max(r['bbox'][2] for r in line_recs)
            #y2_max = max(r['bbox'][3] for r in line_recs)
            line_text = ''.join(r['text'] for r in line_recs)
            all_outputs.append({
                'capture': capture_id,
                'line': f'L{i}',
                'text': line_text,
                #'bbox': [x1_min, y1_min, x2_max, y2_max]
            })

        pbar.update(1)

    filtered = remove_redundant_lines(all_outputs, thresh=0.8)
    pbar.close()

    # 4) 결과 JSON 저장
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(filtered, f, ensure_ascii=False, indent=2)
    print(f"Results saved to {output_path}")

if __name__ == '__main__':
    run_pipeline('exex', 'pipeline_result1.json')



