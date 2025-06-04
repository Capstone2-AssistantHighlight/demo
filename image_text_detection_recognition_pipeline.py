import os
import glob
import json
import argparse
import sys
import yaml, numpy as np
from tqdm import tqdm
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

# Recognition에 필요한 charset loader 함수

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



def recognize_image(crop_pil, rec_model, converter, transform, args, device):
    # 모델 학습 시 rgb 옵션에 맞춰 채널 변환
    if args.rgb:
        crop_pil = crop_pil.convert('RGB')
    else:
        crop_pil = crop_pil.convert('L')  # grayscale

    # transform expects batch of (image, label)
    tensor, _ = transform([(crop_pil, '')])
    image = tensor.to(device).float()
    bs = image.size(0)
    text_input = torch.LongTensor(bs, args.batch_max_length + 1).fill_(0).to(device)

    with torch.no_grad():
        if 'CTC' in args.Prediction:
            preds = rec_model(image, text_input)
            sizes = torch.IntTensor([preds.size(1)] * bs)
            _, idx = preds.max(2)
            texts = converter.decode(idx, sizes)
            probs = preds.log_softmax(2).exp()
            max_probs, _ = probs.max(dim=2)
            confs = [p.cumprod(dim=0)[-1].item() for p in max_probs]
        else:
            preds = rec_model(image, text_input, is_train=False)
            _, idx = preds.max(2)
            texts = converter.decode(idx, torch.IntTensor([args.batch_max_length] * bs))
            text = texts[0]
            if '[s]' in text:
                eos = text.find('[s]')
                text = text[:eos]

            probs = torch.softmax(preds, dim=2)  # [bs, T, C]
            max_probs, _ = probs.max(dim=2)  # [bs, T]
            cumprod = max_probs.cumprod(dim=1)  # [bs, T] ← dim=1: 시간축
            conf = cumprod[0, -1].item()  # 첫 배치(first sample)의 마지막 time-step

    return text, conf


def process_image_with_precomputed(image_path, detections_line: torch.Tensor,
                                   char_model, rec_model, converter, transform,
                                   args, device):
    capture_id = os.path.splitext(os.path.basename(image_path))[0]
    img_bgr = cv2.imread(image_path)
    if img_bgr is None:
        return []
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    H, W = img_rgb.shape[:2]
    pad = 5
    records = []

    # 1) 모든 라인 크롭 배치
    line_crops = []
    line_meta  = []
    for line_idx, det in enumerate(detections_line.tolist()):
        x1, y1, x2, y2, det_conf, _ = det
        if det_conf < args.line_conf:
            continue
        px1, py1 = max(int(x1)-pad,0), max(int(y1)-pad,0)
        px2, py2 = min(int(x2)+pad,W), min(int(y2)+pad,H)
        line_crops.append(img_rgb[py1:py2, px1:px2])
        line_meta.append((line_idx, px1, py1))

    if not line_crops:
        return []

    # 2) char detection 배치
    char_results = char_model(line_crops)            # Detections 리스트
    # 3) 모든 문자 크롭 준비
    char_crops = []
    char_meta  = []
    for (line_idx, px1, py1), char_det in zip(line_meta, char_results.xyxy):
        for char_idx, cd in enumerate(char_det.tolist()):
            cx1, cy1, cx2, cy2, cconf, _ = cd
            if cconf < args.char_conf or cconf < 0.4:
                continue
            ax1, ay1 = px1+int(cx1), py1+int(cy1)
            ax2, ay2 = px1+int(cx2), py1+int(cy2)
            # → PIL로 변환 후, 그레이스케일(1채널) 혹은 RGB(3채널)로 맞춰줌
            crop = Image.fromarray(img_rgb[ay1:ay2, ax1:ax2])
            if args.rgb:
                crop = crop.convert('RGB')
            else:
                crop = crop.convert('L')
            char_crops.append(crop)
            char_meta.append((line_idx, char_idx, ax1, ay1))

    # 4) recognition 배치
    texts = []
    if char_crops:
        batch = [(crop, '') for crop in char_crops]
        tensor, _ = transform(batch)             # [N, C, H, W]
        image_batch = tensor.to(device).float()
        bs = image_batch.size(0)
        text_input = torch.LongTensor(bs, args.batch_max_length+1).fill_(0).to(device)

        from torch.amp import autocast
        with autocast(device_type='cuda'):
            if 'CTC' in args.Prediction:
                preds = rec_model(image_batch, text_input)
                sizes = torch.IntTensor([preds.size(1)]*bs)
                _, idx = preds.max(2)
                texts = converter.decode(idx, sizes)
            else:
                preds = rec_model(image_batch, text_input, is_train=False)
                _, idx = preds.max(2)
                texts = converter.decode(idx, torch.IntTensor([args.batch_max_length]*bs))
                # '[s]' 이후 제거
                texts = [t.split('[s]')[0] if '[s]' in t else t for t in texts]

    # 5) 결과 조립
    for (line_idx, char_idx, ax1, ay1), text in zip(char_meta, texts):
        records.append({
            'capture': capture_id,
            'line':    f'L{line_idx}',
            'char':    f'C{char_idx}',
            'bbox':    [ax1, ay1, ax1, ay1],  # bbox를 정확히 쓰고 싶으면 별도 저장
            'text':    text
        })

    return records

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


def run_pipeline(capture_dir,output_pipeline):
    defaults = {
        'rgb': False,
        'line_det_weights': 'yolov5/runs/train/detect_line3/weights/best.pt',
        'char_det_weights': 'yolov5/runs/train/detect_text/weights/best.pt',
        'rec_weights': 'dtrb/saved_models/text_nn_rec/best_accuracy.pth',
        'character': 'dtrb/charset.txt', 'input_channel': 1, 'output_channel':512, 'hidden_size':256,
        'imgH': 32, 'imgW': 100, 'batch_max_length': 25,
        'PAD': True, 'line_conf': 0.25, 'char_conf': 0.25,
        'workers': 4, 'device': 'cuda:0', 'batch_size': 8,
        'Prediction': 'Attn', 'Transformation':'None', 'FeatureExtraction':'ResNet', 'SequenceModeling':'None', 'num_fiducial':20,
    }
    defaults['character'] = ''.join(load_charset(defaults['character']))

    args_dict = defaults.copy()
    args_dict['capture_dir'] = capture_dir
    args_dict['output'] = output_pipeline
    args = SimpleNamespace(**args_dict)

    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')

    # Detection 모델 로드 (torch.hub local)
    # 1) 라인 검출 모델
    line_det = torch.hub.load(yolov5_dir, 'custom', path=args.line_det_weights, source='local')
    line_det.to(device).eval()
    line_det.half()
    inject_anchors_from_yaml(line_det, yolov5_dir, 'yolov5s', device)

    # 2) 문자(글자) 검출 모델
    char_det = torch.hub.load(yolov5_dir, 'custom', path=args.char_det_weights, source='local')
    char_det.to(device).eval()
    char_det.half()
    inject_anchors_from_yaml(char_det, yolov5_dir, 'yolov5s', device)


    rec_model, converter, transform = load_recognition_model(args, device)

    image_paths = sorted(glob.glob(os.path.join(args.capture_dir, '*.*')))
    all_records = []

    BATCH_SIZE = 8
    pbar = tqdm(total=len(image_paths), desc="Images processed")
    from torch.amp import autocast

    for i in range(0, len(image_paths), BATCH_SIZE):
        batch_paths = image_paths[i:i + BATCH_SIZE]
        imgs = [cv2.cvtColor(cv2.imread(p), cv2.COLOR_BGR2RGB)
                for p in batch_paths]

        with autocast(device_type='cuda'):
            batch_results = line_det(imgs)

        # batch_results.xyxy: list of [n_det,6] tensors
        for img_path, det_tensor in zip(batch_paths, batch_results.xyxy):
            recs = process_image_with_precomputed(
                img_path, det_tensor, char_det,
                rec_model, converter, transform,
                args, device
            )
            all_records.extend(recs)
            pbar.update(1)
    pbar.close()

    from collections import defaultdict
    output_data = []

    # 1) capture별로 레코드 묶기
    records_by_capture = defaultdict(list)
    for rec in all_records:
        records_by_capture[rec['capture']].append(rec)

    # 2) 각 capture에 대해
    for cap, recs in sorted(records_by_capture.items(), key=lambda x: int(x[0]) if x[0].isdigit() else x[0]):
        # 2-1) 원래 라벨(L0, L1…)별로 묶기
        lines = defaultdict(list)
        for r in recs:
            lines[r['line']].append(r)

        # 2-2) 라인 그룹을 y축(최소 y1) 기준으로 정렬
        sorted_lines = sorted(
            lines.items(),
            key=lambda item: min(rec['bbox'][1] for rec in item[1])
        )

        # 2-3) 정렬된 라인 순서대로, 각 라인 내 문자들을 x축 기준으로 정렬 후 텍스트 합치기
        for line_label, chars in sorted_lines:
            chars.sort(key=lambda r: r['bbox'][0])
            line_text = ''.join(r['text'] for r in chars)

            output_data.append({
                'capture': cap,
                'line': line_label,
                'text': line_text
            })

    filtered = remove_redundant_lines(output_data, thresh=0.9)


    # 3) JSON 저장
    with open(args.output, 'w', encoding='utf-8') as f:
        json.dump(filtered, f, ensure_ascii=False, indent=2)
    print(f'Results saved to {args.output}')


if __name__ == '__main__':
    run_pipeline('frame','pipeline_result.json')
