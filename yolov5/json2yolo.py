import os
import json
from PIL import Image

def clamp(v, lo=0.0, hi=1.0):
    return min(max(v, lo), hi)

def convert_bbox_to_yolo(img_w, img_h, left, top, right, bottom):
    w = right - left
    h = bottom - top
    xc = left + w/2
    yc = top  + h/2

    xcn = xc / img_w
    ycn = yc / img_h
    wn  = w  / img_w
    hn  = h  / img_h

    return xcn, ycn, wn, hn


def create_yolo_labels(json_path, images_dir, output_dir):
    os.makedirs(output_dir, exist_ok=True)

    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    for img_name, info in data.items():
        img_path = os.path.join(images_dir, img_name)
        if not os.path.exists(img_path):
            print(f"[Warning] {img_path} 없음")
            continue

        with Image.open(img_path) as img:
            w_img, h_img = img.size

        txt_path = os.path.join(output_dir, os.path.splitext(img_name)[0] + ".txt")
        with open(txt_path, 'w', encoding='utf-8') as fw:
            for line in info.get("lines", []):
                l, t, r, b = line["bbox"]
                xcn, ycn, wn, hn = convert_bbox_to_yolo(w_img, h_img, l, t, r, b)

                # 유효 영역 필터링
                if wn <= 0 or hn <= 0 or wn > 1 or hn > 1:
                    continue

                # 0~1 범위로 클램핑 (필요시)
                xcn, ycn, wn, hn = map(clamp, (xcn, ycn, wn, hn))

                class_id = 0
                fw.write(f"{class_id} {xcn:.6f} {ycn:.6f} {wn:.6f} {hn:.6f}\n")


if __name__ == "__main__":
    # 실제 경로에 맞게 아래 값들을 수정해 주세요.
    JSON_PATH = "C:/Users/LEE/Desktop/dataset/multiline_dataset/train_annotations.json"  # 예: "./data/annotations.json"
    IMAGES_DIR = "C:/Users/LEE/Desktop/dataset/multiline_dataset/images/train"  # 예: "./data/images"
    OUTPUT_DIR = "C:/Users/LEE/Desktop/dataset/multiline_dataset/labels/train"  # 예: "./data/labels"

    create_yolo_labels(JSON_PATH, IMAGES_DIR, OUTPUT_DIR)
