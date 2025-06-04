import os
import json
import argparse
from PIL import Image, ImageDraw

def visualize_crops(json_path, images_dir, output_dir, annotate=False):
    with open(json_path, 'r', encoding='utf-8') as f:
        records = json.load(f)

    crops_dir = os.path.join(output_dir, 'crops')
    os.makedirs(crops_dir, exist_ok=True)
    if annotate:
        ann_dir = os.path.join(output_dir, 'annotated')
        os.makedirs(ann_dir, exist_ok=True)

    # group records by capture so we annotate each image only once
    groups = {}
    for rec in records:
        cap = rec['capture']
        groups.setdefault(cap, []).append(rec)

    for cap, recs in groups.items():
        # 원본 이미지 파일(.png, .jpg 등) 자동 찾기
        for ext in ('png','jpg','jpeg','bmp'):
            img_path = os.path.join(images_dir, f"{cap}.{ext}")
            if os.path.exists(img_path):
                break
        else:
            print(f"[!] image for capture {cap} not found, skipping")
            continue

        img = Image.open(img_path).convert('RGB')
        if annotate:
            draw = ImageDraw.Draw(img)

        for i, rec in enumerate(recs):
            x1, y1, x2, y2 = rec['bbox']
            crop = img.crop((x1, y1, x2, y2))
            # 저장
            crop_name = f"{cap}_{rec.get('line', '')}_{rec.get('char','')}_{i:02d}.png"
            crop.save(os.path.join(crops_dir, crop_name))

            if annotate:
                # 빨간 박스 그리기
                draw.rectangle([x1, y1, x2, y2], outline='red', width=2)
                # 인덱스나 텍스트도 표시 가능
                draw.text((x1, y1-10), rec.get('text',''), fill='red')

        if annotate:
            img.save(os.path.join(ann_dir, f"{cap}_annotated.png"))

    print("Done. crops ->", crops_dir, ("annotated -> " + ann_dir) if annotate else "")

if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--json',      required=True, help='pipeline 결과 JSON 파일')
    p.add_argument('--images_dir',required=True, help='원본 이미지들이 있는 폴더')
    p.add_argument('--output',    default='output', help='결과 저장 폴더')
    p.add_argument('--annotate',  action='store_true', help='원본 이미지에 bbox 그려서 저장')
    args = p.parse_args()

    visualize_crops(args.json, args.images_dir, args.output, args.annotate)