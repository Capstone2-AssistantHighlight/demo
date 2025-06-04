import os
import json

def create_gt_file_from_json(json_path, images_dir, output_gt_file, delimiter="\t"):
    """
    json_path:  예: synthetic_detection_data/train_annotations.json
    images_dir: 예: synthetic_detection_data/train
    output_gt_file: 예: synthetic_detection_data/train_gt.txt
    """
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    with open(output_gt_file, 'w', encoding='utf-8') as fout:
        for img_name, info in data.items():
            # 1) 이미지 경로
            # image_path = os.path.join(images_dir, img_name)
            image_path = img_name
            # 2) lines 리스트에서 text만 꺼내서 합치기
            lines = info.get("lines", [])
            # 한 줄짜리면 lines[0]["text"] 만 쓰면 되고,
            # 멀티라인이면 공백으로 join 해도 됩니다.
            texts = [ line.get("text","").strip() for line in lines ]
            text = " ".join(texts)

            fout.write(f"{image_path}{delimiter}{text}\n")

    print(f"GT 파일이 생성되었습니다: {output_gt_file}")


if __name__ == '__main__':
    json_path = "C:/Users/LEE/Desktop/deep/deep4_dataset/val_annotations.json"
    images_dir = "C:/Users/LEE/Desktop/deep/deep4_dataset/images/val"
    output_gt_file = "C:/Users/LEE/Desktop/deep/deep4_dataset/val_gt.txt"
    create_gt_file_from_json(json_path, images_dir, output_gt_file)
