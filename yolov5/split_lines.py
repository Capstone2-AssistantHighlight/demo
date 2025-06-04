import os
import json
from PIL import Image

def split_lines_and_create_new_json(json_path, images_dir, output_images_dir, output_json_path):
    # 잘라낸 이미지를 저장할 폴더가 없으면 생성
    if not os.path.exists(output_images_dir):
        os.makedirs(output_images_dir)

    # 기존 JSON 파일 읽기
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    new_data = {}  # 새 JSON 데이터 저장용

    # 기존 JSON의 각 이미지에 대해 처리
    for img_filename, info in data.items():
        image_path = os.path.join(images_dir, img_filename)
        if not os.path.exists(image_path):
            print(f"[Warning] 이미지 파일이 존재하지 않습니다: {image_path}")
            continue

        # 이미지 열기
        image = Image.open(image_path)
        lines = info.get("lines", [])
        # 각 라인마다 crop 수행
        for idx, line in enumerate(lines):
            bbox = line.get("bbox")
            if not bbox or len(bbox) != 4:
                print(f"[Warning] {img_filename}의 {idx}번째 라인 bbox 정보가 올바르지 않습니다.")
                continue

            # bbox는 [left, top, right, bottom] 형식임 (필요한 경우, int 변환)
            left, top, right, bottom = map(int, bbox)
            # 이미지 crop (PIL.Image.crop는 (left, top, right, bottom) 튜플을 요구)
            cropped = image.crop((left, top, right, bottom))

            # 새 이미지 파일명 생성 (예: "img_00002_line_1.png")
            new_filename = f"{os.path.splitext(img_filename)[0]}_line_{idx+1}.png"
            cropped_image_path = os.path.join(output_images_dir, new_filename)
            cropped.save(cropped_image_path)

            # 새 JSON에 정보 추가: 새 이미지 파일명, text, 원본 이미지명, 원래 bbox 등 필요 정보 기록
            new_data[new_filename] = {
                "text": line.get("text", "").strip(),
                "original_image": img_filename,
            }

    # 새 JSON 파일 저장 (indent와 ensure_ascii=False로 한글이 정상적으로 표시되도록)
    with open(output_json_path, 'w', encoding='utf-8') as fw:
        json.dump(new_data, fw, indent=4, ensure_ascii=False)
    print(f"새 JSON 파일이 생성되었습니다: {output_json_path}")

if __name__ == "__main__":
    # 실제 경로에 맞게 아래 변수들을 수정하세요.
    json_path = "dataset/test_annotations.json"         # 기존 JSON 파일 경로
    images_dir = "dataset/test"                    # 원본 이미지 폴더 경로
    output_images_dir = "dataset/dtrb/test"        # 잘라낸 이미지 저장 폴더
    output_json_path = "dataset/dtrb/test_annotations.json"         # 생성될 새 JSON 파일 경로

    split_lines_and_create_new_json(json_path, images_dir, output_images_dir, output_json_path)
