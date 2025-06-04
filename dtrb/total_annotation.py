import os
import shutil

# 원본 이미지가 들어 있는 폴더들
src_dirs = [
    "synthetic_detection_data/val",
    "synthetic_detection_data/test"
]

# 합쳐서 저장할 대상 폴더
combined_dir = "synthetic_detection_data/val_test_images"
os.makedirs(combined_dir, exist_ok=True)

# 각 폴더를 순회하며 이미지 복사(or 이동)
for src in src_dirs:
    for fname in os.listdir(src):
        src_path = os.path.join(src, fname)
        dst_path = os.path.join(combined_dir, fname)
        if os.path.isfile(src_path):
            # 복사하려면 copy2, 이동하려면 rename 으로 바꿔 쓰시면 됩니다.
            shutil.copy2(src_path, dst_path)
            # shutil.move(src_path, dst_path)  # 이동을 원하면 이걸 사용

print(f"모든 이미지를 '{combined_dir}' 폴더로 통합했습니다. 총 {len(os.listdir(combined_dir))}개")
