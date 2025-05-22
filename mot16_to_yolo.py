import os
import glob
import shutil
import numpy as np
import cv2
import argparse
from tqdm import tqdm
from random import shuffle

# ========== CẤU HÌNH ==========
MOT16_PATH = "data_ext/MOT16"
OUTPUT_PATH = "data_mot16"
VAL_SPLIT = 0.2  # Tỷ lệ validation (20%)
PRESERVE_SEQUENCE = False  # Giữ nguyên thứ tự frame trong sequence
RESIZE_DIM = None  # Kích thước resize (None: giữ nguyên), ví dụ: (640, 640)

# ========== TẠO CẤU TRÚC THƯ MỤC ==========
def create_folders():
    folders = [
        f"{OUTPUT_PATH}/images/train",
        f"{OUTPUT_PATH}/images/val",
        f"{OUTPUT_PATH}/labels/train",
        f"{OUTPUT_PATH}/labels/val",
    ]
    for folder in folders:
        os.makedirs(folder, exist_ok=True)
    print(f"[+] Tạo cấu trúc thư mục thành công: {OUTPUT_PATH}")

# ========== CHUYỂN ĐỔI MỘT SEQUENCE ==========
def convert_sequence(sequence_path, preserve_sequence=PRESERVE_SEQUENCE, resize_dim=RESIZE_DIM):
    sequence_name = os.path.basename(sequence_path)
    print(f"\n[>] Đang xử lý sequence: {sequence_name}")

    img_dir = os.path.join(sequence_path, "img1")
    gt_file = os.path.join(sequence_path, "gt", "gt.txt")

    if not os.path.exists(gt_file):
        print(f"[!] Không tìm thấy file groundtruth: {gt_file}")
        return 0, 0

    # Đọc annotations
    gt_data = {}
    with open(gt_file, 'r') as f:
        for line in f:
            line = line.strip().split(',')
            frame_id = int(line[0])
            x, y, w, h = map(float, line[2:6])
            is_active = int(line[6])
            class_id = int(line[7])

            if is_active != 1 or class_id != 1:  # Chỉ lấy người đi bộ
                continue

            gt_data.setdefault(frame_id, []).append((x, y, w, h))

    # Lấy danh sách ảnh
    img_files = sorted(glob.glob(os.path.join(img_dir, "*.jpg")))
    print(f"[=] Tổng ảnh: {len(img_files)}")
    
    # Nếu không giữ nguyên thứ tự frame, trộn ngẫu nhiên để chia train/val
    if not preserve_sequence:
        shuffle(img_files)
    
    split_idx = int(len(img_files) * (1 - VAL_SPLIT))
    train_files = img_files[:split_idx]
    val_files = img_files[split_idx:]
    
    if preserve_sequence:
        print(f"[i] Giữ nguyên thứ tự frames trong sequence")
    
    if resize_dim:
        print(f"[i] Resize ảnh thành {resize_dim[0]}x{resize_dim[1]}")

    counts = {"train": 0, "val": 0}

    for split, files in [("train", train_files), ("val", val_files)]:
        for img_file in tqdm(files, desc=f"--> {split.upper()}"):
            img_name = os.path.basename(img_file)
            frame_id = int(img_name.split('.')[0])
            img = cv2.imread(img_file)
            if img is None:
                print(f"[!] Không đọc được {img_file}")
                continue

            original_h, original_w = img.shape[:2]
            
            # Xử lý resize nếu có yêu cầu
            if resize_dim:
                img = cv2.resize(img, resize_dim, interpolation=cv2.INTER_AREA)
                h, w = resize_dim[1], resize_dim[0]  # resize_dim là (width, height)
            else:
                h, w = original_h, original_w
                
            dst_img_path = os.path.join(OUTPUT_PATH, f"images/{split}", f"{sequence_name}_{img_name}")
            
            # Lưu ảnh (nếu resize thì lưu bản resize, ngược lại copy nguyên bản gốc)
            if resize_dim:
                cv2.imwrite(dst_img_path, img)
            else:
                shutil.copy(img_file, dst_img_path)

            label_path = os.path.join(OUTPUT_PATH, f"labels/{split}", f"{sequence_name}_{img_name.replace('.jpg', '.txt')}")

            valid_lines = []
            if frame_id in gt_data:
                for x, y, bw, bh in gt_data[frame_id]:
                    x_c = (x + bw / 2) / w
                    y_c = (y + bh / 2) / h
                    bw_n = bw / w
                    bh_n = bh / h

                    if all(0 <= v <= 1 for v in [x_c, y_c, bw_n, bh_n]):
                        valid_lines.append(f"0 {x_c:.6f} {y_c:.6f} {bw_n:.6f} {bh_n:.6f}\n")
                    else:
                        print(f"[!] BBox out of bounds in {img_name} – bỏ qua.")

            with open(label_path, 'w') as f:
                f.writelines(valid_lines)

            counts[split] += 1

    return counts["train"], counts["val"]

# ========== TẠO FILE YAML ==========
def create_yaml():
    yaml_content = f"""path: ./{OUTPUT_PATH}
train: images/train
val: images/val

nc: 1
names: ['pedestrian']
"""
    with open(f"{OUTPUT_PATH}/mot16.yaml", "w") as f:
        f.write(yaml_content)
    print(f"[+] Tạo file {OUTPUT_PATH}/mot16.yaml thành công")

# ========== XỬ LÝ THAM SỐ DÒNG LỆNH ==========
def parse_args():
    parser = argparse.ArgumentParser(description="Chuyển đổi MOT16 sang định dạng YOLO")
    parser.add_argument('--mot16-dir', type=str, default=MOT16_PATH, help=f'Đường dẫn đến thư mục MOT16 (mặc định: {MOT16_PATH})')
    parser.add_argument('--output-dir', type=str, default=OUTPUT_PATH, help=f'Đường dẫn đến thư mục đầu ra (mặc định: {OUTPUT_PATH})')
    parser.add_argument('--val-split', type=float, default=VAL_SPLIT, help=f'Tỷ lệ validation (mặc định: {VAL_SPLIT})')
    parser.add_argument('--preserve-sequence', action='store_true', help='Giữ nguyên thứ tự các frame trong mỗi sequence')
    parser.add_argument('--resize', type=int, nargs=2, metavar=('WIDTH', 'HEIGHT'), help='Kích thước resize ảnh (ví dụ: --resize 640 640), mặc định: không resize')
    return parser.parse_args()

# ========== HÀM CHÍNH ==========
def main():
    args = parse_args()
    
    # Cập nhật các tham số toàn cục từ tham số dòng lệnh
    global MOT16_PATH, OUTPUT_PATH, VAL_SPLIT, PRESERVE_SEQUENCE, RESIZE_DIM
    MOT16_PATH = args.mot16_dir
    OUTPUT_PATH = args.output_dir
    VAL_SPLIT = args.val_split
    PRESERVE_SEQUENCE = args.preserve_sequence
    RESIZE_DIM = tuple(args.resize) if args.resize else None
    
    print("=== BẮT ĐẦU CHUYỂN ĐỔI MOT16 SANG YOLO ===")
    print(f"- Thư mục MOT16: {MOT16_PATH}")
    print(f"- Thư mục đầu ra: {OUTPUT_PATH}")
    print(f"- Tỷ lệ validation: {VAL_SPLIT*100:.1f}%")
    print(f"- Giữ nguyên thứ tự frame: {'Có' if PRESERVE_SEQUENCE else 'Không'}")
    print(f"- Resize ảnh: {f'{RESIZE_DIM[0]}x{RESIZE_DIM[1]}' if RESIZE_DIM else 'Không'}")
    
    create_folders()

    train_dir = os.path.join(MOT16_PATH, "train")
    sequences = sorted(glob.glob(os.path.join(train_dir, "MOT16-*")))
    if not sequences:
        print("[!] Không tìm thấy sequence nào trong train/")
        return

    total_train, total_val = 0, 0
    for seq_path in sequences:
        t, v = convert_sequence(seq_path, PRESERVE_SEQUENCE, RESIZE_DIM)
        total_train += t
        total_val += v

    create_yaml()
    print(f"\n✅ Hoàn tất! Tổng ảnh train: {total_train} | val: {total_val}")
    print(f"→ File YAML: {OUTPUT_PATH}/mot16.yaml")
    print("→ Sẵn sàng huấn luyện mô hình với YOLOv9!")

if __name__ == "__main__":
    main()
