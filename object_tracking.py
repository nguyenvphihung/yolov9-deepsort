import cv2
import torch
import numpy as np
import os
import glob
from deep_sort_realtime.deepsort_tracker import DeepSort
from models.common import DetectMultiBackend, AutoShape

# Config value - MOT16 dataset
mot16_sequence = "MOT16-01"  # Thay đổi tên sequence theo nhu cầu
mot16_base_path = "data_ext/MOT16/test"  # Đường dẫn đến thư mục test của MOT16
sequence_path = os.path.join(mot16_base_path, mot16_sequence)
img_path = os.path.join(sequence_path, "img1")  # MOT16 lưu ảnh trong thư mục img1

conf_threshold = 0.5
tracking_class = 0  # 0: thường là người đi bộ trong COCO/YOLOv9

# Khởi tạo DeepSort
tracker = DeepSort(max_age=30)

# Khởi tạo YOLOv9
device = "cpu" # "cuda": GPU, "cpu": CPU, "mps:0"
model  = DetectMultiBackend(weights="weights/yolov9-c.pt", device=device, fuse=True )
model  = AutoShape(model)

# Load classname từ file classes.names
with open("data_ext/classes.names") as f:
    class_names = f.read().strip().split('\n')

colors = np.random.randint(0,255, size=(len(class_names),3 ))
tracks = []

# Lấy danh sách ảnh từ sequence MOT16
image_files = sorted(glob.glob(os.path.join(img_path, "*.jpg")))
if not image_files:
    print(f"Không tìm thấy ảnh trong {img_path}")
    exit(1)

print(f"Đã tìm thấy {len(image_files)} ảnh trong {mot16_sequence}")

# Tiến hành đọc từng ảnh trong sequence
for i, img_file in enumerate(image_files):
    # Đọc ảnh
    frame = cv2.imread(img_file)
    if frame is None:
        print(f"Không thể đọc {img_file}")
        continue
    
    # Hiển thị thông tin tiến trình
    print(f"Xử lý ảnh {i+1}/{len(image_files)}: {os.path.basename(img_file)}")
    # Đưa qua model để detect
    results = model(frame)

    detect = []
    for detect_object in results.pred[0]:
        label, confidence, bbox = detect_object[5], detect_object[4], detect_object[:4]
        x1, y1, x2, y2 = map(int, bbox)
        class_id = int(label)

        if tracking_class is None:
            if confidence < conf_threshold:
                continue
        else:
            if class_id != tracking_class or confidence < conf_threshold:
                continue

        detect.append([ [x1, y1, x2-x1, y2 - y1], confidence, class_id ])


    # Cập nhật,gán ID băằng DeepSort
    tracks = tracker.update_tracks(detect, frame = frame)

    # Vẽ lên màn hình các khung chữ nhật kèm ID
    for track in tracks:
        if track.is_confirmed():
            track_id = track.track_id

            # Lấy toạ độ, class_id để vẽ lên hình ảnh
            ltrb = track.to_ltrb()
            class_id = track.get_det_class()
            x1, y1, x2, y2 = map(int, ltrb)
            color = colors[class_id]
            B, G, R = map(int,color)

            label = "{}-{}".format(class_names[class_id], track_id)

            cv2.rectangle(frame, (x1, y1), (x2, y2), (B, G, R), 2)
            cv2.rectangle(frame, (x1 - 1, y1 - 20), (x1 + len(label) * 12, y1), (B, G, R), -1)
            cv2.putText(frame, label, (x1 + 5, y1 - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

    # Show hình ảnh lên màn hình
    cv2.imshow("OT", frame)
    # Bấm Q thì thoát
    key = cv2.waitKey(1)
    if key == ord("q") or key == 27:  
        break

cv2.destroyAllWindows()
print("Hoàn thành xử lý toàn bộ sequence MOT16")
