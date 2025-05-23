# YOLOv9-DeepSORT: Hệ thống Tracking Người Đi Bộ

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.8+-blue.svg" alt="Python Version"/>
  <img src="https://img.shields.io/badge/PyTorch-2.0+-green.svg" alt="PyTorch Version"/>
  <img src="https://img.shields.io/badge/Status-Active-success.svg" alt="Status"/>
</p>

Dự án này kết hợp sức mạnh của **YOLOv9** (một trong những detector hiện đại nhất) với **DeepSORT** (giải pháp tracking hiệu quả) để tạo nên một hệ thống tracking người đi bộ hiệu suất cao.

## Giới thiệu

Hệ thống tracking người đi bộ có nhiều ứng dụng trong thực tế như giám sát an ninh, phân tích dòng người, nghiên cứu hành vi... Dự án này được thiết kế để tracking người đi bộ trong video tự quay, sử dụng mô hình YOLOv9 được fine-tune trên tập dữ liệu MOT16.

### Đặc điểm

- **Detection chính xác**: Sử dụng YOLOv9 fine-tuned để phát hiện người đi bộ với độ chính xác cao
- **Tracking bền vững**: DeepSORT giải quyết vấn đề occlusion và giúp duy trì ID tracking ngay cả khi đối tượng tạm thời bị che khuất
- **Tối ưu hóa hiệu suất**: Thiết kế để cân bằng giữa độ chính xác và tốc độ xử lý

## Cấu trúc thư mục chính của dự án

```
├── data_mot16/        # Dữ liệu MOT16 đã chuyển đổi
│   ├── images/        # Ảnh train và validation
│   ├── labels/        # Nhãn theo định dạng YOLO
│   └── mot16.yaml     # Cấu hình dataset
├── models/           # Kiến trúc mô hình
├── runs/             # Kết quả huấn luyện và inference
├── weights/          # Các file weights mô hình
├── deep_sort/        # Module DeepSORT để tracking
├── mot16_to_yolo.py  # Script chuyển đổi MOT16 sang định dạng YOLO
├── train_modified.py # Script huấn luyện đã chỉnh sửa cho PyTorch 2.6+
├── object_tracking.py # Script chính để tracking
└── README.md        # Tài liệu dự án
```

## Quy trình Fine-tuning YOLOv9

### 1. Chuẩn bị dữ liệu

Tập dữ liệu MOT16 (Multiple Object Tracking Benchmark) được chuyển đổi sang định dạng YOLO sử dụng script `mot16_to_yolo.py`. Quy trình bao gồm:

1. **Chuyển đổi dữ liệu**: Chuyển format MOT16 (x,y,w,h) sang YOLO (x_center,y_center,w,h đã chuẩn hóa)
2. **Chia train/val**: 80% dữ liệu dùng cho training, 20% dùng cho validation
3. **Lọc dữ liệu**: Loại bỏ các bounding box không hợp lệ hoặc nằm ngoài khung hình

### 2. Fine-tuning YOLOv9

**Fine-tuning** là quá trình tận dụng mô hình YOLOv9 đã được huấn luyện trước (pretrained) trên bộ dữ liệu lớn (COCO dataset) và điều chỉnh cho tập dữ liệu cụ thể (MOT16) với một mục tiêu hẹp hơn (chỉ phát hiện người đi bộ).

Quá trình fine-tuning bao gồm:

1. **Tải mô hình pretrained**: Sử dụng YOLOv9-C đã huấn luyện trên COCO với 80 lớp
2. **Điều chỉnh kiến trúc**: Thay đổi lớp đầu ra từ 80 lớp xuống 1 lớp (pedestrian)
3. **Huấn luyện có điều chỉnh**: Giữ nguyên các đặc trưng cấp thấp (edges, textures), chỉ điều chỉnh các lớp cao hơn để tối ưu cho việc phát hiện người đi bộ
4. **Lưu mô hình tốt nhất**: File `best.pt` chứa mô hình có hiệu suất tốt nhất trên tập validation

Lệnh huấn luyện:

```bash
python train_modified.py --device cpu --batch 1 --epochs 50 --data data_mot16/mot16.yaml --cfg models/detect/yolov9-c.yaml --weights weights/yolov9-c.pt --name best --hyp data/hyps/hyp.scratch-high.yaml --exist-ok
```

### 3. Lợi ích của Fine-tuning

- **Hiệu suất tốt hơn**: Mô hình fine-tuned phát hiện người đi bộ chính xác hơn mô hình chung
- **Tiết kiệm thời gian**: Không cần huấn luyện từ đầu với dữ liệu hạn chế
- **Đảm bảo tổng quát hóa**: Mô hình giữ lại kiến thức nền về các đặc trưng hình ảnh từ dữ liệu lớn COCO

## Hướng dẫn sử dụng

### 1. Chuẩn bị môi trường

```bash
# Tạo môi trường
pip install -r requirements.txt
pip install -r setup.txt

# Tải mô hình YOLOv9 (nếu chưa có)
# Tạo thư mục weights và tải YOLOv9-C về
```

### 2. Chuyển đổi dữ liệu (nếu cần)

```bash
# Chuyển đổi MOT16 sang format YOLO
python mot16_to_yolo.py

# Resize ảnh và giữ nguyên thứ tự các frame trong sequence
python mot16_to_yolo.py --resize 640 640 --preserve-sequence
```

### 3. Fine-tune YOLOv9

```bash
# Fine-tune mô hình YOLOv9-C
python train_modified.py --device cpu --batch 1 --epochs 50 --data data_mot16/mot16.yaml --cfg models/detect/yolov9-c.yaml --weights weights/yolov9-c.pt --name yolov9-c-MOT16 --hyp data/hyps/hyp.scratch-high.yaml --exist-ok

# Hoặc sử dụng mô hình nhẹ hơn nếu CPU yếu
python train_modified.py --device cpu --batch 2 --epochs 50 --data data_mot16/mot16.yaml --cfg models/detect/yolov9-s.yaml --weights weights/yolov9-s.pt --name yolov9-s-MOT16 --hyp data/hyps/hyp.scratch-high.yaml --exist-ok
```

### 4. Tracking người đi bộ trong video

```bash
# Sử dụng mô hình đã fine-tune để tracking
python object_tracking.py --source path/to/your/video.mp4 --weights runs/train/yolov9-c-MOT16/weights/best.pt --output output/result.mp4
```

## Kết quả

Sau khi fine-tune YOLOv9 trên MOT16 và kết hợp với DeepSORT, hệ thống tracking đạt hiệu suất ổn định với:

- **Độ chính xác phát hiện người khá hiệu quả**
- **Tracking ổn định ngay cả trong các trường hợp khó**
- **Duy trì ID người qua nhiều frame**

## Ghi chú

- Hệ thống hoạt động tốt nhất với GPU, nhưng vẫn có thể chạy trên CPU với tốc độ chậm hơn
- Bộ dữ liệu MOT16 chỉ được sử dụng để fine-tune mô hình, mục tiêu cuối cùng là tracking trên video tự quay

## Nguồn tham khảo

- [YOLOv9 Repository](https://github.com/WongKinYiu/yolov9)
- [DeepSORT Repository](https://github.com/nwojke/deep_sort)
- [MOT16 Dataset](https://motchallenge.net/data/MOT16/)
