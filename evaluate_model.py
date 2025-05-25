import cv2
import torch
import numpy as np
import os
import json
import argparse
from tqdm import tqdm
from models.common import DetectMultiBackend, AutoShape

def parse_args():
    parser = argparse.ArgumentParser(description='Evaluate YOLOv9 model')
    parser.add_argument('--weights', type=str, default='runs/train/yolov9-c-MOT16/weights/best.pt', help='YOLOv9 weights path')
    parser.add_argument('--data', type=str, default='data_ext/MOT16/test', help='MOT16 test data path')
    parser.add_argument('--sequence', type=str, default='MOT16-01', help='MOT16 sequence to evaluate')
    parser.add_argument('--conf', type=float, default=0.5, help='confidence threshold')
    parser.add_argument('--iou-thresh', type=float, default=0.5, help='IoU threshold')
    parser.add_argument('--device', type=str, default='cpu', help='cuda device, i.e. 0 or cpu')
    parser.add_argument('--save-results', action='store_true', help='save evaluation results to file')
    return parser.parse_args()

def calculate_iou(boxA, boxB):
    """Calculate IoU between two bounding boxes"""
    # boxA and boxB format: [x1, y1, x2, y2]
    
    # Find coordinates of intersection
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])
    
    # Compute area of intersection
    interArea = max(0, xB - xA) * max(0, yB - yA)
    
    # Compute area of both boxes
    boxAArea = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
    boxBArea = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])
    
    # Compute IoU
    iou = interArea / float(boxAArea + boxBArea - interArea)
    
    return iou

def load_ground_truth(gt_file, frame_id):
    """Load ground truth bounding boxes for a specific frame"""
    gt_boxes = []
    gt_classes = []
    
    if not os.path.exists(gt_file):
        print(f"Ground truth file not found: {gt_file}")
        return gt_boxes, gt_classes
    
    # Debug information
    print(f"Loading ground truth from {gt_file} for frame {frame_id}")
    
    with open(gt_file, 'r') as f:
        for line in f:
            parts = line.strip().split(',')
            if len(parts) < 8:
                continue
                
            # MOT16 format: <frame>, <id>, <bb_left>, <bb_top>, <bb_width>, <bb_height>, <conf>, <class>
            frame_num = int(parts[0])
            if frame_num != frame_id:
                continue
                
            # For MOT16, we're mainly interested in pedestrians (class 1)
            class_id = int(parts[7])
            if class_id != 1:  # Only consider pedestrians (class 1 in MOT16)
                continue
                
            # Visibility might be in a different column or format depending on the MOT version
            # Skip visibility check for now to make sure we're not filtering incorrectly
            
            x, y, w, h = map(float, parts[2:6])
            # Convert to [x1, y1, x2, y2] format
            gt_boxes.append([x, y, x + w, y + h])
            gt_classes.append(class_id)
            
    # Debug info
    if len(gt_boxes) == 0:
        print(f"WARNING: No ground truth boxes found for frame {frame_id}")
    else:
        print(f"Found {len(gt_boxes)} ground truth boxes for frame {frame_id}")
    
    return gt_boxes, gt_classes

def evaluate_model(args):
    # Khởi tạo YOLOv9
    device = args.device
    model = DetectMultiBackend(weights=args.weights, device=device, fuse=True)
    model = AutoShape(model)
    
    # Đường dẫn đến sequence cần đánh giá
    sequence_path = os.path.join(args.data, args.sequence)
    img_path = os.path.join(sequence_path, "img1")
    gt_file = os.path.join(sequence_path, "gt", "gt.txt")
    
    # Kiểm tra đường dẫn
    if not os.path.exists(img_path):
        print(f"Image path not found: {img_path}")
        return
    
    # Lấy danh sách ảnh
    image_files = sorted([f for f in os.listdir(img_path) if f.endswith('.jpg')])
    if not image_files:
        print(f"No images found in {img_path}")
        return
    
    # Khởi tạo biến thống kê
    total_frames = len(image_files)
    total_gt_objects = 0
    total_detections = 0
    total_true_positives = 0
    total_false_positives = 0
    average_iou = 0
    iou_values = []
    
    # Tạo thư mục kết quả
    results_dir = "evaluation_results"
    if args.save_results and not os.path.exists(results_dir):
        os.makedirs(results_dir)
    
    # Xử lý từng frame
    for img_file in tqdm(image_files, desc=f"Evaluating {args.sequence}"):
        # Lấy frame_id từ tên file
        frame_id = int(os.path.splitext(img_file)[0])
        
        # Đọc ảnh
        img_path_full = os.path.join(img_path, img_file)
        frame = cv2.imread(img_path_full)
        if frame is None:
            print(f"Could not read image: {img_path_full}")
            continue
        
        # Load ground truth boxes cho frame hiện tại
        gt_boxes, gt_classes = load_ground_truth(gt_file, frame_id)
        total_gt_objects += len(gt_boxes)
        
        # Detect objects với YOLOv9
        results = model(frame)
        
        # Lấy các detection với confidence > threshold
        detections = []
        for det in results.pred[0]:
            label, confidence, bbox = det[5], det[4], det[:4]
            class_id = int(label)
            if confidence < args.conf:
                continue
            
            # Chỉ quan tâm đến người (class 0 trong YOLOv9)
            if class_id != 0:  
                continue
                
            x1, y1, x2, y2 = map(int, bbox)
            detections.append([x1, y1, x2, y2, confidence])
        
        total_detections += len(detections)
        
        # Tính IoU và đếm TP, FP
        true_positives = 0
        false_positives = 0
        used_gt = [False] * len(gt_boxes)
        
        for det in detections:
            best_iou = 0
            best_gt_idx = -1
            
            for i, gt_box in enumerate(gt_boxes):
                if used_gt[i]:
                    continue
                    
                iou = calculate_iou(det[:4], gt_box)
                if iou > best_iou:
                    best_iou = iou
                    best_gt_idx = i
            
            # Nếu IoU > threshold, đây là true positive
            if best_iou > args.iou_thresh and best_gt_idx >= 0:
                true_positives += 1
                used_gt[best_gt_idx] = True
                iou_values.append(best_iou)
            else:
                false_positives += 1
        
        total_true_positives += true_positives
        total_false_positives += false_positives
    
    # Tính các chỉ số đánh giá
    precision = total_true_positives / max(1, (total_true_positives + total_false_positives))
    recall = total_true_positives / max(1, total_gt_objects)
    f1_score = 2 * precision * recall / max(1e-6, precision + recall)
    average_iou = sum(iou_values) / max(1, len(iou_values))
    
    # Hiển thị kết quả
    print("\n===== EVALUATION RESULTS =====")
    print(f"Sequence: {args.sequence}")
    print(f"Total frames: {total_frames}")
    print(f"Total ground truth objects: {total_gt_objects}")
    print(f"Total detections: {total_detections}")
    print(f"True positives: {total_true_positives}")
    print(f"False positives: {total_false_positives}")
    print(f"False negatives: {total_gt_objects - total_true_positives}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1 Score: {f1_score:.4f}")
    print(f"Average IoU: {average_iou:.4f}")
    
    # Lưu kết quả vào file
    if args.save_results:
        results = {
            "sequence": args.sequence,
            "total_frames": total_frames,
            "total_gt_objects": total_gt_objects,
            "total_detections": total_detections,
            "true_positives": total_true_positives,
            "false_positives": total_false_positives,
            "false_negatives": total_gt_objects - total_true_positives,
            "precision": precision,
            "recall": recall,
            "f1_score": f1_score,
            "average_iou": average_iou
        }
        
        results_file = os.path.join(results_dir, f"{args.sequence}_results.json")
        with open(results_file, 'w') as f:
            json.dump(results, f, indent=4)
        print(f"Results saved to {results_file}")

if __name__ == "__main__":
    args = parse_args()
    evaluate_model(args)
