import cv2
import torch
import numpy as np
import os
import json
import argparse
from tqdm import tqdm
from models.common import DetectMultiBackend, AutoShape

def parse_args():
    parser = argparse.ArgumentParser(description='Evaluate YOLOv9 model on all test sequences')
    parser.add_argument('--weights', type=str, default='weights/best.pt', help='YOLOv9 weights path')
    parser.add_argument('--data', type=str, default='data_ext/MOT16/train', help='MOT16 train data path')
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
        print(f"ERROR: Ground truth file not found: {gt_file}")
        return gt_boxes, gt_classes
        
    # Debug: Check if we can read any content from the file
    try:
        with open(gt_file, 'r') as f:
            sample_lines = [next(f) for _ in range(5)]
            print(f"Sample lines from {gt_file}:{sample_lines}")
    except Exception as e:
        print(f"Error reading sample lines: {e}")
    
    try:
        with open(gt_file, 'r') as f:
            lines = f.readlines()
            frame_ids = set([int(line.split(',')[0]) for line in lines if len(line.split(',')) >= 8])
            print(f"Available frame IDs in {gt_file}: {min(frame_ids)} to {max(frame_ids)}")
            
            # Check if our frame_id exists in the file
            if frame_id not in frame_ids:
                print(f"WARNING: Frame ID {frame_id} not found in ground truth file!")
                # Try to find first frame in sequence
                if 1 in frame_ids:
                    print(f"First frame (ID=1) exists in ground truth. You might need to adjust frame numbering.")
    except Exception as e:
        print(f"Error analyzing frame IDs: {e}")
    
    with open(gt_file, 'r') as f:
        frame_gt_count = 0
        for line in f:
            parts = line.strip().split(',')
            if len(parts) < 8:
                continue
                
            # MOT16 format: <frame>, <id>, <bb_left>, <bb_top>, <bb_width>, <bb_height>, <conf>, <class>
            try:
                frame_num = int(parts[0])
                if frame_num != frame_id:
                    continue
                    
                # Count all entries for this frame, regardless of other filters
                frame_gt_count += 1
                
                # For MOT16, pedestrians are class 1
                try:
                    class_id = int(parts[7])
                    if class_id != 1:  # Only consider pedestrians (class 1 in MOT16)
                        continue
                except ValueError:
                    print(f"Invalid class ID in line: {line}")
                    continue
                
                # Skip visibility filtering for debugging
                # if float(parts[6]) < 0.25:
                #     continue
                
                try:
                    x, y, w, h = map(float, parts[2:6])
                    # Convert to [x1, y1, x2, y2] format
                    gt_boxes.append([x, y, x + w, y + h])
                    gt_classes.append(class_id)
                except ValueError:
                    print(f"Invalid bbox format in line: {line}")
                    continue
            except Exception as e:
                print(f"Error processing line: {line}. Error: {e}")
        
        # Debug info about this specific frame
        print(f"Frame {frame_id}: Found {frame_gt_count} total entries, {len(gt_boxes)} valid pedestrian boxes")
    
    return gt_boxes, gt_classes

def evaluate_sequence(model, sequence_path, args):
    """Evaluate model on a single sequence"""
    sequence_name = os.path.basename(sequence_path)
    img_path = os.path.join(sequence_path, "img1")
    gt_file = os.path.join(sequence_path, "gt", "gt.txt")
    
    # Kiu1ec3m tra u0111u01b0u1eddng du1eabn
    if not os.path.exists(img_path):
        print(f"Image path not found: {img_path}")
        return None
    
    # Lu1ea5y danh su00e1ch u1ea3nh
    image_files = sorted([f for f in os.listdir(img_path) if f.endswith('.jpg')])
    if not image_files:
        print(f"No images found in {img_path}")
        return None
    
    # Khu1edfi tu1ea1o biu1ebfn thu1ed1ng ku00ea
    total_frames = len(image_files)
    total_gt_objects = 0
    total_detections = 0
    total_true_positives = 0
    total_false_positives = 0
    iou_values = []
    
    # Xu1eed lu00fd tu1eebng frame
    for img_file in tqdm(image_files, desc=f"Evaluating {sequence_name}"):
        # Lu1ea5y frame_id tu1eeb tu00ean file (000001.jpg -> 1)
        filename_without_ext = os.path.splitext(img_file)[0]
        # Chuynu1ec3n '000001' thu00e0nh 1 bu1eb1ng cu00e1ch loi1ea1i bu1ecf cu00e1c su1ed1 0 u1edf u0111u1ea7u ngu1ea7m u0111u1ecbnh vu00e0 chuynu1ec3n thu00e0nh su1ed1
        frame_id = int(filename_without_ext)
        
        if frame_id % 50 == 0 or frame_id < 5:
            print(f"Processing image {img_file} -> frame_id = {frame_id}")
        
        # u0110u1ecdc u1ea3nh
        img_path_full = os.path.join(img_path, img_file)
        frame = cv2.imread(img_path_full)
        if frame is None:
            continue
        
        # Load ground truth boxes cho frame hiu1ec7n tu1ea1i
        gt_boxes, gt_classes = load_ground_truth(gt_file, frame_id)
        total_gt_objects += len(gt_boxes)
        
        # Detect objects vu1edbi YOLOv9
        results = model(frame)
        
        # Lu1ea5y cu00e1c detection vu1edbi confidence > threshold
        detections = []
        for det in results.pred[0]:
            label, confidence, bbox = det[5], det[4], det[:4]
            class_id = int(label)
            if confidence < args.conf:
                continue
            
            # Chu1ec9 quan tu00e2m u0111u1ebfn ngu01b0u1eddi (class 0 trong YOLOv9)
            if class_id != 0:  
                continue
                
            x1, y1, x2, y2 = map(int, bbox)
            detections.append([x1, y1, x2, y2, confidence])
        
        total_detections += len(detections)
        
        # Tu00ednh IoU vu00e0 u0111u1ebfm TP, FP
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
            
            # Nu1ebfu IoU > threshold, u0111u00e2y lu00e0 true positive
            if best_iou > args.iou_thresh and best_gt_idx >= 0:
                true_positives += 1
                used_gt[best_gt_idx] = True
                iou_values.append(best_iou)
            else:
                false_positives += 1
        
        total_true_positives += true_positives
        total_false_positives += false_positives
    
    # Tu00ednh cu00e1c chu1ec9 su1ed1 u0111u00e1nh giu00e1
    precision = total_true_positives / max(1, (total_true_positives + total_false_positives))
    recall = total_true_positives / max(1, total_gt_objects)
    f1_score = 2 * precision * recall / max(1e-6, precision + recall)
    average_iou = sum(iou_values) / max(1, len(iou_values))
    
    # Tru1ea3 vu1ec1 ku1ebft quu1ea3 cu1ee7a sequence
    return {
        "sequence": sequence_name,
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

def evaluate_all_sequences(args):
    # Khu1edfi tu1ea1o YOLOv9
    print(f"Loading model from {args.weights}...")
    device = args.device
    model = DetectMultiBackend(weights=args.weights, device=device, fuse=True)
    model = AutoShape(model)
    
    # Tu1ea1o thu01b0 mu1ee5c ku1ebft quu1ea3
    results_dir = "evaluation_results"
    if args.save_results and not os.path.exists(results_dir):
        os.makedirs(results_dir)
    
    # Tu00ecm tu1ea5t cu1ea3 cu00e1c sequences trong thu01b0 mu1ee5c test
    all_sequences = []
    for entry in os.listdir(args.data):
        if entry.startswith("MOT16-") and os.path.isdir(os.path.join(args.data, entry)):
            all_sequences.append(os.path.join(args.data, entry))
    
    if not all_sequences:
        print(f"No MOT16 sequences found in {args.data}")
        return
    
    # Su1eafp xu1ebfp cu00e1c sequences
    all_sequences = sorted(all_sequences)
    print(f"Found {len(all_sequences)} sequences: {[os.path.basename(s) for s in all_sequences]}")
    
    # Khu1edfi tu1ea1o biu1ebfn u0111u1ec3 tu00edch lu0169y ku1ebft quu1ea3 tou00e0n cu1ee5c
    global_results = {
        "total_frames": 0,
        "total_gt_objects": 0,
        "total_detections": 0,
        "true_positives": 0,
        "false_positives": 0,
        "false_negatives": 0,
        "all_iou_values": []
    }
    
    # u0110u00e1nh giu00e1 tu1eebng sequence
    sequence_results = []
    for seq_path in all_sequences:
        result = evaluate_sequence(model, seq_path, args)
        if result is not None:
            sequence_results.append(result)
            
            # Cu1ed9ng du1ed3n vu00e0o ku1ebft quu1ea3 tou00e0n cu1ee5c
            global_results["total_frames"] += result["total_frames"]
            global_results["total_gt_objects"] += result["total_gt_objects"]
            global_results["total_detections"] += result["total_detections"]
            global_results["true_positives"] += result["true_positives"]
            global_results["false_positives"] += result["false_positives"]
            global_results["false_negatives"] += result["false_negatives"]
    
    # Tu00ednh tou00e1n cu00e1c chu1ec9 su1ed1 tou00e0n cu1ee5c
    global_precision = global_results["true_positives"] / max(1, (global_results["true_positives"] + global_results["false_positives"]))
    global_recall = global_results["true_positives"] / max(1, (global_results["true_positives"] + global_results["false_negatives"]))
    global_f1 = 2 * global_precision * global_recall / max(1e-6, global_precision + global_recall)
    
    # Tu00ednh average IoU tou00e0n cu1ee5c
    avg_iou_by_sequence = [r["average_iou"] for r in sequence_results]
    global_avg_iou = sum(avg_iou_by_sequence) / len(avg_iou_by_sequence) if avg_iou_by_sequence else 0
    
    # Hiu1ec3n thu1ecb ku1ebft quu1ea3 cho tu1eebng sequence
    print("\n===== RESULTS BY SEQUENCE =====")
    for result in sequence_results:
        print(f"{result['sequence']}:\n  Precision: {result['precision']:.4f}, Recall: {result['recall']:.4f}, F1: {result['f1_score']:.4f}, IoU: {result['average_iou']:.4f}")
    
    # Hiu1ec3n thu1ecb ku1ebft quu1ea3 tou00e0n cu1ee5c
    print("\n===== OVERALL RESULTS =====")
    print(f"Total frames: {global_results['total_frames']}")
    print(f"Total ground truth objects: {global_results['total_gt_objects']}")
    print(f"Total detections: {global_results['total_detections']}")
    print(f"True positives: {global_results['true_positives']}")
    print(f"False positives: {global_results['false_positives']}")
    print(f"False negatives: {global_results['false_negatives']}")
    print(f"GLOBAL PRECISION: {global_precision:.4f}")
    print(f"GLOBAL RECALL: {global_recall:.4f}")
    print(f"GLOBAL F1 SCORE: {global_f1:.4f}")
    print(f"AVERAGE IoU (across sequences): {global_avg_iou:.4f}")
    
    # Lu01b0u ku1ebft quu1ea3 vu00e0o file
    if args.save_results:
        # Lu01b0u ku1ebft quu1ea3 chi tiu1ebft cho tu1eebng sequence
        for result in sequence_results:
            seq_result_file = os.path.join(results_dir, f"{result['sequence']}_results.json")
            with open(seq_result_file, 'w') as f:
                json.dump(result, f, indent=4)
        
        # Lu01b0u ku1ebft quu1ea3 tu1ed5ng quu00e1t
        global_result_file = os.path.join(results_dir, "global_results.json")
        global_summary = {
            "global_precision": global_precision,
            "global_recall": global_recall,
            "global_f1_score": global_f1,
            "global_avg_iou": global_avg_iou,
            "total_frames": global_results["total_frames"],
            "total_gt_objects": global_results["total_gt_objects"],
            "total_detections": global_results["total_detections"],
            "true_positives": global_results["true_positives"],
            "false_positives": global_results["false_positives"],
            "false_negatives": global_results["false_negatives"],
            "sequence_results": sequence_results
        }
        
        with open(global_result_file, 'w') as f:
            json.dump(global_summary, f, indent=4)
        
        print(f"\nResults saved to {results_dir}/")

if __name__ == "__main__":
    args = parse_args()
    evaluate_all_sequences(args)
