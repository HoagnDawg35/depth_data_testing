#!/usr/bin/env python3
import os
import sys
import time
import cv2
import numpy as np
import torch
from pathlib import Path

# Set MPS fallback for Apple Silicon
if hasattr(torch, 'backends') and hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
    os.environ['PYTORCH_ENABLE_MPS_FALLBACK'] = '1'

# Import modules
from detection_model import ObjectDetector
from depth_model import DepthEstimator
from bbox3d_utils import BBox3DEstimator, BirdEyeView
from load_camera_params import load_camera_params, apply_camera_params_to_estimator

def main():
    # Configs
    source = r"H:\AIC25\data\Camera_0006.mp4"
    output_path = "output_depth.mp4"
    yolo_model_size = "nano"
    depth_model_size = "small"
    device = 'cpu'
    conf_threshold = 0.25
    iou_threshold = 0.45
    classes = None
    enable_tracking = True
    enable_bev = True
    enable_pseudo_3d = True
    camera_params_file = None

    print(f"Using device: {device}")
    print("Initializing models...")

    try:
        detector = ObjectDetector(
            model_size=yolo_model_size,
            conf_thres=conf_threshold,
            iou_thres=iou_threshold,
            classes=classes,
            device=device
        )
    except Exception as e:
        print(f"Error initializing object detector: {e}")
        print("Falling back to CPU for object detection")
        detector = ObjectDetector(
            model_size=yolo_model_size,
            conf_thres=conf_threshold,
            iou_thres=iou_threshold,
            classes=classes,
            device='cpu'
        )

    try:
        depth_estimator = DepthEstimator(
            model_size=depth_model_size,
            device=device
        )
    except Exception as e:
        print(f"Error initializing depth estimator: {e}")
        print("Falling back to CPU for depth estimation")
        depth_estimator = DepthEstimator(
            model_size=depth_model_size,
            device='cpu'
        )

    bbox3d_estimator = BBox3DEstimator()
    if enable_bev:
        bev = BirdEyeView(scale=60, size=(300, 300))

    # Handle camera/video input
    if isinstance(source, str) and source.isdigit():
        source = int(source)

    print(f"Opening video source: {source}")
    cap = cv2.VideoCapture(source)

    if not cap.isOpened():
        print(f"Error: Could not open video source {source}")
        return

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS)) or 30

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    frame_count = 0
    start_time = time.time()
    fps_display = "FPS: --"

    print("Starting processing...")

    while True:
        try:
            ret, frame = cap.read()
            if not ret:
                break

            original_frame = frame.copy()
            detection_frame = frame.copy()
            result_frame = frame.copy()

            # Step 1: Detection
            try:
                detection_frame, detections = detector.detect(detection_frame, track=enable_tracking)
            except Exception as e:
                print(f"[Frame {frame_count}] Detection error: {e}")
                detections = []
                cv2.putText(detection_frame, "Detection Error", (10, 60),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

            # Step 2: Depth
            try:
                depth_map = depth_estimator.estimate_depth(original_frame)
                depth_colored = depth_estimator.colorize_depth(depth_map)
            except Exception as e:
                print(f"[Frame {frame_count}] Depth error: {e}")
                depth_map = np.zeros((height, width), dtype=np.float32)
                depth_colored = np.zeros((height, width, 3), dtype=np.uint8)
                cv2.putText(depth_colored, "Depth Error", (10, 60),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

            # Step 3: 3D Boxes
            boxes_3d = []
            active_ids = []
            for detection in detections:
                try:
                    bbox, score, class_id, obj_id = detection
                    class_name = detector.get_class_names()[class_id]

                    if class_name.lower() in ['person', 'cat', 'dog']:
                        center_x = int((bbox[0] + bbox[2]) / 2)
                        center_y = int((bbox[1] + bbox[3]) / 2)
                        depth_value = depth_estimator.get_depth_at_point(depth_map, center_x, center_y)
                        method = 'center'
                    else:
                        depth_value = depth_estimator.get_depth_in_region(depth_map, bbox, method='median')
                        method = 'median'

                    box_3d = {
                        'bbox_2d': bbox,
                        'depth_value': depth_value,
                        'depth_method': method,
                        'class_name': class_name,
                        'object_id': obj_id,
                        'score': score
                    }
                    boxes_3d.append(box_3d)
                    if obj_id is not None:
                        active_ids.append(obj_id)
                except Exception as e:
                    print(f"[Frame {frame_count}] Box processing error: {e}")
                    continue

            bbox3d_estimator.cleanup_trackers(active_ids)

            # Step 4: Visualization
            for box in boxes_3d:
                try:
                    class_name = box['class_name'].lower()
                    if 'car' in class_name:
                        color = (0, 0, 255)
                    elif 'person' in class_name:
                        color = (0, 255, 0)
                    elif 'bicycle' in class_name:
                        color = (255, 0, 0)
                    elif 'plant' in class_name:
                        color = (0, 255, 255)
                    else:
                        color = (255, 255, 255)
                    result_frame = bbox3d_estimator.draw_box_3d(result_frame, box, color)
                except Exception as e:
                    print(f"[Frame {frame_count}] Draw error: {e}")

            # BEV
            if enable_bev:
                try:
                    bev.reset()
                    for box in boxes_3d:
                        bev.draw_box(box)
                    bev_image = bev.get_image()
                    bev_h = height // 4
                    bev_w = bev_h
                    if bev_h > 0 and bev_w > 0:
                        bev_resized = cv2.resize(bev_image, (bev_w, bev_h))
                        result_frame[height - bev_h:height, 0:bev_w] = bev_resized
                        cv2.rectangle(result_frame, (0, height - bev_h), (bev_w, height), (255, 255, 255), 1)
                        cv2.putText(result_frame, "Bird's Eye View",
                                    (10, height - bev_h + 20),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                except Exception as e:
                    print(f"[Frame {frame_count}] BEV error: {e}")

            # FPS
            frame_count += 1
            if frame_count % 10 == 0:
                elapsed = time.time() - start_time
                fps_display = f"FPS: {frame_count / elapsed:.1f}"

            cv2.putText(result_frame, f"{fps_display} | Device: {device}", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

            # Add depth preview
            try:
                depth_h = height // 4
                depth_w = depth_h * width // height
                depth_resized = cv2.resize(depth_colored, (depth_w, depth_h))
                result_frame[0:depth_h, 0:depth_w] = depth_resized
            except Exception as e:
                print(f"[Frame {frame_count}] Depth preview error: {e}")

            # Output and display
            out.write(result_frame)
            cv2.imshow("3D Object Detection", result_frame)
            cv2.imshow("Depth Map", depth_colored)
            cv2.imshow("Object Detection", detection_frame)

            key = cv2.waitKey(1)
            if key == ord('q') or key == 27:
                print("Exit requested.")
                break

        except Exception as e:
            print(f"[Frame {frame_count}] General processing error: {e}")
            key = cv2.waitKey(1)
            if key == ord('q') or key == 27:
                break
            continue

    # Clean up
    print("Cleaning up...")
    cap.release()
    out.release()
    cv2.destroyAllWindows()
    print(f"Output saved to {output_path}")

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nInterrupted by user.")
        cv2.destroyAllWindows()
