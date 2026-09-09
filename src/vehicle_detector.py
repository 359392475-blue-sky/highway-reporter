"""
vehicle_detector.py - 车辆检测模块
使用 YOLOv8 检测视频帧中的车辆
"""
# CREATED_BY: highway-reporter MVP
# STATUS: active

import cv2
import numpy as np
from ultralytics import YOLO
from pathlib import Path

from .config import YOLO_CONFIDENCE, YOLO_MODEL


# COCO 中的车辆类别 ID
VEHICLE_CLASSES = {
    2: 'car',
    3: 'motorcycle',
    5: 'bus',
    7: 'truck',
}

# 特殊车辆关键词（用于后续过滤救护车等）
EMERGENCY_KEYWORDS = ['救护', '警察', '消防', '施救', '拖车', 'ambulance', 'police', 'fire']


class VehicleDetector:
    """车辆检测器 - 基于 YOLOv8"""

    def __init__(self, model_name=YOLO_MODEL, confidence=YOLO_CONFIDENCE):
        """
        Args:
            model_name: YOLO 模型名称，首次运行会自动下载
                       n=nano(最快), s=small, m=medium, l=large, x=最准
            confidence: 检测置信度阈值
        """
        self.model = YOLO(model_name)
        self.confidence = confidence

    def detect_vehicles(self, frame):
        """
        检测单帧中的车辆

        Args:
            frame: BGR 图像 (numpy array)

        Returns:
            list of dict, 每个检测结果包含:
            {
                'bbox': [x1, y1, x2, y2],  # 像素坐标
                'class': 'car'|'truck'|'bus'|'motorcycle',
                'confidence': 0.0-1.0,
                'center': (cx, cy),  # 中心点
                'area': int,  # 面积（像素）
                'bbox_norm': [x1, y1, x2, y2],  # 归一化坐标 0-1
            }
        """
        h, w = frame.shape[:2]
        results = self.model(frame, conf=self.confidence, verbose=False)

        detections = []
        for result in results:
            for box in result.boxes:
                cls_id = int(box.cls[0])
                if cls_id not in VEHICLE_CLASSES:
                    continue

                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                cx = (x1 + x2) / 2
                cy = (y1 + y2) / 2

                detections.append({
                    'bbox': [int(x1), int(y1), int(x2), int(y2)],
                    'class': VEHICLE_CLASSES[cls_id],
                    'confidence': float(box.conf[0]),
                    'center': (float(cx), float(cy)),
                    'area': int((x2 - x1) * (y2 - y1)),
                    'bbox_norm': [
                        float(x1 / w), float(y1 / h),
                        float(x2 / w), float(y2 / h)
                    ],
                })

        return detections

    def detect_batch(self, frames):
        """批量检测多帧"""
        return [self.detect_vehicles(f) for f in frames]


def draw_detections(frame, detections, color=(0, 255, 0), thickness=2):
    """在帧上绘制检测结果（调试用）"""
    annotated = frame.copy()
    for det in detections:
        x1, y1, x2, y2 = det['bbox']
        label = f"{det['class']} {det['confidence']:.2f}"
        cv2.rectangle(annotated, (x1, y1), (x2, y2), color, thickness)
        cv2.putText(annotated, label, (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
    return annotated


if __name__ == '__main__':
    import sys
    if len(sys.argv) < 2:
        print("Usage: python vehicle_detector.py <image_or_video>")
        sys.exit(1)

    detector = VehicleDetector()
    path = sys.argv[1]

    if path.lower().endswith(('.mp4', '.avi', '.mov', '.mkv')):
        cap = cv2.VideoCapture(path)
        frame_count = 0
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            if frame_count % 30 == 0:  # 每秒取一帧（假设30fps）
                dets = detector.detect_vehicles(frame)
                print(f"Frame {frame_count}: {len(dets)} vehicles detected")
                for d in dets:
                    print(f"  {d['class']} ({d['confidence']:.2f}) at {d['bbox']}")
            frame_count += 1
        cap.release()
    else:
        frame = cv2.imread(path)
        if frame is None:
            print(f"Cannot read: {path}")
            sys.exit(1)
        dets = detector.detect_vehicles(frame)
        print(f"Detected {len(dets)} vehicles:")
        for d in dets:
            print(f"  {d['class']} ({d['confidence']:.2f}) at {d['bbox']}")

        # 保存标注图
        out = draw_detections(frame, dets)
        out_path = Path(path).stem + '_detected.jpg'
        cv2.imwrite(out_path, out)
        print(f"Saved: {out_path}")
