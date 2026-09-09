"""
lane_analyzer.py - 应急车道分析模块
判断车辆是否在应急车道行驶

策略：不完全依赖车道线检测（标线磨损/遮挡常见），
而是结合多种信号综合判断：
1. 画面位置（应急车道在最右侧）
2. 车辆与车流的相对位置
3. 可选：车道线辅助确认
"""
# CREATED_BY: highway-reporter MVP
# STATUS: deprecated (logic lives in ViolationTracker; kept for old pipeline)

import cv2
import numpy as np
from dataclasses import dataclass
from typing import List, Optional, Tuple

from .config import EMERGENCY_X_THRESHOLD, MIN_VIOLATION_FRAMES, ROI_Y_RANGE


@dataclass
class LaneRegion:
    """车道区域定义"""
    name: str           # 'emergency', 'driving', 'overtaking'
    x_start_norm: float  # 归一化 x 起始 (0-1)
    x_end_norm: float    # 归一化 x 结束 (0-1)


@dataclass
class EmergencyLaneConfig:
    """应急车道检测配置

    行车记录仪安装在车辆前挡风中央，
    因此画面是以驾驶者视角拍摄的前方道路。
    应急车道在画面右侧。

    这些参数需要根据实际视频标定，
    默认值是一个粗略的起点。
    """
    # 应急车道在画面中的大致位置（归一化 x 坐标）
    # 0.0 = 画面最左, 1.0 = 画面最右
    emergency_lane_x_start: float = 0.65   # 应急车道左边界（保守估计）
    emergency_lane_x_end: float = 1.0      # 应急车道右边界

    # ROI (感兴趣区域) — 只分析道路区域，忽略天空/仪表盘
    roi_y_start: float = ROI_Y_RANGE[0]
    roi_y_end: float = ROI_Y_RANGE[1]

    # 车辆大小过滤（太小的车太远，车牌读不清）
    min_vehicle_area_ratio: float = 0.005  # 最小车辆面积占画面比例
    max_vehicle_area_ratio: float = 0.4    # 最大（太大说明离太近/检测错误）

    # 判定阈值
    # 车辆中心点 x 坐标超过此值，认为在应急车道
    emergency_threshold_x: float = EMERGENCY_X_THRESHOLD

    # 车辆需要在应急车道持续多少帧才判定为违法（非临时经过）
    min_consecutive_frames: int = MIN_VIOLATION_FRAMES


class EmergencyLaneAnalyzer:
    """应急车道占用分析器"""

    def __init__(self, config: Optional[EmergencyLaneConfig] = None):
        self.config = config or EmergencyLaneConfig()
        # 车辆追踪状态: vehicle_id -> {'frames_in_lane': int, 'last_seen': int}
        self.tracking = {}
        self.frame_idx = 0

    def is_in_roi(self, detection: dict, frame_shape: tuple) -> bool:
        """检查检测结果是否在 ROI 内"""
        h, w = frame_shape[:2]
        _, cy = detection['center']
        cy_norm = cy / h

        return (self.config.roi_y_start <= cy_norm <= self.config.roi_y_end)

    def is_valid_size(self, detection: dict, frame_shape: tuple) -> bool:
        """检查车辆大小是否合理"""
        h, w = frame_shape[:2]
        area_ratio = detection['area'] / (h * w)
        return (self.config.min_vehicle_area_ratio <= area_ratio
                <= self.config.max_vehicle_area_ratio)

    def is_in_emergency_lane(self, detection: dict) -> Tuple[bool, float]:
        """
        判断单个车辆是否在应急车道区域

        Args:
            detection: 车辆检测结果

        Returns:
            (is_in_lane, confidence)
            - is_in_lane: 是否在应急车道
            - confidence: 置信度 0-1
        """
        x1_n, y1_n, x2_n, y2_n = detection['bbox_norm']
        cx_norm = (x1_n + x2_n) / 2

        # 核心判断：车辆中心是否在应急车道区域
        threshold = self.config.emergency_threshold_x

        if cx_norm >= threshold:
            # 在应急车道区域，计算置信度
            # 越靠右越确定
            conf = min(1.0, (cx_norm - threshold) / (1.0 - threshold) * 2)
            return True, conf
        else:
            return False, 0.0

    def analyze_frame(self, detections: list, frame_shape: tuple) -> list:
        """
        分析单帧中所有车辆的应急车道占用情况

        Args:
            detections: 来自 VehicleDetector 的检测结果列表
            frame_shape: 帧的 shape (h, w, c)

        Returns:
            list of dict, 可能的违法车辆:
            {
                'detection': {...},  # 原始检测
                'in_emergency_lane': True,
                'lane_confidence': 0.0-1.0,
            }
        """
        self.frame_idx += 1
        suspects = []

        for det in detections:
            # 过滤：ROI 内 + 大小合理
            if not self.is_in_roi(det, frame_shape):
                continue
            if not self.is_valid_size(det, frame_shape):
                continue

            in_lane, confidence = self.is_in_emergency_lane(det)

            if in_lane:
                suspects.append({
                    'detection': det,
                    'in_emergency_lane': True,
                    'lane_confidence': confidence,
                    'frame_idx': self.frame_idx,
                })

        return suspects

    def reset(self):
        """重置状态（处理新视频时调用）"""
        self.tracking = {}
        self.frame_idx = 0


def detect_lane_lines(frame, roi_y_start=0.4, roi_y_end=0.85):
    """
    基于边缘检测的车道线检测（辅助参考）

    这是一个简单的基于 Canny + HoughLines 的实现。
    不作为主要判断依据，仅作为辅助信号。

    Returns:
        lines: 检测到的线段列表
        mask: 边缘检测结果
    """
    h, w = frame.shape[:2]
    y1 = int(h * roi_y_start)
    y2 = int(h * roi_y_end)
    roi = frame[y1:y2, :]

    # 转灰度 + 高斯模糊
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)

    # Canny 边缘检测
    edges = cv2.Canny(blur, 50, 150)

    # 霍夫变换检测线段
    lines = cv2.HoughLinesP(
        edges,
        rho=1,
        theta=np.pi / 180,
        threshold=50,
        minLineLength=50,
        maxLineGap=30
    )

    return lines, edges


def draw_lane_analysis(frame, suspects, config=None):
    """绘制应急车道分析结果（调试用）"""
    if config is None:
        config = EmergencyLaneConfig()

    annotated = frame.copy()
    h, w = frame.shape[:2]

    # 画应急车道区域（半透明红色）
    overlay = annotated.copy()
    x_start = int(w * config.emergency_threshold_x)
    y_start = int(h * config.roi_y_start)
    y_end = int(h * config.roi_y_end)
    cv2.rectangle(overlay, (x_start, y_start), (w, y_end), (0, 0, 255), -1)
    cv2.addWeighted(overlay, 0.15, annotated, 0.85, 0, annotated)

    # 画应急车道边界线
    cv2.line(annotated, (x_start, y_start), (x_start, y_end), (0, 0, 255), 2)
    cv2.putText(annotated, "Emergency Lane", (x_start + 5, y_start + 20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

    # 标记嫌疑车辆
    for sus in suspects:
        det = sus['detection']
        x1, y1_, x2, y2_ = det['bbox']
        conf = sus['lane_confidence']
        color = (0, 0, 255) if conf > 0.5 else (0, 165, 255)
        cv2.rectangle(annotated, (x1, y1_), (x2, y2_), color, 3)
        label = f"VIOLATION? {conf:.0%}"
        cv2.putText(annotated, label, (x1, y1_ - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

    return annotated
