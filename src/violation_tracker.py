"""
violation_tracker.py - 违法行为追踪器
跨帧追踪车辆，判定是否构成应急车道违法

核心逻辑：
- 单帧检测到车辆在应急车道区域 ≠ 违法
- 需要连续多帧确认（排除正常变道经过）
- 同一车辆需要跨帧关联（简易 IoU 追踪）
"""
# CREATED_BY: highway-reporter MVP
# STATUS: active

import numpy as np
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple
from collections import defaultdict

from .config import (
    EMERGENCY_X_THRESHOLD,
    EVIDENCE_INTERVAL,
    IOU_THRESHOLD,
    MAX_AREA_RATIO,
    MAX_LOST_FRAMES,
    MIN_AREA_RATIO,
    MIN_VIOLATION_FRAMES,
    ROI_Y_RANGE,
)


@dataclass
class TrackedVehicle:
    """被追踪的车辆"""
    track_id: int
    last_bbox: list              # 最近一次检测框 [x1,y1,x2,y2]
    last_bbox_norm: list         # 归一化坐标
    vehicle_class: str           # car/truck/bus
    frames_in_emergency: int = 0  # 在应急车道区域的累计帧数
    frames_total: int = 0        # 总追踪帧数
    last_seen_frame: int = 0     # 最后一次被看到的帧号
    plate_text: Optional[str] = None
    plate_confidence: float = 0.0
    best_frame_idx: int = 0      # 车牌最清晰的帧
    best_plate_crop: Optional[np.ndarray] = None
    violation_confirmed: bool = False
    evidence_frames: list = field(default_factory=list)  # 证据帧索引列表


@dataclass
class ViolationEvent:
    """确认的违法事件"""
    track_id: int
    plate_text: Optional[str]
    plate_confidence: float
    vehicle_class: str
    start_frame: int
    end_frame: int
    duration_frames: int
    evidence_frame_indices: list  # 关键帧索引（用于截图取证）
    confidence: float            # 综合置信度


class ViolationTracker:
    """
    违法行为追踪器

    工作流程：
    1. 接收每帧的车辆检测结果
    2. 用 IoU 关联跨帧的同一车辆
    3. 判断车辆位置是否在应急车道区域
    4. 连续多帧在应急车道 → 确认违法
    5. 输出违法事件 + 证据帧
    """

    def __init__(
        self,
        # 应急车道判定区域（归一化 x 坐标）
        emergency_x_threshold: float = EMERGENCY_X_THRESHOLD,
        # ROI（只看路面区域）
        roi_y_range: Tuple[float, float] = ROI_Y_RANGE,
        # 车辆大小过滤
        min_area_ratio: float = MIN_AREA_RATIO,
        max_area_ratio: float = MAX_AREA_RATIO,
        # 追踪参数
        iou_threshold: float = IOU_THRESHOLD,
        max_lost_frames: int = MAX_LOST_FRAMES,
        # 违法判定
        min_violation_frames: int = MIN_VIOLATION_FRAMES,
        # 证据采集
        evidence_interval: int = EVIDENCE_INTERVAL,
    ):
        self.emergency_x_threshold = emergency_x_threshold
        self.roi_y_range = roi_y_range
        self.min_area_ratio = min_area_ratio
        self.max_area_ratio = max_area_ratio
        self.iou_threshold = iou_threshold
        self.max_lost_frames = max_lost_frames
        self.min_violation_frames = min_violation_frames
        self.evidence_interval = evidence_interval

        self.tracks: Dict[int, TrackedVehicle] = {}
        self.next_track_id = 1
        self.frame_idx = 0
        self.violations: List[ViolationEvent] = []

    def _iou(self, box1, box2):
        """计算两个框的 IoU"""
        x1 = max(box1[0], box2[0])
        y1 = max(box1[1], box2[1])
        x2 = min(box1[2], box2[2])
        y2 = min(box1[3], box2[3])

        inter = max(0, x2 - x1) * max(0, y2 - y1)
        area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
        area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
        union = area1 + area2 - inter

        return inter / union if union > 0 else 0

    def _is_in_roi(self, det: dict, frame_h: int) -> bool:
        """检查是否在感兴趣区域"""
        _, cy = det['center']
        cy_norm = cy / frame_h
        return self.roi_y_range[0] <= cy_norm <= self.roi_y_range[1]

    def _is_valid_size(self, det: dict, frame_area: int) -> bool:
        """检查大小是否合理"""
        ratio = det['area'] / frame_area
        return self.min_area_ratio <= ratio <= self.max_area_ratio

    def _is_in_emergency_lane(self, det: dict) -> bool:
        """判断车辆是否在应急车道区域"""
        x1_n, _, x2_n, _ = det['bbox_norm']
        cx_norm = (x1_n + x2_n) / 2
        return cx_norm >= self.emergency_x_threshold

    def _match_detections(self, detections: list) -> Dict[int, dict]:
        """
        用 IoU 将当前帧检测结果匹配到已有追踪

        Returns:
            {track_id: detection} 匹配结果
            未匹配的检测会创建新追踪
        """
        matched = {}
        unmatched_dets = list(range(len(detections)))

        if self.tracks:
            # 计算所有 track-detection 对的 IoU
            track_ids = list(self.tracks.keys())
            iou_matrix = np.zeros((len(track_ids), len(detections)))

            for i, tid in enumerate(track_ids):
                for j, det in enumerate(detections):
                    iou_matrix[i, j] = self._iou(
                        self.tracks[tid].last_bbox, det['bbox']
                    )

            # 贪心匹配（简单有效，比匈牙利算法快）
            while True:
                if iou_matrix.size == 0:
                    break
                max_iou = iou_matrix.max()
                if max_iou < self.iou_threshold:
                    break
                i, j = np.unravel_index(iou_matrix.argmax(), iou_matrix.shape)
                matched[track_ids[i]] = detections[j]
                if j in unmatched_dets:
                    unmatched_dets.remove(j)
                iou_matrix[i, :] = 0
                iou_matrix[:, j] = 0

        # 未匹配的检测 → 新追踪
        for j in unmatched_dets:
            det = detections[j]
            tid = self.next_track_id
            self.next_track_id += 1
            self.tracks[tid] = TrackedVehicle(
                track_id=tid,
                last_bbox=det['bbox'],
                last_bbox_norm=det['bbox_norm'],
                vehicle_class=det['class'],
                last_seen_frame=self.frame_idx,
            )
            matched[tid] = det

        return matched

    def update(self, detections: list, frame_shape: tuple,
               plate_results: Optional[Dict[int, dict]] = None):
        """
        更新追踪器（每帧调用一次）

        Args:
            detections: VehicleDetector 的检测结果
            frame_shape: (h, w, c)
            plate_results: 可选，{detection_index: plate_result}

        Returns:
            list of new ViolationEvent (本帧新确认的违法)
        """
        self.frame_idx += 1
        h, w = frame_shape[:2]
        frame_area = h * w

        # 过滤：ROI + 大小
        valid_dets = [
            d for d in detections
            if self._is_in_roi(d, h) and self._is_valid_size(d, frame_area)
        ]

        # 匹配追踪
        matched = self._match_detections(valid_dets)

        # 更新每个追踪
        new_violations = []
        for tid, det in matched.items():
            track = self.tracks[tid]
            track.last_bbox = det['bbox']
            track.last_bbox_norm = det['bbox_norm']
            track.last_seen_frame = self.frame_idx
            track.frames_total += 1

            # 检查是否在应急车道
            if self._is_in_emergency_lane(det):
                track.frames_in_emergency += 1

                # 采集证据帧
                if track.frames_in_emergency % self.evidence_interval == 1:
                    track.evidence_frames.append(self.frame_idx)

                # 判定违法
                if (track.frames_in_emergency >= self.min_violation_frames
                        and not track.violation_confirmed):
                    track.violation_confirmed = True
                    violation = ViolationEvent(
                        track_id=tid,
                        plate_text=track.plate_text,
                        plate_confidence=track.plate_confidence,
                        vehicle_class=track.vehicle_class,
                        start_frame=self.frame_idx - track.frames_in_emergency,
                        end_frame=self.frame_idx,
                        duration_frames=track.frames_in_emergency,
                        evidence_frame_indices=list(track.evidence_frames),
                        confidence=min(1.0, track.frames_in_emergency / (self.min_violation_frames * 2)),
                    )
                    self.violations.append(violation)
                    new_violations.append(violation)

        # 清理丢失的追踪
        lost_ids = [
            tid for tid, t in self.tracks.items()
            if self.frame_idx - t.last_seen_frame > self.max_lost_frames
        ]
        for tid in lost_ids:
            del self.tracks[tid]

        return new_violations

    def update_plate(self, track_id: int, plate_text: str, confidence: float):
        """更新追踪车辆的车牌信息"""
        if track_id in self.tracks:
            track = self.tracks[track_id]
            if confidence > track.plate_confidence:
                track.plate_text = plate_text
                track.plate_confidence = confidence
                # 同步更新已确认的违法事件
                for v in self.violations:
                    if v.track_id == track_id:
                        v.plate_text = plate_text
                        v.plate_confidence = confidence

    def get_active_suspects(self) -> List[TrackedVehicle]:
        """获取当前在应急车道区域的车辆"""
        return [
            t for t in self.tracks.values()
            if t.frames_in_emergency > 0
            and self.frame_idx - t.last_seen_frame <= 3
        ]

    def get_violations(self) -> List[ViolationEvent]:
        """获取所有已确认的违法事件"""
        return list(self.violations)

    def reset(self):
        """重置（处理新视频时调用）"""
        self.tracks.clear()
        self.next_track_id = 1
        self.frame_idx = 0
        self.violations.clear()
