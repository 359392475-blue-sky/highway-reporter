"""
pipeline.py - 旧版视频分析流水线（已废弃）

现行入口是 full_pipeline.py（YOLO + 云端精读）。
本模块仍走本地 PaddleOCR，仅保留给 analyze.py。
"""
# CREATED_BY: highway-reporter MVP
# STATUS: deprecated

import cv2
import json
import time
import numpy as np
from pathlib import Path
from dataclasses import asdict
from typing import Optional

from .vehicle_detector import VehicleDetector, draw_detections
from .lane_analyzer import EmergencyLaneAnalyzer, EmergencyLaneConfig, draw_lane_analysis
from .plate_reader import PlateReader
from .violation_tracker import ViolationTracker, ViolationEvent
from .report_helper import build_report_package, ReportServer


class AnalysisResult:
    """单个视频的分析结果"""

    def __init__(self, video_path: str):
        self.video_path = video_path
        self.violations = []
        self.total_frames = 0
        self.fps = 0
        self.duration_sec = 0
        self.processing_time_sec = 0
        self.evidence_dir = None

    def add_violation(self, event: ViolationEvent):
        self.violations.append(event)

    def summary(self) -> str:
        lines = [
            f"=== 分析报告 ===",
            f"视频: {self.video_path}",
            f"时长: {self.duration_sec:.1f}秒 ({self.total_frames}帧 @{self.fps}fps)",
            f"处理耗时: {self.processing_time_sec:.1f}秒",
            f"发现违法: {len(self.violations)}起",
        ]
        for i, v in enumerate(self.violations):
            plate = v.plate_text or '未识别'
            lines.append(
                f"\n  [{i+1}] 车牌: {plate} (置信度: {v.plate_confidence:.0%})"
                f"\n      车型: {v.vehicle_class}"
                f"\n      持续: {v.duration_frames}帧 "
                f"({v.duration_frames/self.fps:.1f}秒)"
                f"\n      时间段: 第{v.start_frame/self.fps:.1f}秒 - "
                f"第{v.end_frame/self.fps:.1f}秒"
                f"\n      证据帧数: {len(v.evidence_frame_indices)}张"
            )
        if not self.violations:
            lines.append("\n  未发现应急车道违法行为")
        if self.evidence_dir:
            lines.append(f"\n证据目录: {self.evidence_dir}")
        return '\n'.join(lines)


class Pipeline:
    """
    视频分析流水线

    使用方式:
        pipeline = Pipeline(output_dir='./output')
        result = pipeline.analyze_video('highway_video.mp4')
        print(result.summary())
    """

    def __init__(
        self,
        output_dir: str = './output',
        # 车辆检测参数
        yolo_model: str = 'yolov8n.pt',
        detection_confidence: float = 0.45,
        # 应急车道判定参数
        emergency_x_threshold: float = 0.68,
        # 违法确认参数
        min_violation_frames: int = 15,
        # 处理参数
        frame_skip: int = 2,         # 每隔几帧处理一次（加速）
        plate_check_interval: int = 5,  # 每隔几帧做一次车牌识别（OCR 较慢）
        save_evidence: bool = True,   # 是否保存证据截图
        save_debug_video: bool = False,  # 是否保存标注视频（调试）
    ):
        self.output_dir = Path(output_dir)
        self.frame_skip = frame_skip
        self.plate_check_interval = plate_check_interval
        self.save_evidence = save_evidence
        self.save_debug_video = save_debug_video

        # 初始化模块
        print("[Pipeline] 初始化车辆检测器...")
        self.detector = VehicleDetector(
            model_name=yolo_model,
            confidence=detection_confidence,
        )

        print("[Pipeline] 初始化应急车道分析器...")
        self.lane_analyzer = EmergencyLaneAnalyzer(
            config=EmergencyLaneConfig(
                emergency_threshold_x=emergency_x_threshold,
            )
        )

        print("[Pipeline] 初始化车牌识别器...")
        self.plate_reader = PlateReader()

        print("[Pipeline] 初始化违法追踪器...")
        self.tracker = ViolationTracker(
            emergency_x_threshold=emergency_x_threshold,
            min_violation_frames=min_violation_frames,
        )

    def analyze_video(self, video_path: str,
                      max_frames: Optional[int] = None) -> AnalysisResult:
        """
        分析单个视频文件

        Args:
            video_path: 视频文件路径
            max_frames: 最大处理帧数（调试用，None=处理全部）

        Returns:
            AnalysisResult
        """
        video_path = str(video_path)
        result = AnalysisResult(video_path)

        # 打开视频
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise FileNotFoundError(f"无法打开视频: {video_path}")

        result.fps = cap.get(cv2.CAP_PROP_FPS) or 30
        result.total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        result.duration_sec = result.total_frames / result.fps

        print(f"\n[Pipeline] 开始分析: {video_path}")
        print(f"  分辨率: {int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))}x"
              f"{int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))}")
        print(f"  时长: {result.duration_sec:.1f}s, "
              f"帧数: {result.total_frames}, FPS: {result.fps:.0f}")
        print(f"  处理间隔: 每{self.frame_skip}帧")

        # 准备输出目录
        video_name = Path(video_path).stem
        evidence_dir = self.output_dir / video_name
        if self.save_evidence:
            evidence_dir.mkdir(parents=True, exist_ok=True)
            result.evidence_dir = str(evidence_dir)

        # 准备 debug 视频写入器
        debug_writer = None
        if self.save_debug_video:
            w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            debug_path = str(evidence_dir / f"{video_name}_debug.mp4")
            debug_writer = cv2.VideoWriter(
                debug_path,
                cv2.VideoWriter_fourcc(*'mp4v'),
                result.fps / self.frame_skip,
                (w, h)
            )

        # 重置追踪器
        self.tracker.reset()
        self.lane_analyzer.reset()

        # 证据帧缓存（保存最近的帧用于证据截取）
        frame_buffer = {}  # frame_idx -> frame
        buffer_max_size = 100

        start_time = time.time()
        frame_idx = 0
        processed = 0

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            frame_idx += 1
            if max_frames and frame_idx > max_frames:
                break

            # 帧跳过
            if frame_idx % self.frame_skip != 0:
                continue

            processed += 1

            # 1. 车辆检测
            detections = self.detector.detect_vehicles(frame)

            # 2. 追踪 + 应急车道判定
            new_violations = self.tracker.update(
                detections, frame.shape
            )

            # 3. 车牌识别（间隔执行，OCR 较慢）
            if processed % self.plate_check_interval == 0:
                suspects = self.tracker.get_active_suspects()
                for track in suspects:
                    if track.plate_confidence < 0.8:  # 还没有高置信度车牌
                        plate = self.plate_reader.read_plate(
                            frame, track.last_bbox
                        )
                        if plate and plate['confidence'] > track.plate_confidence:
                            self.tracker.update_plate(
                                track.track_id,
                                plate['plate_text'],
                                plate['confidence'],
                            )

            # 4. 保存证据帧
            if self.save_evidence:
                # 缓存当前帧
                frame_buffer[frame_idx] = frame.copy()
                # 清理旧帧
                if len(frame_buffer) > buffer_max_size:
                    oldest = min(frame_buffer.keys())
                    del frame_buffer[oldest]

                # 新发现违法 → 立即保存证据
                for v in new_violations:
                    self._save_evidence(v, frame_buffer, evidence_dir,
                                        result.fps)
                    result.add_violation(v)

            # 5. Debug 视频
            if debug_writer:
                annotated = self._draw_debug(frame, detections)
                debug_writer.write(annotated)

            # 进度
            if processed % 100 == 0:
                pct = frame_idx / result.total_frames * 100
                elapsed = time.time() - start_time
                eta = elapsed / processed * (result.total_frames / self.frame_skip - processed)
                print(f"  进度: {pct:.0f}% ({processed}帧处理) "
                      f"ETA: {eta:.0f}s "
                      f"违法: {len(self.tracker.violations)}起")

        cap.release()
        if debug_writer:
            debug_writer.release()

        # 处理追踪结束后仍在应急车道的车辆
        # （视频结束时可能有未确认的违法）

        result.processing_time_sec = time.time() - start_time

        # 保存分析报告 JSON
        if self.save_evidence:
            report = {
                'video': video_path,
                'duration_sec': result.duration_sec,
                'total_frames': result.total_frames,
                'fps': result.fps,
                'processing_time_sec': result.processing_time_sec,
                'violations': [
                    {
                        'track_id': v.track_id,
                        'plate_text': v.plate_text,
                        'plate_confidence': v.plate_confidence,
                        'vehicle_class': v.vehicle_class,
                        'start_sec': v.start_frame / result.fps,
                        'end_sec': v.end_frame / result.fps,
                        'duration_sec': v.duration_frames / result.fps,
                        'confidence': v.confidence,
                    }
                    for v in result.violations
                ],
            }
            report_path = evidence_dir / 'report.json'
            with open(report_path, 'w', encoding='utf-8') as f:
                json.dump(report, f, ensure_ascii=False, indent=2)

        print(f"\n{result.summary()}")
        return result

    def _save_evidence(self, violation: ViolationEvent,
                       frame_buffer: dict, evidence_dir: Path,
                       fps: float):
        """保存违法证据（截图）"""
        v_dir = evidence_dir / f"violation_{violation.track_id}"
        v_dir.mkdir(exist_ok=True)

        saved = 0
        for fi in violation.evidence_frame_indices:
            if fi in frame_buffer:
                filename = f"frame_{fi}_t{fi/fps:.1f}s.jpg"
                cv2.imwrite(str(v_dir / filename), frame_buffer[fi])
                saved += 1

        # 也保存最近的帧作为额外证据
        if frame_buffer:
            latest_fi = max(frame_buffer.keys())
            latest_frame = frame_buffer[latest_fi]
            cv2.imwrite(
                str(v_dir / f"latest_{latest_fi}_t{latest_fi/fps:.1f}s.jpg"),
                latest_frame
            )

        print(f"  [证据] 违法#{violation.track_id} 保存{saved+1}张截图 → {v_dir}")

    def _draw_debug(self, frame, detections):
        """绘制调试标注"""
        annotated = draw_detections(frame, detections)
        suspects = self.tracker.get_active_suspects()
        if suspects:
            for t in suspects:
                x1, y1, x2, y2 = t.last_bbox
                color = (0, 0, 255) if t.violation_confirmed else (0, 165, 255)
                cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 3)
                label = f"{'VIOLATION' if t.violation_confirmed else 'SUSPECT'}"
                if t.plate_text:
                    label += f" {t.plate_text}"
                label += f" ({t.frames_in_emergency}f)"
                cv2.putText(annotated, label, (x1, y1 - 25),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
        return annotated


def analyze_video(video_path: str, output_dir: str = './output',
                  **kwargs) -> AnalysisResult:
    """便捷函数：分析单个视频"""
    pipeline = Pipeline(output_dir=output_dir, **kwargs)
    return pipeline.analyze_video(video_path)
