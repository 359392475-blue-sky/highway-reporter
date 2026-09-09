"""
evidence.py - 证据包生成模块
根据违法事件生成符合各平台要求的证据包

证据包内容（以深圳随手e拍为标准）：
1. 车牌清晰截图 × 1
2. 违法行为全景截图 × 2（不同时间点，证明持续违法）
3. 视频片段 15-30秒（含违法全过程）
4. 元数据（时间、地点、车牌号、车辆信息）
"""
# CREATED_BY: highway-reporter MVP
# STATUS: active

import cv2
import json
import subprocess
from pathlib import Path
from typing import Optional, List
from dataclasses import dataclass

from .config import JPEG_QUALITY
from .preflight import find_ffmpeg


@dataclass
class EvidencePackage:
    """证据包"""
    violation_id: int
    plate_text: Optional[str]
    vehicle_class: str
    # 文件路径
    plate_screenshot: Optional[str] = None     # 车牌截图
    scene_screenshots: List[str] = None        # 全景截图（2-3张）
    video_clip: Optional[str] = None           # 视频片段
    metadata_file: Optional[str] = None        # 元数据JSON
    # 状态
    quality_score: float = 0.0                 # 证据质量评分 0-1
    issues: List[str] = None                   # 质量问题
    plate_confidence: float = 0.0

    def __post_init__(self):
        if self.scene_screenshots is None:
            self.scene_screenshots = []
        if self.issues is None:
            self.issues = []


class EvidenceBuilder:
    """证据包生成器"""

    def __init__(self, output_dir: str = './output'):
        self.output_dir = Path(output_dir)

    def extract_video_clip(
        self,
        video_path: str,
        start_sec: float,
        end_sec: float,
        output_path: str,
        padding_sec: float = 3.0,
        max_duration: float = 30.0,
    ) -> Optional[str]:
        """
        用 FFmpeg 截取视频片段（无损截取，不重编码）

        Args:
            video_path: 原始视频路径
            start_sec: 违法开始时间（秒）
            end_sec: 违法结束时间（秒）
            output_path: 输出路径
            padding_sec: 前后各扩展秒数
            max_duration: 最大时长

        Returns:
            输出文件路径，失败返回 None
        """
        # 计算截取范围（前后各扩展几秒，确保包含完整违法过程）
        clip_start = max(0, start_sec - padding_sec)
        clip_duration = min(
            end_sec - start_sec + padding_sec * 2,
            max_duration
        )

        ffmpeg = find_ffmpeg()
        if not ffmpeg:
            return None

        out = Path(output_path)
        attempts = [
            [ffmpeg, '-y', '-ss', str(clip_start), '-i', video_path,
             '-t', str(clip_duration), '-c', 'copy', '-avoid_negative_ts', '1',
             str(out)],
            [ffmpeg, '-y', '-ss', str(clip_start), '-i', video_path,
             '-t', str(clip_duration), '-c:v', 'libx264', '-preset', 'veryfast',
             '-crf', '23', '-c:a', 'aac', '-movflags', '+faststart', str(out)],
        ]
        for cmd in attempts:
            try:
                proc = subprocess.run(cmd, capture_output=True, timeout=60)
            except (subprocess.TimeoutExpired, OSError):
                continue
            if proc.returncode == 0 and out.is_file() and out.stat().st_size > 0:
                return str(out)
            if out.exists():
                out.unlink()

        return None

    def select_best_frames(
        self,
        video_path: str,
        frame_indices: List[int],
        fps: float,
    ) -> List[tuple]:
        """
        从视频中提取指定帧，并按清晰度排序

        Returns:
            [(frame_idx, frame, sharpness_score), ...] 按清晰度降序
        """
        cap = cv2.VideoCapture(video_path)
        frames = []

        for fi in frame_indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, fi)
            ret, frame = cap.read()
            if ret:
                # 用拉普拉斯算子衡量清晰度
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                sharpness = cv2.Laplacian(gray, cv2.CV_64F).var()
                frames.append((fi, frame, sharpness))

        cap.release()

        # 按清晰度降序排列
        frames.sort(key=lambda x: x[2], reverse=True)
        return frames

    def build_evidence(
        self,
        video_path: str,
        violation,  # ViolationEvent
        fps: float,
        evidence_dir: Optional[str] = None,
        plate_text: Optional[str] = None,
        plate_confidence: Optional[float] = None,
    ) -> EvidencePackage:
        """
        为一起违法事件生成完整证据包

        Args:
            video_path: 原始视频
            violation: ViolationEvent
            fps: 视频帧率
            evidence_dir: 证据输出目录
            plate_text: L2 精读车牌（优先于 ViolationEvent）
            plate_confidence: L2 车牌置信度

        Returns:
            EvidencePackage
        """
        if evidence_dir is None:
            evidence_dir = self.output_dir / f"violation_{violation.track_id}"
        evidence_dir = Path(evidence_dir)
        evidence_dir.mkdir(parents=True, exist_ok=True)

        resolved_plate = plate_text if plate_text is not None else violation.plate_text
        resolved_conf = (
            plate_confidence if plate_confidence is not None
            else (violation.plate_confidence or 0.0)
        )

        pkg = EvidencePackage(
            violation_id=violation.track_id,
            plate_text=resolved_plate,
            vehicle_class=violation.vehicle_class,
            plate_confidence=resolved_conf,
        )

        # 1. 提取证据帧
        frames = self.select_best_frames(
            video_path,
            violation.evidence_frame_indices,
            fps,
        )

        if not frames:
            pkg.issues.append("无法提取证据帧")
            return pkg

        # 2. 保存全景截图（选 2-3 张，时间间隔尽量大）
        if len(frames) >= 2:
            # 取最清晰的和时间跨度最大的
            selected = [frames[0]]  # 最清晰
            # 找一张时间上距离最远的
            best_dist = 0
            best_frame = None
            for f in frames[1:]:
                dist = abs(f[0] - frames[0][0])
                if dist > best_dist:
                    best_dist = dist
                    best_frame = f
            if best_frame:
                selected.append(best_frame)

            for i, (fi, frame, _) in enumerate(selected):
                path = str(evidence_dir / f"scene_{i+1}_frame{fi}.jpg")
                cv2.imwrite(path, frame, [cv2.IMWRITE_JPEG_QUALITY, JPEG_QUALITY])
                pkg.scene_screenshots.append(path)
        elif frames:
            path = str(evidence_dir / f"scene_1_frame{frames[0][0]}.jpg")
            cv2.imwrite(path, frames[0][1], [cv2.IMWRITE_JPEG_QUALITY, JPEG_QUALITY])
            pkg.scene_screenshots.append(path)
            pkg.issues.append("只有1张全景截图（建议至少2张）")

        # 3. 视频片段
        start_sec = violation.start_frame / fps
        end_sec = violation.end_frame / fps
        clip_path = str(evidence_dir / "clip.mp4")
        result = self.extract_video_clip(
            video_path, start_sec, end_sec, clip_path
        )
        if result:
            pkg.video_clip = result
        else:
            if find_ffmpeg():
                pkg.issues.append("视频片段截取失败")
            else:
                pkg.issues.append("视频片段截取失败（未找到可用的 FFmpeg）")

        # 4. 元数据
        metadata = {
            'plate_text': resolved_plate,
            'plate_confidence': resolved_conf,
            'vehicle_class': violation.vehicle_class,
            'start_sec': start_sec,
            'end_sec': end_sec,
            'duration_sec': (end_sec - start_sec),
            'evidence_screenshots': pkg.scene_screenshots,
            'video_clip': pkg.video_clip,
            'analysis_confidence': violation.confidence,
        }
        meta_path = str(evidence_dir / "metadata.json")
        with open(meta_path, 'w', encoding='utf-8') as f:
            json.dump(metadata, f, ensure_ascii=False, indent=2)
        pkg.metadata_file = meta_path

        # 5. 质量评分
        pkg.quality_score = self._assess_quality(pkg, violation)

        return pkg

    def _assess_quality(self, pkg: EvidencePackage, violation) -> float:
        """评估证据包质量"""
        score = 0.0

        # 有车牌 +0.4
        plate_conf = pkg.plate_confidence or getattr(violation, 'plate_confidence', 0) or 0
        if pkg.plate_text and plate_conf > 0.6:
            score += 0.4
        elif pkg.plate_text:
            score += 0.2

        # 有 >=2 张截图 +0.2
        if len(pkg.scene_screenshots) >= 2:
            score += 0.2
        elif len(pkg.scene_screenshots) >= 1:
            score += 0.1

        # 有视频片段 +0.2
        if pkg.video_clip:
            score += 0.2

        # 持续时间够长 +0.2
        duration = violation.duration_frames
        if duration >= 30:
            score += 0.2
        elif duration >= 15:
            score += 0.1

        return min(1.0, score)
