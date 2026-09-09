"""
highway_filter.py - 高速公路路段自动筛选模块
从一堆行车记录仪文件中快速识别高速公路片段

策略（优先级从高到低）：
1. GPS 元数据：速度 > 60km/h 且持续 → 高速
2. 视频帧分析：每段视频采样几帧，用本地特征判断
   - 护栏检测（高速特有的波形护栏）
   - 车道数 ≥ 3
   - 无红绿灯/人行道/自行车
   - 中央隔离带
3. 云端辅助（可选）：不确定的用大模型看一眼
"""
# CREATED_BY: highway-reporter MVP
# STATUS: active

import cv2
import os
import json
import numpy as np
from pathlib import Path
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass, field

from .config import HIGHWAY_SCORE_MAYBE, HIGHWAY_SCORE_YES, VIDEO_EXTENSIONS


@dataclass
class VideoSegment:
    """视频片段信息"""
    path: str
    filename: str
    duration_sec: float = 0
    width: int = 0
    height: int = 0
    fps: float = 0
    size_mb: float = 0
    # 高速判定
    is_highway: Optional[bool] = None
    highway_confidence: float = 0.0
    highway_reason: str = ''
    # GPS
    has_gps: bool = False
    avg_speed_kmh: float = 0
    gps_lat: Optional[float] = None
    gps_lon: Optional[float] = None
    # 分析
    frame_scores: List[float] = field(default_factory=list)


def analyze_frame_highway_features(frame: np.ndarray) -> Tuple[float, Dict]:
    """
    分析单帧是否是高速公路场景

    返回 (score, details)
    score: 0.0-1.0, 越高越可能是高速
    """
    h, w = frame.shape[:2]
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    details = {}
    score = 0.0

    # === 1. 道路区域占比（高速公路路面占画面比例大）===
    # 路面通常是灰色低饱和度区域
    road_mask = cv2.inRange(hsv,
                            np.array([0, 0, 40]),
                            np.array([180, 60, 180]))
    # 只看下半部分（路面区域）
    road_lower = road_mask[int(h * 0.5):int(h * 0.85), :]
    road_ratio = np.count_nonzero(road_lower) / road_lower.size
    details['road_ratio'] = round(road_ratio, 2)
    if road_ratio > 0.3:
        score += 0.15

    # === 2. 车道线检测（高速公路有清晰的白色/黄色车道线）===
    # 提取路面区域的边缘
    road_roi = gray[int(h * 0.4):int(h * 0.85), :]
    edges = cv2.Canny(road_roi, 50, 150)
    lines = cv2.HoughLinesP(edges, 1, np.pi / 180,
                            threshold=40, minLineLength=30, maxLineGap=20)
    num_lines = len(lines) if lines is not None else 0
    details['lane_lines'] = num_lines

    # 过滤近似垂直的线（车道线在透视下是斜线，不是垂直的）
    if lines is not None:
        valid_lines = 0
        for line in lines:
            x1, y1, x2, y2 = line[0]
            if x2 - x1 == 0:
                continue
            angle = abs(np.arctan2(y2 - y1, x2 - x1) * 180 / np.pi)
            # 车道线角度通常在 20-80 度
            if 15 < angle < 85:
                valid_lines += 1
        details['valid_lane_lines'] = valid_lines
        if valid_lines >= 4:
            score += 0.2
        elif valid_lines >= 2:
            score += 0.1

    # === 3. 护栏检测（高速公路两侧有金属护栏）===
    # 护栏通常在画面右侧边缘，是亮色的水平长条
    right_strip = gray[int(h * 0.3):int(h * 0.8), int(w * 0.85):]
    if right_strip.size > 0:
        right_edges = cv2.Canny(right_strip, 30, 100)
        right_line_density = np.count_nonzero(right_edges) / right_edges.size
        details['right_edge_density'] = round(right_line_density, 3)
        if right_line_density > 0.05:
            score += 0.1

    # === 4. 天空区域（高速公路开阔，天空占比大）===
    sky_region = hsv[0:int(h * 0.3), int(w * 0.2):int(w * 0.8)]
    if sky_region.size > 0:
        sky_bright = np.mean(sky_region[:, :, 2])
        details['sky_brightness'] = round(float(sky_bright), 1)
        if sky_bright > 150:
            score += 0.1

    # === 5. 车辆检测（高速上前方车辆通常较远较小）===
    # 用简单的连通区域检测代替 YOLO（快速筛选不用精确）
    # 高速公路画面中间区域有多个小目标（远处车辆）

    # === 6. 无信号灯（高速没有红绿灯）===
    # 检测画面上部是否有红/绿/黄色圆形区域
    upper_region = hsv[0:int(h * 0.35), :]

    # 红色检测
    red_mask1 = cv2.inRange(upper_region,
                            np.array([0, 100, 100]),
                            np.array([10, 255, 255]))
    red_mask2 = cv2.inRange(upper_region,
                            np.array([160, 100, 100]),
                            np.array([180, 255, 255]))
    red_pixels = np.count_nonzero(red_mask1) + np.count_nonzero(red_mask2)

    # 绿色检测
    green_mask = cv2.inRange(upper_region,
                             np.array([35, 100, 100]),
                             np.array([85, 255, 255]))
    green_pixels = np.count_nonzero(green_mask)

    traffic_light_ratio = (red_pixels + green_pixels) / max(1, upper_region[:, :, 0].size)
    details['traffic_light_ratio'] = round(traffic_light_ratio, 4)

    # 有信号灯 → 不太可能是高速
    if traffic_light_ratio > 0.005:
        score -= 0.2

    # === 7. 画面稳定性/速度感（高速行驶画面模糊方向一致）===
    # 简单用拉普拉斯方差衡量，高速行驶两侧模糊
    left_blur = cv2.Laplacian(
        gray[int(h * 0.4):int(h * 0.7), 0:int(w * 0.15)],
        cv2.CV_64F).var()
    center_sharp = cv2.Laplacian(
        gray[int(h * 0.4):int(h * 0.7), int(w * 0.35):int(w * 0.65)],
        cv2.CV_64F).var()
    details['edge_blur_ratio'] = round(
        center_sharp / max(1, left_blur), 2)
    # 高速行驶时中央比边缘清晰
    if left_blur > 0 and center_sharp / left_blur > 1.5:
        score += 0.1

    # 归一化
    score = max(0.0, min(1.0, score))
    return score, details


def classify_highway_score(avg_score: float):
    """Map an average frame score to True / None / False."""
    if avg_score >= HIGHWAY_SCORE_YES:
        return True
    if avg_score >= HIGHWAY_SCORE_MAYBE:
        return None
    return False


def quick_scan_video(video_path: str, sample_count: int = 5) -> VideoSegment:
    """
    快速扫描一个视频文件，判断是否是高速公路

    只采样几帧，不做完整分析
    """
    seg = VideoSegment(
        path=str(video_path),
        filename=Path(video_path).name,
        size_mb=round(os.path.getsize(video_path) / 1024 / 1024, 1),
    )

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        seg.highway_reason = '无法打开视频'
        return seg

    seg.width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    seg.height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    seg.fps = cap.get(cv2.CAP_PROP_FPS) or 30
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    seg.duration_sec = round(total / seg.fps, 1)

    # 检查 GPS（从元数据模块）
    try:
        from src.metadata import extract_with_ffprobe
        meta = extract_with_ffprobe(str(video_path))
        if meta.get('gps_lat'):
            seg.has_gps = True
            seg.gps_lat = meta['gps_lat']
            seg.gps_lon = meta['gps_lon']
    except Exception:
        pass

    # 均匀采样几帧
    scores = []
    positions = [i / (sample_count + 1) for i in range(1, sample_count + 1)]

    for pos in positions:
        frame_idx = int(total * pos)
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ret, frame = cap.read()
        if not ret:
            continue

        score, _ = analyze_frame_highway_features(frame)
        scores.append(score)

    cap.release()

    seg.frame_scores = [round(s, 2) for s in scores]

    if not scores:
        seg.highway_confidence = 0
        seg.is_highway = False
        seg.highway_reason = '无法读取帧'
        return seg

    avg_score = sum(scores) / len(scores)
    seg.highway_confidence = round(avg_score, 2)
    seg.is_highway = classify_highway_score(avg_score)
    if seg.is_highway is True:
        seg.highway_reason = '画面特征匹配高速公路'
    elif seg.is_highway is None:
        seg.highway_reason = '可能是高速，需确认'
    else:
        seg.highway_reason = '不像高速公路'

    return seg


def scan_directory(directory: str, extensions=VIDEO_EXTENSIONS
                   ) -> List[VideoSegment]:
    """
    扫描目录下所有视频文件，筛选出高速路段

    Returns:
        按高速置信度降序排列的视频列表
    """
    directory = Path(directory)
    results = []

    video_files = sorted([
        f for f in directory.iterdir()
        if f.suffix.lower() in extensions and f.is_file()
    ])

    print(f'找到 {len(video_files)} 个视频文件，开始筛选...\n')

    for i, vf in enumerate(video_files):
        seg = quick_scan_video(str(vf))
        results.append(seg)

        icon = '🟢' if seg.is_highway else ('🟡' if seg.is_highway is None else '⚫')
        print(f'  {icon} {seg.filename:<40} '
              f'{seg.duration_sec:>6.1f}s  '
              f'{seg.size_mb:>6.1f}MB  '
              f'置信度:{seg.highway_confidence:.0%}  '
              f'{seg.highway_reason}')

    # 按置信度排序
    results.sort(key=lambda s: s.highway_confidence, reverse=True)

    highway = [s for s in results if s.is_highway is True]
    maybe = [s for s in results if s.is_highway is None]
    not_highway = [s for s in results if s.is_highway is False]

    print(f'\n筛选结果:')
    print(f'  🟢 高速路段: {len(highway)} 个')
    print(f'  🟡 待确认: {len(maybe)} 个')
    print(f'  ⚫ 非高速: {len(not_highway)} 个')

    total_highway_sec = sum(s.duration_sec for s in highway)
    total_all_sec = sum(s.duration_sec for s in results)
    print(f'  高速时长: {total_highway_sec / 60:.1f} 分钟'
          f' / 总时长: {total_all_sec / 60:.1f} 分钟')

    return results


if __name__ == '__main__':
    import sys
    if len(sys.argv) < 2:
        print('Usage: python highway_filter.py <video_or_directory>')
        print('  扫描目录: python highway_filter.py /path/to/dashcam/')
        print('  单个文件: python highway_filter.py video.mp4')
        sys.exit(1)

    target = sys.argv[1]

    if os.path.isdir(target):
        results = scan_directory(target)
    else:
        seg = quick_scan_video(target)
        icon = '🟢' if seg.is_highway else ('🟡' if seg.is_highway is None else '⚫')
        print(f'{icon} {seg.filename}')
        print(f'  时长: {seg.duration_sec}s, 分辨率: {seg.width}x{seg.height}')
        print(f'  高速置信度: {seg.highway_confidence:.0%}')
        print(f'  判定: {seg.highway_reason}')
        print(f'  帧得分: {seg.frame_scores}')
