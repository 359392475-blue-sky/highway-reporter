#!/usr/bin/env python3
"""
cloud_reader.py - 云端视觉大模型精读模块
负责车牌 OCR、时间戳提取、车辆信息识别

分层架构中的 L2 层：只对本地粗筛选出的关键帧调用云端 API
"""
# CREATED_BY: highway-reporter MVP
# STATUS: active

import json
import base64
import os
import ssl
import urllib.request
import re
from pathlib import Path
from typing import Optional, List, Dict
from collections import Counter
from dataclasses import dataclass, field

from .config import (
    CLOUD_BASE_URL,
    CLOUD_FALLBACK_MODEL,
    CLOUD_MODEL,
    PLATE_MIN_CONFIDENCE,
    PLATE_MIN_VOTES,
)
from .preflight import load_api_key


@dataclass
class CloudReadResult:
    """单帧云端读取结果"""
    frame_idx: int
    plate_text: Optional[str] = None
    plate_confidence: float = 0.0  # 基于多帧投票的置信度
    timestamp: Optional[str] = None  # 日期时间
    speed: Optional[str] = None  # 车速
    vehicle_color: Optional[str] = None
    vehicle_type: Optional[str] = None
    recorder_brand: Optional[str] = None
    road_signs: Optional[str] = None
    raw_response: str = ''


@dataclass
class ViolationReport:
    """单起违法的完整举报材料"""
    violation_id: int
    # 车辆信息
    plate_text: Optional[str] = None
    plate_confidence: float = 0.0
    plate_votes: Dict[str, int] = field(default_factory=dict)
    vehicle_color: Optional[str] = None
    vehicle_type: Optional[str] = None
    # 时间地点
    violation_time: Optional[str] = None
    violation_location: Optional[str] = None
    speed: Optional[str] = None
    recorder_brand: Optional[str] = None
    # 违法信息
    violation_type: str = '占用应急车道'
    start_sec: float = 0
    end_sec: float = 0
    duration_sec: float = 0
    detection_confidence: float = 0
    # 证据
    evidence_frames: List[str] = field(default_factory=list)
    # 质量
    can_report: bool = False
    issues: List[str] = field(default_factory=list)
    api_calls: int = 0


# 中国车牌正则（用于从大模型回复中提取）
PLATE_RE = re.compile(
    r'[京津沪渝冀豫云辽黑湘皖鲁新苏浙赣鄂桂甘晋蒙陕吉闽贵粤川青藏琼宁]'
    r'[A-HJ-NP-Z]'
    r'[A-HJ-NP-Z0-9?]{4,6}'
)

# 时间戳正则
TIME_RE = re.compile(r'(\d{4}[-/]\d{1,2}[-/]\d{1,2}\s+\d{1,2}:\d{2}(?::\d{2})?)')
SPEED_RE = re.compile(r'(\d+)\s*[Kk][Mm]/[Hh]')


class CloudReader:
    """云端视觉大模型读取器"""

    def __init__(
        self,
        model: str = CLOUD_MODEL,
        fallback_model: str = CLOUD_FALLBACK_MODEL,
        api_key: Optional[str] = None,
        base_url: str = CLOUD_BASE_URL,
        timeout: int = 45,
        max_retries: int = 2,
    ):
        self.model = model
        self.fallback_model = fallback_model
        self.base_url = base_url
        self.timeout = timeout
        self.max_retries = max_retries

        # 清除代理（国内直连）
        for k in ['ALL_PROXY', 'HTTPS_PROXY', 'HTTP_PROXY',
                   'all_proxy', 'https_proxy', 'http_proxy']:
            os.environ.pop(k, None)

        # API key
        if api_key:
            self.api_key = api_key
        else:
            self.api_key = load_api_key()

        self.ssl_ctx = ssl.create_default_context()

    @staticmethod
    def _compress_image(image_path: str, max_bytes: int = 150_000) -> bytes:
        """压缩图片到指定大小以内"""
        import cv2
        data = open(image_path, 'rb').read()
        if len(data) <= max_bytes:
            return data

        img = cv2.imread(image_path)
        if img is None:
            return data

        # 先缩小分辨率
        h, w = img.shape[:2]
        if w > 1280:
            scale = 1280 / w
            img = cv2.resize(img, (1280, int(h * scale)))

        # 降低 JPEG 质量直到满足大小
        for quality in [75, 60, 45, 30]:
            _, buf = cv2.imencode('.jpg', img,
                                 [cv2.IMWRITE_JPEG_QUALITY, quality])
            if len(buf) <= max_bytes:
                return buf.tobytes()

        return buf.tobytes()

    def _call_vision(self, image_path: str, prompt: str,
                     model: Optional[str] = None) -> Optional[str]:
        """调用视觉模型"""
        model = model or self.model

        img_bytes = self._compress_image(image_path)
        img_b64 = base64.b64encode(img_bytes).decode()

        mime = 'image/jpeg'

        body = json.dumps({
            'model': model,
            'messages': [{
                'role': 'user',
                'content': [
                    {'type': 'text', 'text': prompt},
                    {'type': 'image_url',
                     'image_url': {'url': 'data:%s;base64,%s' % (mime, img_b64)}}
                ]
            }],
            'max_tokens': 500,
            'temperature': 0.1,  # 低温度减少幻觉
        }).encode()

        for attempt in range(self.max_retries + 1):
            try:
                req = urllib.request.Request(
                    self.base_url + '/chat/completions',
                    data=body,
                    headers={
                        'Authorization': 'Bearer ' + self.api_key,
                        'Content-Type': 'application/json',
                    },
                    method='POST',
                )
                resp = urllib.request.urlopen(
                    req, context=self.ssl_ctx, timeout=self.timeout)
                result = json.loads(resp.read())
                return result['choices'][0]['message']['content'].strip()
            except Exception as e:
                print(f'         云端调用失败({type(e).__name__})，重试 {attempt+1}/{self.max_retries}', flush=True)
                if attempt < self.max_retries:
                    continue
                # 最后一次失败，尝试 fallback 模型
                if model != self.fallback_model:
                    try:
                        return self._call_vision(
                            image_path, prompt, self.fallback_model)
                    except Exception:
                        pass
                return None

    def read_plate(self, image_path: str) -> Optional[str]:
        """读取车牌号"""
        prompt = (
            '请精确辨认这张中国高速公路行车记录仪截图中，'
            '应急车道（最右侧靠护栏车道）内车辆的车牌号码。\n'
            '中国车牌格式：省份汉字+字母+5位字母数字，如 粤A00000（虚构格式示例，不代表图中车辆）\n'
            '请逐个字符辨认。只输出车牌号，不要其他任何文字。\n'
            '如果看不清，输出"无法识别"。'
        )
        resp = self._call_vision(image_path, prompt)
        if resp and '无法' not in resp and '识别' not in resp:
            # 提取车牌格式
            match = PLATE_RE.search(resp.replace(' ', ''))
            if match:
                return match.group()
        return None

    def read_full_info(self, image_path: str) -> CloudReadResult:
        """一次调用提取所有信息"""
        prompt = (
            '这是中国高速公路行车记录仪截图。请提取以下信息，每项一行：\n'
            '车牌: （应急车道内车辆的完整车牌号，格式如粤A00000，虚构格式示例，不代表图中车辆）\n'
            '时间: （水印中的日期时间，格式如2000/01/01 00:00:00）\n'
            '车速: （水印中的速度，如07KM/H）\n'
            '颜色: （应急车道车辆颜色，如灰色）\n'
            '车型: （如小型轿车/SUV/货车）\n'
            '记录仪: （记录仪品牌，如360记录仪）\n'
            '路标: （可见的路牌文字，没有则写无）\n\n'
            '只按上述格式输出，不要解释。看不清的写"不清"。'
        )
        result = CloudReadResult(frame_idx=0)

        resp = self._call_vision(image_path, prompt)
        if not resp:
            return result

        result.raw_response = resp

        for line in resp.split('\n'):
            line = line.strip()
            if not line:
                continue

            if line.startswith('车牌'):
                text = line.split(':', 1)[-1].split('：', 1)[-1].strip()
                match = PLATE_RE.search(text.replace(' ', ''))
                if match:
                    result.plate_text = match.group()
            elif line.startswith('时间'):
                text = line.split(':', 1)[-1].split('：', 1)[-1].strip()
                match = TIME_RE.search(text)
                if match:
                    result.timestamp = match.group(1)
            elif line.startswith('车速'):
                text = line.split(':', 1)[-1].split('：', 1)[-1].strip()
                match = SPEED_RE.search(text)
                if match:
                    result.speed = match.group(0)
                elif text and '不清' not in text:
                    result.speed = text
            elif line.startswith('颜色'):
                text = line.split(':', 1)[-1].split('：', 1)[-1].strip()
                if text and '不清' not in text:
                    result.vehicle_color = text
            elif line.startswith('车型'):
                text = line.split(':', 1)[-1].split('：', 1)[-1].strip()
                if text and '不清' not in text:
                    result.vehicle_type = text
            elif line.startswith('记录仪'):
                text = line.split(':', 1)[-1].split('：', 1)[-1].strip()
                if text and '不清' not in text:
                    result.recorder_brand = text
            elif line.startswith('路标'):
                text = line.split(':', 1)[-1].split('：', 1)[-1].strip()
                if text and text != '无' and '不清' not in text:
                    result.road_signs = text

        return result

    def read_plates_multi_frame(
        self, image_paths: List[str], min_votes: int = PLATE_MIN_VOTES
    ) -> tuple:
        """
        多帧投票读取车牌

        Args:
            image_paths: 多张证据帧路径
            min_votes: 最少票数才确认

        Returns:
            (plate_text, confidence, vote_details)
        """
        plates = []
        for i, path in enumerate(image_paths, 1):
            print(f'       读车牌 {i}/{len(image_paths)} ...', flush=True)
            plate = self.read_plate(path)
            if plate:
                plate = plate.replace(' ', '')
                plates.append(plate)
                print(f'         → {plate}', flush=True)
            else:
                print('         → 未识别', flush=True)

        if not plates:
            return None, 0.0, {}

        counter = Counter(plates)
        best_plate, best_count = counter.most_common(1)[0]

        confidence = best_count / len(image_paths)

        if best_count >= min_votes:
            return best_plate, confidence, dict(counter)
        elif len(plates) == 1:
            # 只有一帧读到了，置信度低
            return best_plate, 0.3, dict(counter)
        else:
            # 多帧结果不一致，取最多的但降低置信度
            return best_plate, confidence * 0.7, dict(counter)


def evaluate_can_report(plate_text, plate_confidence: float) -> bool:
    """Hard gate for whether a violation is complete enough to report."""
    return (
        plate_text is not None
        and '?' not in (plate_text or '')
        and plate_confidence >= PLATE_MIN_CONFIDENCE
    )


def build_violation_report(
    cloud_reader: CloudReader,
    violation,
    evidence_frame_paths: List[str],
    fps: float,
) -> ViolationReport:
    """
    为一起违法事件构建完整举报材料

    Args:
        cloud_reader: 云端读取器
        violation: ViolationEvent
        evidence_frame_paths: 证据帧图片路径列表
        fps: 视频帧率

    Returns:
        ViolationReport
    """
    report = ViolationReport(
        violation_id=violation.track_id,
        violation_type='占用应急车道',
        start_sec=round(violation.start_frame / fps, 2),
        end_sec=round(violation.end_frame / fps, 2),
        duration_sec=round(violation.duration_frames / fps, 2),
        detection_confidence=round(violation.confidence, 2),
        evidence_frames=evidence_frame_paths,
    )

    if not evidence_frame_paths:
        report.issues.append('无证据帧')
        return report

    # 1. 多帧投票读车牌
    plate, conf, votes = cloud_reader.read_plates_multi_frame(
        evidence_frame_paths, min_votes=2
    )
    report.plate_text = plate
    report.plate_confidence = conf
    report.plate_votes = votes
    report.api_calls += len(evidence_frame_paths)

    # 2. 用最佳帧读取完整信息（时间戳、颜色等）
    # 选中间那帧（通常最清晰）
    mid_idx = len(evidence_frame_paths) // 2
    print('       读时间/车色等 ...', flush=True)
    full_info = cloud_reader.read_full_info(evidence_frame_paths[mid_idx])
    report.api_calls += 1

    report.violation_time = full_info.timestamp
    report.speed = full_info.speed
    report.vehicle_color = full_info.vehicle_color
    report.vehicle_type = full_info.vehicle_type
    report.recorder_brand = full_info.recorder_brand
    report.violation_location = full_info.road_signs

    # 如果车牌投票没结果，用 full_info 里的
    if not report.plate_text and full_info.plate_text:
        report.plate_text = full_info.plate_text
        report.plate_confidence = 0.3  # 单次读取置信度低

    # 3. 质量评估
    if not report.plate_text:
        report.issues.append('车牌未识别 — 无法举报')
    elif report.plate_confidence < PLATE_MIN_CONFIDENCE:
        report.issues.append('车牌置信度低 — 建议人工确认')
    if '?' in (report.plate_text or ''):
        report.issues.append('车牌有不确定字符')

    if not report.violation_time:
        report.issues.append('违法时间未提取 — 需手动填写')
    if not report.violation_location:
        report.issues.append('违法地点缺失 — 需GPS或手动补充')
    if report.duration_sec < 1.0:
        report.issues.append('违法持续时间短 — 证据可能不充分')

    report.can_report = evaluate_can_report(
        report.plate_text, report.plate_confidence)

    return report
