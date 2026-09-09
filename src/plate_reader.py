"""
plate_reader.py - 车牌识别模块
主引擎: PaddleOCR（通用 OCR + 车牌正则匹配）
备选: HyperLPR3（如果模型可用）

注意：最终打包分发时会统一导出为 ONNX，去掉 PaddlePaddle 依赖。
MVP 阶段先用 PaddleOCR 跑通逻辑。
"""
# CREATED_BY: highway-reporter MVP
# STATUS: deprecated

import cv2
import re
import numpy as np
from typing import Optional, List, Tuple


# 中国车牌正则
PLATE_PATTERN = re.compile(
    r'[京津沪渝冀豫云辽黑湘皖鲁新苏浙赣鄂桂甘晋蒙陕吉闽贵粤川青藏琼宁]'
    r'[A-HJ-NP-Z]'         # 省份后字母（排除 I/O）
    r'[A-HJ-NP-Z0-9]{4,5}' # 4-5位字母数字
    r'[A-HJ-NP-Z0-9挂学警港澳]?'  # 可选尾部
)

# 宽松版（OCR 常错 0/O, 1/I 等）
PLATE_PATTERN_LOOSE = re.compile(
    r'[京津沪渝冀豫云辽黑湘皖鲁新苏浙赣鄂桂甘晋蒙陕吉闽贵粤川青藏琼宁]'
    r'[A-Z0-9]'
    r'[A-Z0-9]{4,6}'
)

# 省份映射
PROVINCE_MAP = {
    '京': '北京', '津': '天津', '沪': '上海', '渝': '重庆',
    '冀': '河北', '豫': '河南', '云': '云南', '辽': '辽宁',
    '黑': '黑龙江', '湘': '湖南', '皖': '安徽', '鲁': '山东',
    '新': '新疆', '苏': '江苏', '浙': '浙江', '赣': '江西',
    '鄂': '湖北', '桂': '广西', '甘': '甘肃', '晋': '山西',
    '蒙': '内蒙古', '陕': '陕西', '吉': '吉林', '闽': '福建',
    '贵': '贵州', '粤': '广东', '川': '四川', '青': '青海',
    '藏': '西藏', '琼': '海南', '宁': '宁夏',
}


class PlateReader:
    """车牌识别器"""

    def __init__(self):
        self._ocr = None

    def _ensure_ocr(self):
        if self._ocr is None:
            from paddleocr import PaddleOCR
            self._ocr = PaddleOCR(lang='ch')

    def _preprocess_crop(self, crop: np.ndarray) -> np.ndarray:
        """预处理车牌区域，提升 OCR 精度"""
        h, w = crop.shape[:2]

        # 如果太小，放大到合理尺寸
        if w < 200:
            scale = 200 / w
            crop = cv2.resize(crop, None, fx=scale, fy=scale,
                              interpolation=cv2.INTER_CUBIC)

        # 锐化（对模糊车牌有帮助）
        kernel = np.array([[-1, -1, -1],
                           [-1,  9, -1],
                           [-1, -1, -1]])
        sharpened = cv2.filter2D(crop, -1, kernel)

        return sharpened

    def _extract_plate_from_ocr(self, ocr_results) -> Optional[dict]:
        """从 OCR 结果中提取车牌号"""
        if not ocr_results:
            return None

        best_plate = None
        best_conf = 0

        # PaddleOCR 返回格式可能是嵌套列表
        results_list = ocr_results
        if results_list and isinstance(results_list[0], list):
            # 展平嵌套结构
            flat = []
            for page in results_list:
                if page:
                    flat.extend(page)
            results_list = flat

        for item in results_list:
            if not item or len(item) < 2:
                continue

            # 不同版本的返回格式
            if isinstance(item, dict):
                text = item.get('text', '')
                conf = item.get('score', 0)
            elif isinstance(item, (list, tuple)) and len(item) >= 2:
                # [[box], (text, conf)] 格式
                if isinstance(item[1], (list, tuple)):
                    text = str(item[1][0])
                    conf = float(item[1][1]) if len(item[1]) > 1 else 0
                else:
                    text = str(item[1])
                    conf = 0
            else:
                continue

            text = text.replace(' ', '').replace('·', '').upper()
            text = self._fix_ocr_errors(text)

            # 先严格匹配
            match = PLATE_PATTERN.search(text)
            if not match:
                match = PLATE_PATTERN_LOOSE.search(text)

            if match and conf > best_conf:
                best_plate = {
                    'plate_text': match.group(),
                    'confidence': float(conf),
                }
                best_conf = conf

        return best_plate

    @staticmethod
    def _fix_ocr_errors(text: str) -> str:
        """修复 OCR 常见误识别"""
        if len(text) < 6:
            return text
        # 第二位应该是字母，修正数字误识别
        if len(text) >= 2:
            fixes = {'0': 'O', '1': 'I', '8': 'B', '6': 'G', '2': 'Z'}
            if text[1] in fixes:
                text = text[0] + fixes[text[1]] + text[2:]
        return text

    def read_plate(self, frame: np.ndarray, bbox: list) -> Optional[dict]:
        """
        从车辆检测框中识别车牌

        Args:
            frame: 完整帧 (BGR)
            bbox: [x1, y1, x2, y2]

        Returns:
            {'plate_text': '粤B12345', 'confidence': 0.95} or None
        """
        self._ensure_ocr()

        x1, y1, x2, y2 = [int(v) for v in bbox]
        h, w = frame.shape[:2]

        # 取车辆下半部分（车牌区域）
        py1 = y1 + int((y2 - y1) * 0.35)
        py2 = min(y2 + int((y2 - y1) * 0.1), h)
        px1 = max(0, x1 - int((x2 - x1) * 0.05))
        px2 = min(x2 + int((x2 - x1) * 0.05), w)

        crop = frame[py1:py2, px1:px2]
        if crop.size == 0:
            return None

        crop = self._preprocess_crop(crop)

        try:
            results = self._ocr.ocr(crop, cls=True)
            return self._extract_plate_from_ocr(results)
        except Exception:
            return None

    def read_plate_full_frame(self, frame: np.ndarray) -> List[dict]:
        """对整帧做车牌识别"""
        self._ensure_ocr()
        try:
            results = self._ocr.ocr(frame, cls=True)
            plate = self._extract_plate_from_ocr(results)
            return [plate] if plate else []
        except Exception:
            return []

    @staticmethod
    def get_province(plate_text: str) -> Optional[str]:
        if plate_text:
            return PROVINCE_MAP.get(plate_text[0])
        return None

    @staticmethod
    def validate_plate(plate_text: str) -> bool:
        return bool(PLATE_PATTERN.fullmatch(plate_text))
