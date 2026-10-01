#!/usr/bin/env python3
"""
test_modules.py - 模块测试脚本
用合成图像验证各模块能正常工作
"""
import sys
import os
sys.path.insert(0, os.path.dirname(__file__))

import cv2
import numpy as np


def test_vehicle_detector():
    """测试车辆检测器"""
    print("\n=== 测试车辆检测器 ===")
    from src.vehicle_detector import VehicleDetector

    detector = VehicleDetector(model_name='data/yolov8n.pt', confidence=0.3)

    # 创建一张测试图（黑色背景，不会检测到车）
    fake_frame = np.zeros((720, 1280, 3), dtype=np.uint8)
    results = detector.detect_vehicles(fake_frame)
    print(f"  空白帧检测: {len(results)} vehicles (预期: 0)")
    assert len(results) == 0, "空白帧不应检测到车辆"
    print("  ✅ 车辆检测器正常")
    return True


def test_lane_analyzer():
    """测试车道分析器"""
    print("\n=== 测试车道分析器 ===")
    from src.lane_analyzer import EmergencyLaneAnalyzer

    analyzer = EmergencyLaneAnalyzer()

    # 模拟一个在应急车道的检测结果
    det_in_lane = {
        'bbox': [900, 400, 1100, 600],
        'bbox_norm': [0.70, 0.56, 0.86, 0.83],
        'class': 'car',
        'confidence': 0.9,
        'center': (1000.0, 500.0),
        'area': 40000,
    }

    # 模拟一个在正常车道的检测结果
    det_normal = {
        'bbox': [400, 400, 600, 600],
        'bbox_norm': [0.31, 0.56, 0.47, 0.83],
        'class': 'car',
        'confidence': 0.9,
        'center': (500.0, 500.0),
        'area': 40000,
    }

    frame_shape = (720, 1280, 3)
    suspects = analyzer.analyze_frame([det_in_lane, det_normal], frame_shape)
    print(f"  2辆车(1在应急车道): 嫌疑 {len(suspects)} 辆 (预期: 1)")
    assert len(suspects) == 1, "应该只有1辆嫌疑车"
    print("  ✅ 车道分析器正常")
    return True


def test_violation_tracker():
    """测试违法追踪器"""
    print("\n=== 测试违法追踪器 ===")
    from src.violation_tracker import ViolationTracker

    tracker = ViolationTracker(
        emergency_x_threshold=0.68,
        min_violation_frames=5,  # 测试用，降低阈值
    )

    frame_shape = (720, 1280, 3)

    # 模拟车辆在应急车道持续行驶
    det = {
        'bbox': [900, 350, 1100, 550],
        'bbox_norm': [0.70, 0.49, 0.86, 0.76],
        'class': 'car',
        'confidence': 0.9,
        'center': (1000.0, 450.0),
        'area': 40000,
    }

    violations = []
    for i in range(20):
        # 模拟车辆轻微移动
        moved_det = det.copy()
        moved_det['bbox'] = [900 + i, 350, 1100 + i, 550]
        new_v = tracker.update([moved_det], frame_shape)
        violations.extend(new_v)

    print(f"  20帧持续在应急车道: {len(violations)} 起违法 (预期: 1)")
    assert len(violations) == 1, "应该检测到1起违法"
    print(f"  违法详情: track_id={violations[0].track_id}, "
          f"duration={violations[0].duration_frames}帧")
    print("  ✅ 违法追踪器正常")
    return True


def test_plate_reader():
    """测试车牌识别器"""
    print("\n=== 测试车牌识别器 ===")
    from src.plate_reader import PlateReader

    reader = PlateReader()

    # 测试正则匹配
    assert reader.validate_plate('粤B12345')
    assert reader.validate_plate('京A88888')
    assert reader.validate_plate('沪C1234F')
    assert not reader.validate_plate('ABC1234')
    assert not reader.validate_plate('粤')
    print("  ✅ 车牌格式验证正常")

    # 测试省份
    assert reader.get_province('粤B12345') == '广东'
    assert reader.get_province('京A88888') == '北京'
    print("  ✅ 省份识别正常")

    # OCR 初始化测试（用空白图，不期望识别出车牌）
    fake = np.zeros((100, 300, 3), dtype=np.uint8)
    result = reader.read_plate(fake, [0, 0, 300, 100])
    print(f"  空白图 OCR: {result} (预期: None)")
    print("  ✅ 车牌 OCR 初始化正常")
    return True


def test_pipeline_import():
    """测试 pipeline 导入"""
    print("\n=== 测试 Pipeline 导入 ===")
    from src.pipeline import Pipeline
    print("  ✅ Pipeline 模块导入正常")
    return True


if __name__ == '__main__':
    print("=" * 50)
    print("Highway Reporter - 模块测试")
    print("=" * 50)

    tests = [
        ('车辆检测器', test_vehicle_detector),
        ('车道分析器', test_lane_analyzer),
        ('违法追踪器', test_violation_tracker),
        ('车牌识别器', test_plate_reader),
        ('Pipeline导入', test_pipeline_import),
    ]

    passed = 0
    failed = 0
    for name, test_fn in tests:
        try:
            test_fn()
            passed += 1
        except Exception as e:
            print(f"  ❌ {name} 失败: {e}")
            import traceback
            traceback.print_exc()
            failed += 1

    print(f"\n{'=' * 50}")
    print(f"结果: {passed} 通过, {failed} 失败")
    print(f"{'=' * 50}")
    sys.exit(0 if failed == 0 else 1)
