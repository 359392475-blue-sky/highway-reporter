#!/usr/bin/env python3
"""
extract_report.py - 完整举报素材提取测试
按深圳随手e拍的字段要求，从视频中提取所有可提取的信息

深圳随手e拍 / 12123 举报所需字段：
1. 违法时间 — 视频时间戳 / 行车记录仪水印时间
2. 违法地点 — GPS / 路标 / 行车记录仪水印
3. 车牌号码 — OCR
4. 车辆颜色 — 视觉识别
5. 车辆品牌/车型 — 视觉识别（可选）
6. 违法类型 — 占用应急车道
7. 证据照片 — 至少2-3张（车牌清晰+违法行为全景）
8. 证据视频 — 15-30秒片段
"""
# CREATED_BY: highway-reporter MVP test
# STATUS: experimental

import sys
import os
import cv2
import json
import time
import re
import numpy as np
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from src.vehicle_detector import VehicleDetector
from src.violation_tracker import ViolationTracker
from src.plate_reader import PlateReader


def extract_timestamp_from_frame(frame):
    """
    尝试从行车记录仪水印中提取时间戳
    大多数记录仪在画面顶部或底部有时间水印
    """
    h, w = frame.shape[:2]
    
    # 行车记录仪时间戳通常在：
    # - 顶部中间区域
    # - 底部左/右区域
    # 提取这些区域做 OCR
    regions = {
        'top': frame[0:int(h*0.08), :],              # 顶部 8%
        'bottom': frame[int(h*0.90):, :],             # 底部 10%
        'bottom_left': frame[int(h*0.88):, 0:int(w*0.5)],
        'bottom_right': frame[int(h*0.88):, int(w*0.5):],
        'top_right': frame[0:int(h*0.10), int(w*0.5):],
    }
    
    from paddleocr import PaddleOCR
    ocr = PaddleOCR(lang='ch')
    
    timestamps = []
    for region_name, region in regions.items():
        if region.size == 0:
            continue
        # 放大小区域提高 OCR 精度
        if region.shape[0] < 50:
            region = cv2.resize(region, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)
        
        try:
            results = ocr.ocr(region, cls=True)
            if results and results[0]:
                for line in results[0]:
                    if line and len(line) >= 2:
                        text = line[1][0] if isinstance(line[1], (list, tuple)) else str(line[1])
                        conf = line[1][1] if isinstance(line[1], (list, tuple)) and len(line[1]) > 1 else 0
                        # 匹配日期时间格式
                        # 常见格式：2026-03-12 14:23:05, 2026/03/12 14:23, 11:19:01 等
                        time_patterns = [
                            r'(\d{4}[-/]\d{2}[-/]\d{2}\s+\d{2}:\d{2}:\d{2})',  # 完整日期时间
                            r'(\d{4}[-/]\d{2}[-/]\d{2}\s+\d{2}:\d{2})',         # 日期+时分
                            r'(\d{2}:\d{2}:\d{2})',                               # 仅时间
                        ]
                        for pat in time_patterns:
                            m = re.search(pat, text)
                            if m:
                                timestamps.append({
                                    'text': m.group(1),
                                    'region': region_name,
                                    'confidence': float(conf),
                                    'full_text': text,
                                })
        except Exception:
            pass
    
    return timestamps


def extract_vehicle_color(frame, bbox):
    """从检测框中提取车辆主色调"""
    x1, y1, x2, y2 = [int(v) for v in bbox]
    h, w = frame.shape[:2]
    
    # 取车身中间区域（避开车窗和车轮）
    cy1 = y1 + int((y2 - y1) * 0.2)
    cy2 = y1 + int((y2 - y1) * 0.7)
    cx1 = x1 + int((x2 - x1) * 0.15)
    cx2 = x2 - int((x2 - x1) * 0.15)
    
    crop = frame[max(0,cy1):min(h,cy2), max(0,cx1):min(w,cx2)]
    if crop.size == 0:
        return '未知'
    
    # 转 HSV 分析主色调
    hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
    h_vals = hsv[:,:,0].flatten()
    s_vals = hsv[:,:,1].flatten()
    v_vals = hsv[:,:,2].flatten()
    
    avg_h = np.median(h_vals)
    avg_s = np.median(s_vals)
    avg_v = np.median(v_vals)
    
    # 颜色判断
    if avg_s < 40:  # 低饱和度 = 灰/白/黑
        if avg_v > 180:
            return '白色'
        elif avg_v < 60:
            return '黑色'
        else:
            return '灰色'
    elif avg_v < 50:
        return '黑色'
    else:
        # 按色相判断
        if avg_h < 10 or avg_h > 170:
            return '红色'
        elif 10 <= avg_h < 25:
            return '橙色'
        elif 25 <= avg_h < 35:
            return '黄色'
        elif 35 <= avg_h < 85:
            return '绿色'
        elif 85 <= avg_h < 130:
            return '蓝色'
        else:
            return '紫色'


def extract_all_text_from_frame(frame):
    """提取画面中所有文字（路标、水印等）"""
    from paddleocr import PaddleOCR
    ocr = PaddleOCR(lang='ch')
    
    try:
        results = ocr.ocr(frame, cls=True)
        texts = []
        if results and results[0]:
            for line in results[0]:
                if line and len(line) >= 2:
                    text = line[1][0] if isinstance(line[1], (list, tuple)) else str(line[1])
                    conf = line[1][1] if isinstance(line[1], (list, tuple)) and len(line[1]) > 1 else 0
                    if float(conf) > 0.5:
                        texts.append({'text': text, 'confidence': round(float(conf), 2)})
        return texts
    except Exception:
        return []


def find_best_plate_frame(video_path, violation_start, violation_end, fps, detector, plate_reader):
    """
    在违法时间段前后寻找车牌最清晰的帧
    扩大搜索范围：违法前3秒到违法后5秒
    """
    cap = cv2.VideoCapture(video_path)
    
    search_start = max(0, int((violation_start - 3) * fps))
    search_end = int((violation_end + 5) * fps)
    
    cap.set(cv2.CAP_PROP_POS_FRAMES, search_start)
    
    best_plate = None
    best_conf = 0
    best_frame = None
    best_frame_idx = 0
    all_plates = []
    
    frame_idx = search_start
    while frame_idx < search_end:
        ret, frame = cap.read()
        if not ret:
            break
        frame_idx += 1
        
        if frame_idx % 3 != 0:  # 每3帧检查一次
            continue
        
        # 检测车辆
        dets = detector.detect_vehicles(frame)
        
        # 对右侧（应急车道）的车辆做 OCR
        for d in dets:
            cx_norm = (d['bbox_norm'][0] + d['bbox_norm'][2]) / 2
            if cx_norm < 0.55:  # 只看右半部分
                continue
            
            plate = plate_reader.read_plate(frame, d['bbox'])
            if plate:
                all_plates.append({
                    'text': plate['plate_text'],
                    'conf': plate['confidence'],
                    'frame_idx': frame_idx,
                    'time': round(frame_idx / fps, 2),
                    'vehicle_class': d['class'],
                })
                if plate['confidence'] > best_conf:
                    best_conf = plate['confidence']
                    best_plate = plate
                    best_frame = frame.copy()
                    best_frame_idx = frame_idx
    
    cap.release()
    return best_plate, best_frame, best_frame_idx, all_plates


def main():
    video_path = 'data/test-videos/violation_short_39s.mp4'
    output_dir = Path('data/test-videos/report_test')
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("=" * 60)
    print("🚨 深圳随手e拍 举报素材提取测试")
    print("=" * 60)
    
    # 1. 初始化
    print("\n[1/7] 初始化模型...")
    detector = VehicleDetector(model_name='data/yolov8n.pt', confidence=0.40)
    tracker = ViolationTracker(emergency_x_threshold=0.68, min_violation_frames=15)
    plate_reader = PlateReader()
    
    # 2. 扫描视频找违法
    print("\n[2/7] 扫描视频...")
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    violation_frames = {}  # frame_idx -> frame (保存违法附近的帧)
    frame_idx = 0
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        frame_idx += 1
        if frame_idx % 2 != 0:
            continue
        
        dets = detector.detect_vehicles(frame)
        new_v = tracker.update(dets, frame.shape)
        suspects = tracker.get_active_suspects()
        
        # 保存嫌疑车辆附近的帧
        if suspects or new_v:
            violation_frames[frame_idx] = frame.copy()
    
    cap.release()
    violations = tracker.get_violations()
    print(f"   发现 {len(violations)} 起违法行为")
    
    # 3. 提取时间戳
    print("\n[3/7] 提取时间戳（行车记录仪水印）...")
    # 取多帧提取时间戳
    cap = cv2.VideoCapture(video_path)
    timestamps = []
    for pos in [0.0, 0.25, 0.5, 0.75, 1.0]:
        fi = max(1, int(total * pos) - 1)
        cap.set(cv2.CAP_PROP_POS_FRAMES, fi)
        ret, frame = cap.read()
        if ret:
            ts = extract_timestamp_from_frame(frame)
            for t in ts:
                t['frame_idx'] = fi
                t['video_time'] = round(fi / fps, 2)
            timestamps.extend(ts)
    cap.release()
    
    if timestamps:
        print(f"   找到 {len(timestamps)} 个时间戳:")
        for t in timestamps:
            print(f"     [{t['region']}] {t['text']} (置信度:{t['confidence']:.2f}) @ {t['video_time']}s")
    else:
        print("   ⚠️ 未找到时间戳水印")
    
    # 4. 提取车牌
    print("\n[4/7] 精细车牌识别（扩大搜索范围）...")
    plate_results = []
    for v in violations:
        start_sec = v.start_frame / fps
        end_sec = v.end_frame / fps
        print(f"   违法#{v.track_id}: 搜索 {start_sec:.1f}s - {end_sec:.1f}s ± 缓冲区")
        
        best_plate, best_frame, best_fi, all_plates = find_best_plate_frame(
            video_path, start_sec, end_sec, fps, detector, plate_reader
        )
        
        plate_results.append({
            'violation_id': v.track_id,
            'best_plate': best_plate,
            'best_frame_idx': best_fi,
            'all_plates_found': all_plates,
        })
        
        if best_plate:
            print(f"     ✅ 最佳车牌: {best_plate['plate_text']} (置信度: {best_plate['confidence']:.2f})")
            # 保存车牌截图
            if best_frame is not None:
                plate_path = str(output_dir / f"plate_v{v.track_id}.jpg")
                cv2.imwrite(plate_path, best_frame, [cv2.IMWRITE_JPEG_QUALITY, 95])
        else:
            print(f"     ❌ 未识别到车牌")
        
        if all_plates:
            print(f"     所有识别结果: {[p['text'] + '(' + str(round(p['conf'],2)) + ')' for p in all_plates]}")
    
    # 5. 提取车辆颜色
    print("\n[5/7] 车辆颜色识别...")
    color_results = []
    for v in violations:
        # 找违法帧中最近的一帧
        closest_fi = min(violation_frames.keys(), key=lambda x: abs(x - v.end_frame))
        frame = violation_frames[closest_fi]
        
        # 对应急车道区域的车辆检测颜色
        dets = detector.detect_vehicles(frame)
        for d in dets:
            cx_norm = (d['bbox_norm'][0] + d['bbox_norm'][2]) / 2
            if cx_norm >= 0.60:
                color = extract_vehicle_color(frame, d['bbox'])
                color_results.append({
                    'violation_id': v.track_id,
                    'color': color,
                    'vehicle_class': d['class'],
                    'frame_idx': closest_fi,
                })
                print(f"   违法#{v.track_id}: {color} {d['class']}")
    
    # 6. 提取画面文字（路标、地点信息）
    print("\n[6/7] 画面文字提取（路标/地点）...")
    cap = cv2.VideoCapture(video_path)
    all_texts = []
    for pos in [0.1, 0.3, 0.5, 0.7, 0.9]:
        fi = int(total * pos)
        cap.set(cv2.CAP_PROP_POS_FRAMES, fi)
        ret, frame = cap.read()
        if ret:
            texts = extract_all_text_from_frame(frame)
            for t in texts:
                t['frame_idx'] = fi
                t['video_time'] = round(fi / fps, 2)
            all_texts.extend(texts)
    cap.release()
    
    # 去重
    seen = set()
    unique_texts = []
    for t in all_texts:
        if t['text'] not in seen and len(t['text']) > 1:
            seen.add(t['text'])
            unique_texts.append(t)
    
    print(f"   提取到 {len(unique_texts)} 段文字:")
    for t in unique_texts:
        print(f"     \"{t['text']}\" (置信度:{t['confidence']}) @ {t['video_time']}s")
    
    # 7. 保存证据截图
    print("\n[7/7] 保存证据截图...")
    evidence_images = []
    for v in violations:
        # 选取违法时间段内的截图（不同时间点）
        relevant_frames = sorted([
            (fi, f) for fi, f in violation_frames.items()
            if v.start_frame - 30 <= fi <= v.end_frame + 60
        ])
        
        if len(relevant_frames) >= 2:
            # 取第一帧和最后一帧（证明持续违法）
            for i, (fi, frame) in enumerate([relevant_frames[0], relevant_frames[-1]]):
                fname = f"evidence_v{v.track_id}_{i+1}_t{fi/fps:.1f}s.jpg"
                fpath = str(output_dir / fname)
                cv2.imwrite(fpath, frame, [cv2.IMWRITE_JPEG_QUALITY, 95])
                evidence_images.append(fname)
                print(f"   保存: {fname}")
        elif relevant_frames:
            fi, frame = relevant_frames[0]
            fname = f"evidence_v{v.track_id}_1_t{fi/fps:.1f}s.jpg"
            fpath = str(output_dir / fname)
            cv2.imwrite(fpath, frame, [cv2.IMWRITE_JPEG_QUALITY, 95])
            evidence_images.append(fname)
            print(f"   保存: {fname}")
    
    # === 汇总报告 ===
    print("\n" + "=" * 60)
    print("📋 深圳随手e拍 举报表单内容")
    print("=" * 60)
    
    for i, v in enumerate(violations):
        pr = plate_results[i] if i < len(plate_results) else {}
        cr = [c for c in color_results if c['violation_id'] == v.track_id]
        
        plate_text = pr.get('best_plate', {})
        plate_str = plate_text['plate_text'] if plate_text else '❌ 未识别'
        plate_conf = plate_text['confidence'] if plate_text else 0
        
        color_str = cr[0]['color'] if cr else '未识别'
        vtype_cn = {'car': '小型轿车', 'truck': '货车', 'bus': '大型客车'}.get(v.vehicle_class, '小型轿车')
        
        # 时间戳
        best_ts = None
        if timestamps:
            # 找距离违法最近的时间戳
            v_time = v.start_frame / fps
            closest = min(timestamps, key=lambda t: abs(t['video_time'] - v_time))
            best_ts = closest['text']
        
        print(f"\n--- 违法 #{v.track_id} ---")
        print(f"  违法类型:  占用应急车道")
        print(f"  车牌号码:  {plate_str}" + (f" (置信度: {plate_conf:.0%})" if plate_text else ""))
        print(f"  车辆颜色:  {color_str}")
        print(f"  车辆类型:  {vtype_cn}")
        print(f"  违法时间:  {best_ts or '❌ 未提取到'}" + 
              f" (视频第 {v.start_frame/fps:.1f}s - {v.end_frame/fps:.1f}s)")
        print(f"  违法地点:  ❌ 需要GPS或路标补充")
        print(f"  持续时间:  {v.duration_frames}帧 ({v.duration_frames/fps:.1f}秒)")
        print(f"  置信度:    {v.confidence:.0%}")
        print(f"  证据截图:  {len([e for e in evidence_images if f'v{v.track_id}' in e])} 张")
        
        # 质量评估
        print(f"\n  📊 举报可行性评估:")
        issues = []
        if not plate_text:
            issues.append("❌ 车牌未识别 — 无法举报（核心字段）")
        elif plate_conf < 0.7:
            issues.append("⚠️ 车牌置信度低 — 需人工确认")
        if not best_ts:
            issues.append("⚠️ 时间戳未提取 — 需手动填写")
        issues.append("⚠️ 违法地点缺失 — 需GPS或手动填写")
        if v.duration_frames < 20:
            issues.append("⚠️ 违法持续时间短 — 证据可能不够充分")
        
        for issue in issues:
            print(f"     {issue}")
        
        can_report = plate_text is not None and plate_conf > 0.5
        print(f"\n  {'✅ 可以举报（需补充地点）' if can_report else '❌ 无法举报（缺少车牌）'}")
    
    # 保存完整报告 JSON
    report = {
        'video': video_path,
        'video_info': {'width': w, 'height': h, 'fps': fps, 'duration': round(total/fps, 1)},
        'violations': [
            {
                'id': v.track_id,
                'type': '占用应急车道',
                'plate': plate_results[i].get('best_plate') if i < len(plate_results) else None,
                'all_plates': plate_results[i].get('all_plates_found', []) if i < len(plate_results) else [],
                'color': next((c['color'] for c in color_results if c['violation_id'] == v.track_id), None),
                'vehicle_class': v.vehicle_class,
                'start_sec': round(v.start_frame / fps, 2),
                'end_sec': round(v.end_frame / fps, 2),
                'confidence': round(v.confidence, 2),
            }
            for i, v in enumerate(violations)
        ],
        'timestamps': timestamps,
        'texts_found': unique_texts,
        'evidence_images': evidence_images,
    }
    
    with open(output_dir / 'report.json', 'w', encoding='utf-8') as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    
    print(f"\n完整报告: {output_dir / 'report.json'}")
    print(f"证据目录: {output_dir}")


if __name__ == '__main__':
    main()
