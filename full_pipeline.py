#!/usr/bin/env python3
"""
full_pipeline.py - 完整分层 Pipeline
L1: 本地 YOLO 粗筛（免费）→ L1.5: 智能取帧 → L2: 云端大模型精读

用法:
    python full_pipeline.py <video_path> [-o output_dir]
"""
# CREATED_BY: highway-reporter MVP
# STATUS: active

import sys
import os
import cv2
import json
import time
import argparse
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from src.vehicle_detector import VehicleDetector
from src.violation_tracker import ViolationTracker
from src.cloud_reader import CloudReader, build_violation_report
from src.metadata import extract_all_metadata
from src.evidence import EvidenceBuilder
from src.report_helper import build_report_package, write_review_page
from src.highway_filter import quick_scan_video
from src.preflight import load_dotenv, run_preflight
from src.config import (
    CLOUD_MODEL,
    EMERGENCY_X_THRESHOLD,
    FRAMES_PER_VIOLATION,
    JPEG_QUALITY,
    L1_FRAME_STRIDE,
    L15_FRAME_STRIDE,
    L15_MIN_GAP_SEC,
    L15_RIGHT_X_THRESHOLD,
    MIN_VIOLATION_FRAMES,
    MONTHLY_QUOTA,
    VIDEO_EXTENSIONS,
    YOLO_CONFIDENCE,
    YOLO_MODEL,
    default_output_dir,
)


def select_best_frames(video_path, violation, fps, detector,
                       num_frames=FRAMES_PER_VIOLATION, buffer_sec=3):
    """
    L1.5 智能取帧：从违法时间段选出最有价值的帧

    策略：
    1. 车牌面积最大的帧（车辆最近）
    2. 违法开始帧（证明开始时间）
    3. 违法结束帧（证明持续时间）
    4. 清晰度最高的帧
    """
    cap = cv2.VideoCapture(video_path)

    search_start = max(0, int((violation.start_frame / fps - buffer_sec) * fps))
    search_end = min(
        int(cap.get(cv2.CAP_PROP_FRAME_COUNT)),
        int((violation.end_frame / fps + buffer_sec * 2) * fps)
    )

    candidates = []  # (frame_idx, frame, score, reason)

    cap.set(cv2.CAP_PROP_POS_FRAMES, search_start)
    fi = search_start

    while fi < search_end:
        ret, frame = cap.read()
        if not ret:
            break
        fi += 1

        if fi % L15_FRAME_STRIDE != 0:
            continue

        dets = detector.detect_vehicles(frame)
        # 只看右侧（应急车道区域）的车辆
        right_dets = [
            d for d in dets
            if (d['bbox_norm'][0] + d['bbox_norm'][2]) / 2 >= L15_RIGHT_X_THRESHOLD
        ]

        if not right_dets:
            continue

        # 最大的右侧车辆
        biggest = max(right_dets, key=lambda d: d['area'])

        # 清晰度评分（拉普拉斯方差）
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        x1, y1, x2, y2 = biggest['bbox']
        roi = gray[max(0, y1):y2, max(0, x1):x2]
        sharpness = cv2.Laplacian(roi, cv2.CV_64F).var() if roi.size > 0 else 0

        # 综合评分：面积 * 清晰度
        score = biggest['area'] * (1 + sharpness / 1000)

        candidates.append((fi, frame.copy(), score, biggest['area'], sharpness))

    cap.release()

    if not candidates:
        return []

    # 按评分排序
    candidates.sort(key=lambda x: x[2], reverse=True)

    # 选帧策略：保证时间跨度 + 质量
    selected = []
    selected_times = []

    for fi, frame, score, area, sharp in candidates:
        t = fi / fps
        # 确保帧之间至少间隔 0.3 秒
        if any(abs(t - st) < L15_MIN_GAP_SEC for st in selected_times):
            continue
        selected.append((fi, frame))
        selected_times.append(t)
        if len(selected) >= num_frames:
            break

    return selected


def main():
    parser = argparse.ArgumentParser(description='高速违法举报助手 - 完整 Pipeline')
    parser.add_argument('video', nargs='*',
                        help='视频文件或目录路径（支持多个）')
    parser.add_argument('-o', '--output', default=str(default_output_dir()),
                        help='输出目录 (default: 项目 output/)')
    parser.add_argument('--model', default=YOLO_MODEL,
                        help='YOLO 模型路径')
    parser.add_argument('--cloud-model', default=CLOUD_MODEL,
                        help='云端视觉模型')
    parser.add_argument('--frames-per-violation', type=int, default=FRAMES_PER_VIOLATION,
                        help=f'每起违法送云端的帧数 (default: {FRAMES_PER_VIOLATION})')
    parser.add_argument('--quota', type=int, default=MONTHLY_QUOTA,
                        help=f'本月举报配额上限 (default: {MONTHLY_QUOTA}, 深圳月上限)')
    parser.add_argument('--dry-run', action='store_true',
                        help='只跑本地粗筛，不调云端')
    parser.add_argument('--api-key', default=None,
                        help='火山引擎 API Key（也可用环境变量 VOLCENGINE_API_KEY）')
    parser.add_argument('--check', action='store_true',
                        help='只检查运行环境，不分析视频')
    args = parser.parse_args()

    load_dotenv()
    if args.api_key:
        os.environ['VOLCENGINE_API_KEY'] = args.api_key

    errors = run_preflight(dry_run=args.dry_run, model_path=args.model)
    if errors:
        print('环境检查未通过：')
        for err in errors:
            print(f'  ❌ {err}')
        print('\n完整安装步骤见 测试说明.md')
        sys.exit(1)

    if args.check:
        print('✅ 环境检查通过，可以开始分析视频')
        return

    if not args.video:
        parser.error('请提供视频文件或目录，或使用 --check 只检查环境')

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    print('=' * 60)
    print('🚨 Highway Reporter - 完整分层 Pipeline')
    print('=' * 60)

    # ===== L0: 智能筛选高速路段 =====
    # 收集所有视频文件
    all_video_paths = []
    for v in args.video:
        if os.path.isdir(v):
            for f in sorted(Path(v).iterdir()):
                if f.suffix.lower() in VIDEO_EXTENSIONS and f.is_file():
                    all_video_paths.append(str(f))
        elif os.path.isfile(v):
            all_video_paths.append(v)

    if len(all_video_paths) > 1:
        print(f'\n[L0] 智能筛选 — 从 {len(all_video_paths)} 个视频中识别高速路段')
        print('     （零成本，每段视频只采样 5 帧）')

        highway_videos = []
        for vp in all_video_paths:
            seg = quick_scan_video(vp)
            icon = '🟢' if seg.is_highway else ('🟡' if seg.is_highway is None else '⚫')
            print(f'     {icon} {seg.filename:<35} '
                  f'{seg.duration_sec:>5.0f}s  置信度:{seg.highway_confidence:.0%}')
            if seg.is_highway is not False:  # 包含 True 和 None（待确认）
                highway_videos.append(vp)

        skipped = len(all_video_paths) - len(highway_videos)
        if skipped > 0:
            print(f'     跳过 {skipped} 个非高速视频')
        if not highway_videos:
            print('\n⚫ 未发现高速公路视频，分析结束')
            return
        all_video_paths = highway_videos
        print(f'     将分析 {len(all_video_paths)} 个高速视频')
    else:
        print(f'\n  单文件模式: {all_video_paths[0]}')

    # 逐个视频分析
    all_reports = []
    total_api_calls_all = 0

    for video_idx, video_path in enumerate(all_video_paths):
        if len(all_video_paths) > 1:
            print(f'\n{"="*60}')
            print(f'📹 [{video_idx+1}/{len(all_video_paths)}] {Path(video_path).name}')
            print(f'{"="*60}')

        reports, api_calls = analyze_single_video(
            video_path, output_dir, args, video_idx)
        all_reports.extend(reports)
        total_api_calls_all += api_calls

        # 配额检查：可举报的违法达到上限就停
        reportable_so_far = sum(1 for r in all_reports if r.can_report)
        if reportable_so_far >= args.quota:
            remaining = len(all_video_paths) - video_idx - 1
            if remaining > 0:
                print(f'\n🛑 已凑齐 {reportable_so_far} 起可举报违法'
                      f'（配额 {args.quota}），跳过剩余 {remaining} 个视频')
            break

    # 最终汇总
    if len(all_video_paths) > 1:
        print(f'\n{"="*60}')
        print(f'📋 全部视频汇总')
        print(f'{"="*60}')
        reportable = [r for r in all_reports if r.can_report]
        print(f'  分析视频: {min(video_idx + 1, len(all_video_paths))} 个')
        print(f'  总违法: {len(all_reports)} 起')
        print(f'  可举报: {len(reportable)} 起')
        print(f'  总 API 调用: {total_api_calls_all} 次')


def analyze_single_video(video_path, output_dir, args, video_idx=0):

    # ===== L1: 本地粗筛 =====
    print('\n[L1] 本地粗筛 — YOLO 车辆检测 + 应急车道追踪')
    print('     （零成本，本地 CPU 运行）')

    t0 = time.time()

    detector = VehicleDetector(model_name=args.model, confidence=YOLO_CONFIDENCE)
    tracker = ViolationTracker(
        emergency_x_threshold=EMERGENCY_X_THRESHOLD,
        min_violation_frames=MIN_VIOLATION_FRAMES,
    )

    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    duration = total / fps

    print(f'     视频: {Path(video_path).name}')
    print(f'     分辨率: {w}x{h}, FPS: {fps:.0f}, 时长: {duration:.1f}s')

    # 提取视频元数据（GPS/时间）
    print(f'     提取元数据...')
    video_meta = extract_all_metadata(video_path)
    if video_meta.get('gps_lat'):
        print(f'     GPS: {video_meta["gps_lat"]}, {video_meta["gps_lon"]}')
        if video_meta.get('address'):
            print(f'     地址: {video_meta["address"]}')
    if video_meta.get('creation_time'):
        print(f'     录制时间: {video_meta["creation_time"]}')

    frame_idx = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        frame_idx += 1
        if frame_idx % L1_FRAME_STRIDE != 0:
            continue
        dets = detector.detect_vehicles(frame)
        tracker.update(dets, frame.shape)
    cap.release()

    violations = tracker.get_violations()
    l1_time = time.time() - t0

    print(f'     完成: {len(violations)} 起违法, 耗时 {l1_time:.1f}s')
    print(f'     处理速度: {frame_idx / l1_time:.0f} 帧/秒'
          f' ({duration / l1_time:.1f}x 实时)')

    if not violations:
        print('\n✅ 未发现违法行为')
        return [], 0

    for v in violations:
        print(f'     违法 #{v.track_id}: '
              f'{v.start_frame / fps:.1f}s - {v.end_frame / fps:.1f}s '
              f'({v.vehicle_class})')

    # 配额检查
    if len(violations) > args.quota:
        print(f'\n⚠️  发现 {len(violations)} 起违法，超过配额 {args.quota}，'
              f'只处理前 {args.quota} 起')
        violations = violations[:args.quota]

    if args.dry_run:
        print('\n[DRY-RUN] 跳过云端精读')
        return [], 0

    # ===== L1.5: 智能取帧 =====
    print(f'\n[L1.5] 智能取帧 — 为每起违法选择最佳证据帧')
    print(f'       （零成本，本地运行）')

    t1 = time.time()
    violation_frames = {}  # violation_id -> [(frame_idx, frame)]

    for v in violations:
        frames = select_best_frames(
            video_path, v, fps, detector,
            num_frames=args.frames_per_violation,
        )
        violation_frames[v.track_id] = frames

        # 保存帧图片
        frame_dir = output_dir / f'violation_{v.track_id}'
        frame_dir.mkdir(exist_ok=True)
        for fi, frame in frames:
            fname = f'frame_{fi}_t{fi / fps:.1f}s.jpg'
            cv2.imwrite(str(frame_dir / fname), frame,
                        [cv2.IMWRITE_JPEG_QUALITY, JPEG_QUALITY])

        print(f'     违法 #{v.track_id}: 选出 {len(frames)} 帧')

    l15_time = time.time() - t1
    total_frames = sum(len(f) for f in violation_frames.values())
    print(f'     完成: 共 {total_frames} 帧, 耗时 {l15_time:.1f}s')

    # ===== L2: 云端精读 =====
    print(f'\n[L2] 云端精读 — {args.cloud_model}')
    print(f'     （按帧付费，预计 {total_frames + len(violations)} 次 API 调用）')

    t2 = time.time()
    cloud = CloudReader(model=args.cloud_model)
    reports = []
    total_api_calls = 0

    for v in violations:
        frames = violation_frames.get(v.track_id, [])
        frame_dir = output_dir / f'violation_{v.track_id}'
        frame_paths = sorted([
            str(frame_dir / f)
            for f in os.listdir(frame_dir) if f.endswith('.jpg')
        ])

        print(f'\n     --- 违法 #{v.track_id} ---')

        report = build_violation_report(cloud, v, frame_paths, fps)
        reports.append(report)
        total_api_calls += report.api_calls

        # 打印结果
        plate_str = report.plate_text or '未识别'
        if report.plate_votes:
            plate_str += f' (投票: {report.plate_votes})'
        print(f'     车牌: {plate_str}')
        print(f'     置信度: {report.plate_confidence:.0%}')
        print(f'     时间: {report.violation_time or "未提取"}')
        print(f'     颜色: {report.vehicle_color or "未识别"}')
        print(f'     车型: {report.vehicle_type or "未识别"}')
        print(f'     记录仪: {report.recorder_brand or "未识别"}')
        if report.issues:
            for issue in report.issues:
                print(f'     ⚠️  {issue}')
        status = '✅ 可举报' if report.can_report else '❌ 不可举报'
        print(f'     状态: {status}')

    l2_time = time.time() - t2

    # 如果有 GPS 元数据，补充到报告里
    for r in reports:
        if not r.violation_location and video_meta.get('address'):
            r.violation_location = video_meta['address']
        if not r.violation_time and video_meta.get('creation_time'):
            r.violation_time = video_meta['creation_time']

    # ===== L3: 证据包生成 =====
    print(f'\n[L3] 生成举报材料包')

    reportable = [r for r in reports if r.can_report]
    evidence_builder = EvidenceBuilder(output_dir=str(output_dir))

    if not reports:
        print('     无违法事件，跳过证据包')
    elif not reportable:
        print(f'     没有达到自动举报门槛的记录，仍为 {len(reports)} 起生成材料供人工确认')

    for r in reports:
        v = next(v for v in violations if v.track_id == r.violation_id)
        # 用 evidence.py 生成标准证据包（截图+视频片段）
        try:
            pkg = evidence_builder.build_evidence(
                video_path, v, fps,
                evidence_dir=str(output_dir / f'violation_{r.violation_id}'),
                plate_text=r.plate_text,
                plate_confidence=r.plate_confidence,
            )
            print(f'     违法 #{r.violation_id}: 证据包质量 {pkg.quality_score:.0%}')
            if pkg.issues:
                for issue in pkg.issues:
                    print(f'       ⚠️  {issue}')
        except Exception as e:
            print(f'     违法 #{r.violation_id}: 证据包生成失败 - {e}')

        # 生成举报辅助 HTML 页面
        try:
            report_data = {
                'plate_text': r.plate_text or '',
                'vehicle_class': r.vehicle_type or 'car',
                'violation_type': r.violation_type,
                'violation_time': r.violation_time or '',
                'violation_location': r.violation_location or '请根据GPS或路牌补充',
                'vehicle_color': r.vehicle_color or '',
                'duration_sec': r.duration_sec,
                'confidence': r.plate_confidence,
            }
            report_dir = str(output_dir / f'report_{r.violation_id}')
            evidence_dir = str(output_dir / f'violation_{r.violation_id}')
            build_report_package(report_data, evidence_dir, report_dir,
                                 platform='shenzhen')
            print(f'     举报辅助页面: {report_dir}/index.html')
        except Exception as e:
            print(f'     举报页面生成失败: {e}')

    # ===== 汇总 =====
    print('\n' + '=' * 60)
    print('📋 举报材料汇总')
    print('=' * 60)

    reportable = [r for r in reports if r.can_report]
    unreportable = [r for r in reports if not r.can_report]

    print(f'\n可举报: {len(reportable)} 起')
    for r in reportable:
        print(f'\n  🚨 违法 #{r.violation_id}')
        print(f'     车牌号码: {r.plate_text}')
        print(f'     违法类型: {r.violation_type}')
        print(f'     违法时间: {r.violation_time or "需手动填写"}')
        print(f'     违法地点: {r.violation_location or "需手动填写/GPS"}')
        print(f'     车辆颜色: {r.vehicle_color or "未识别"}')
        print(f'     车辆类型: {r.vehicle_type or "小型轿车"}')
        print(f'     车速: {r.speed or "-"}')
        print(f'     证据: {len(r.evidence_frames)} 张截图')

    if unreportable:
        print(f'\n不可举报: {len(unreportable)} 起')
        for r in unreportable:
            print(f'  ❌ 违法 #{r.violation_id}: '
                  f'{", ".join(r.issues)}')

    print(f'\n--- 性能 & 成本 ---')
    print(f'  L1 本地粗筛: {l1_time:.1f}s (免费)')
    print(f'  L1.5 智能取帧: {l15_time:.1f}s (免费)')
    print(f'  L2 云端精读: {l2_time:.1f}s ({total_api_calls} 次 API 调用)')
    print(f'  总耗时: {l1_time + l15_time + l2_time:.1f}s')

    # 保存报告 JSON
    output = {
        'video': video_path,
        'video_info': {
            'width': w, 'height': h, 'fps': fps,
            'duration': round(duration, 1),
        },
        'video_metadata': {
            'creation_time': video_meta.get('creation_time'),
            'gps_lat': video_meta.get('gps_lat'),
            'gps_lon': video_meta.get('gps_lon'),
            'address': video_meta.get('address'),
            'encoder': video_meta.get('encoder'),
        },
        'pipeline': {
            'l1_time': round(l1_time, 1),
            'l15_time': round(l15_time, 1),
            'l2_time': round(l2_time, 1),
            'l2_api_calls': total_api_calls,
            'cloud_model': args.cloud_model,
        },
        'violations': [
            {
                'id': r.violation_id,
                'can_report': r.can_report,
                'plate': r.plate_text,
                'plate_confidence': r.plate_confidence,
                'plate_votes': r.plate_votes,
                'color': r.vehicle_color,
                'vehicle_type': r.vehicle_type,
                'time': r.violation_time,
                'speed': r.speed,
                'location': r.violation_location,
                'recorder': r.recorder_brand,
                'start_sec': r.start_sec,
                'end_sec': r.end_sec,
                'duration_sec': r.duration_sec,
                'evidence_count': len(r.evidence_frames),
                'evidence_files': [os.path.basename(f) for f in r.evidence_frames],
                'issues': r.issues,
                'api_calls': r.api_calls,
            }
            for r in reports
        ],
    }

    report_path = output_dir / 'report.json'
    with open(report_path, 'w', encoding='utf-8') as f:
        json.dump(output, f, ensure_ascii=False, indent=2)

    print(f'\n完整报告: {report_path}')
    print(f'证据目录: {output_dir}')

    try:
        write_review_page(str(output_dir), output)
    except Exception as e:
        print(f'     结果页生成失败: {e}')

    return reports, total_api_calls


if __name__ == '__main__':
    main()
