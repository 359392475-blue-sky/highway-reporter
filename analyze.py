#!/usr/bin/env python3
"""
analyze.py - 旧版 CLI（已废弃）

请改用: python3 full_pipeline.py <video.mp4>
本入口走本地 PaddleOCR，主路径已换成云端 L2。
"""
# CREATED_BY: highway-reporter MVP
# STATUS: deprecated

import argparse
import sys
import warnings
from pathlib import Path

# 添加项目根目录到 path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from src.pipeline import Pipeline
from src.config import (
    EMERGENCY_X_THRESHOLD,
    L1_FRAME_STRIDE,
    MIN_VIOLATION_FRAMES,
    YOLO_CONFIDENCE,
    YOLO_MODEL,
)


def main():
    warnings.warn(
        'analyze.py 已废弃，请使用 full_pipeline.py（本地 YOLO + 云端精读）',
        DeprecationWarning,
        stacklevel=2,
    )
    parser = argparse.ArgumentParser(
        description='[已废弃] 请使用 full_pipeline.py',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  %(prog)s highway_clip.mp4                    # 基本分析
  %(prog)s highway_clip.mp4 --debug-video      # 生成标注视频
  %(prog)s highway_clip.mp4 -o ./evidence      # 指定输出目录
  %(prog)s highway_clip.mp4 --max-frames 300   # 只处理前300帧（测试）
  %(prog)s highway_clip.mp4 --threshold 0.65   # 调整应急车道阈值
        """
    )

    parser.add_argument('video', help='视频文件路径')
    parser.add_argument('-o', '--output', default='./output',
                        help='输出目录 (default: ./output)')
    parser.add_argument('--model', default=YOLO_MODEL,
                        help='YOLO 模型')
    parser.add_argument('--confidence', type=float, default=YOLO_CONFIDENCE,
                        help=f'检测置信度阈值 (default: {YOLO_CONFIDENCE})')
    parser.add_argument('--threshold', type=float, default=EMERGENCY_X_THRESHOLD,
                        help=f'应急车道 x 坐标阈值 (default: {EMERGENCY_X_THRESHOLD})')
    parser.add_argument('--min-frames', type=int, default=MIN_VIOLATION_FRAMES,
                        help=f'确认违法的最少连续帧数 (default: {MIN_VIOLATION_FRAMES})')
    parser.add_argument('--skip', type=int, default=L1_FRAME_STRIDE,
                        help=f'帧处理间隔 (default: {L1_FRAME_STRIDE})')
    parser.add_argument('--plate-interval', type=int, default=5,
                        help='车牌识别间隔 (default: 每5个处理帧)')
    parser.add_argument('--max-frames', type=int, default=None,
                        help='最大处理帧数（调试用）')
    parser.add_argument('--debug-video', action='store_true',
                        help='生成标注 debug 视频')
    parser.add_argument('--no-evidence', action='store_true',
                        help='不保存证据截图')

    args = parser.parse_args()

    # 检查视频文件
    video_path = Path(args.video)
    if not video_path.exists():
        print(f"错误: 视频文件不存在: {video_path}")
        sys.exit(1)

    # 创建 pipeline
    pipeline = Pipeline(
        output_dir=args.output,
        yolo_model=args.model,
        detection_confidence=args.confidence,
        emergency_x_threshold=args.threshold,
        min_violation_frames=args.min_frames,
        frame_skip=args.skip,
        plate_check_interval=args.plate_interval,
        save_evidence=not args.no_evidence,
        save_debug_video=args.debug_video,
    )

    # 分析
    result = pipeline.analyze_video(
        str(video_path),
        max_frames=args.max_frames,
    )

    # 退出码：有违法=0，无违法=0，出错=1
    print(f"\n完成！发现 {len(result.violations)} 起违法行为")
    if result.evidence_dir:
        print(f"证据保存在: {result.evidence_dir}")


if __name__ == '__main__':
    main()
