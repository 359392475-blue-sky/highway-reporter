#!/usr/bin/env python3
"""
server.py - Highway Reporter L1 验收服务器（不跑云端 L2/L3）
启动: python server.py
访问: http://localhost:8199
"""
# CREATED_BY: highway-reporter MVP
# STATUS: active

import os
import sys
import json
import time
import threading
import queue
from pathlib import Path
from http.server import HTTPServer, SimpleHTTPRequestHandler
from urllib.parse import parse_qs, urlparse
import io

# 添加项目路径
sys.path.insert(0, str(Path(__file__).parent))

import cv2
import numpy as np

from src.config import (
    EMERGENCY_X_THRESHOLD,
    L1_FRAME_STRIDE,
    MIN_VIOLATION_FRAMES,
    WEB_PORT,
    YOLO_CONFIDENCE,
    YOLO_MODEL,
)

# 全局状态
analysis_state = {
    'status': 'idle',       # idle | loading | analyzing | done | error
    'progress': 0,          # 0-100
    'current_frame': 0,
    'total_frames': 0,
    'fps': 0,
    'elapsed': 0,
    'violations_count': 0,
    'frames': [],           # 已处理的帧数据
    'violations': [],       # 违法事件
    'video_info': {},
    'error': None,
}
state_lock = threading.Lock()

# 帧图片缓存（内存中的 JPEG bytes）
frame_cache = {}
frame_cache_lock = threading.Lock()

VIDEOS_DIR = Path(__file__).parent / 'data' / 'test-videos'
STATIC_DIR = Path(__file__).parent / 'web'


def get_video_list():
    """列出可用的测试视频"""
    videos = []
    if VIDEOS_DIR.exists():
        for f in sorted(VIDEOS_DIR.iterdir()):
            if f.suffix.lower() in ('.mp4', '.avi', '.mov', '.mkv'):
                cap = cv2.VideoCapture(str(f))
                if cap.isOpened():
                    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                    fps = cap.get(cv2.CAP_PROP_FPS)
                    frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                    dur = frames / fps if fps > 0 else 0
                    cap.release()
                    videos.append({
                        'filename': f.name,
                        'path': str(f),
                        'width': w,
                        'height': h,
                        'fps': round(fps, 1),
                        'frames': frames,
                        'duration': round(dur, 1),
                        'size_mb': round(f.stat().st_size / 1024 / 1024, 1),
                    })
    return videos


def run_analysis(video_path):
    """在后台线程跑分析"""
    global analysis_state, frame_cache

    from src.vehicle_detector import VehicleDetector
    from src.violation_tracker import ViolationTracker

    with state_lock:
        analysis_state = {
            'status': 'loading',
            'progress': 0,
            'current_frame': 0,
            'total_frames': 0,
            'fps': 0,
            'elapsed': 0,
            'violations_count': 0,
            'frames': [],
            'violations': [],
            'video_info': {},
            'error': None,
        }

    with frame_cache_lock:
        frame_cache.clear()

    try:
        # 初始化
        detector = VehicleDetector(
            model_name=YOLO_MODEL,
            confidence=YOLO_CONFIDENCE,
        )
        tracker = ViolationTracker(
            emergency_x_threshold=EMERGENCY_X_THRESHOLD,
            min_violation_frames=MIN_VIOLATION_FRAMES,
        )

        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise Exception('无法打开视频: ' + video_path)

        fps = cap.get(cv2.CAP_PROP_FPS) or 30
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        with state_lock:
            analysis_state['status'] = 'analyzing'
            analysis_state['total_frames'] = total
            analysis_state['fps'] = fps
            analysis_state['video_info'] = {
                'path': video_path,
                'filename': Path(video_path).name,
                'width': w, 'height': h,
                'fps': round(fps, 1),
                'total_frames': total,
                'duration': round(total / fps, 1),
            }

        start_time = time.time()
        frame_idx = 0
        process_interval = L1_FRAME_STRIDE
        sample_interval = 3     # 每3个处理帧输出1帧给前端

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            frame_idx += 1

            if frame_idx % process_interval != 0:
                continue

            processed_count = frame_idx // process_interval

            # 车辆检测
            detections = detector.detect_vehicles(frame)
            new_violations = tracker.update(detections, frame.shape)
            suspects = tracker.get_active_suspects()

            # 采样输出给前端
            if processed_count % sample_interval != 0:
                # 仍然更新进度
                with state_lock:
                    analysis_state['current_frame'] = frame_idx
                    analysis_state['progress'] = round(frame_idx / total * 100, 1)
                    analysis_state['elapsed'] = round(time.time() - start_time, 1)
                continue

            # 画标注
            annotated = frame.copy()

            # 应急车道区域
            x_thresh = int(w * 0.68)
            y_start = int(h * 0.35)
            y_end = int(h * 0.85)
            overlay = annotated.copy()
            cv2.rectangle(overlay, (x_thresh, y_start), (w, y_end), (0, 0, 255), -1)
            cv2.addWeighted(overlay, 0.12, annotated, 0.88, 0, annotated)
            cv2.line(annotated, (x_thresh, y_start), (x_thresh, y_end), (0, 0, 255), 1)

            # 检测框
            det_info = []
            for d in detections:
                x1, y1, x2, y2 = d['bbox']
                cx_norm = (d['bbox_norm'][0] + d['bbox_norm'][2]) / 2
                in_emergency = cx_norm >= 0.68
                color = (0, 0, 255) if in_emergency else (0, 255, 0)
                cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)
                label = d['class'] + ' ' + str(round(d['confidence'], 2))
                if in_emergency:
                    label += ' [!]'
                cv2.putText(annotated, label, (x1, y1 - 5),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)
                det_info.append({
                    'bbox': d['bbox'],
                    'class': d['class'],
                    'conf': round(d['confidence'], 2),
                    'in_emergency': in_emergency,
                    'cx_norm': round(cx_norm, 3),
                })

            # 追踪标注
            for s in suspects:
                x1, y1, x2, y2 = s.last_bbox
                cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 0, 255), 3)
                txt = 'VIOLATION' if s.violation_confirmed else 'SUSPECT'
                txt += ' ' + str(s.frames_in_emergency) + 'f'
                cv2.putText(annotated, txt, (x1, y1 - 20),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

            # 编码为 JPEG
            _, buf = cv2.imencode('.jpg', annotated, [cv2.IMWRITE_JPEG_QUALITY, 85])
            frame_key = 'frame_{:06d}'.format(frame_idx)

            with frame_cache_lock:
                frame_cache[frame_key] = buf.tobytes()

            t_sec = round(frame_idx / fps, 2)
            frame_entry = {
                'idx': frame_idx,
                'time': t_sec,
                'key': frame_key,
                'detections': det_info,
                'suspects': len(suspects),
                'has_violation': len(new_violations) > 0,
            }

            with state_lock:
                analysis_state['frames'].append(frame_entry)
                analysis_state['current_frame'] = frame_idx
                analysis_state['progress'] = round(frame_idx / total * 100, 1)
                analysis_state['elapsed'] = round(time.time() - start_time, 1)

                # 更新违法事件
                for nv in new_violations:
                    analysis_state['violations'].append({
                        'track_id': nv.track_id,
                        'plate': nv.plate_text,
                        'vehicle_class': nv.vehicle_class,
                        'start_sec': round(nv.start_frame / fps, 2),
                        'end_sec': round(nv.end_frame / fps, 2),
                        'duration_frames': nv.duration_frames,
                        'confidence': round(nv.confidence, 2),
                    })
                    analysis_state['violations_count'] = len(analysis_state['violations'])

        cap.release()

        with state_lock:
            analysis_state['status'] = 'done'
            analysis_state['progress'] = 100
            analysis_state['elapsed'] = round(time.time() - start_time, 1)

    except Exception as e:
        import traceback
        with state_lock:
            analysis_state['status'] = 'error'
            analysis_state['error'] = str(e)
        traceback.print_exc()


class APIHandler(SimpleHTTPRequestHandler):
    """处理 API 请求和静态文件"""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, directory=str(STATIC_DIR), **kwargs)

    def do_GET(self):
        parsed = urlparse(self.path)
        path = parsed.path

        if path == '/api/videos':
            self._json_response(get_video_list())

        elif path == '/api/status':
            with state_lock:
                # 返回不含 frames 的轻量状态
                status = {k: v for k, v in analysis_state.items()
                          if k != 'frames'}
                status['frames_count'] = len(analysis_state['frames'])
            self._json_response(status)

        elif path == '/api/frames':
            # 返回帧列表（支持分页）
            params = parse_qs(parsed.query)
            since = int(params.get('since', ['0'])[0])
            with state_lock:
                frames = analysis_state['frames'][since:]
                total_violations = analysis_state['violations_count']
            self._json_response({
                'frames': frames,
                'offset': since,
                'total_violations': total_violations,
            })

        elif path == '/api/violations':
            with state_lock:
                self._json_response(analysis_state['violations'])

        elif path.startswith('/api/frame/'):
            # 返回帧图片
            frame_key = path.split('/api/frame/')[1]
            with frame_cache_lock:
                img_bytes = frame_cache.get(frame_key)
            if img_bytes:
                self.send_response(200)
                self.send_header('Content-Type', 'image/jpeg')
                self.send_header('Content-Length', str(len(img_bytes)))
                self.send_header('Cache-Control', 'public, max-age=3600')
                self.end_headers()
                self.wfile.write(img_bytes)
            else:
                self.send_error(404, 'Frame not found')

        elif path == '/api/result':
            # 完整结果（分析完成后用）
            with state_lock:
                self._json_response({
                    'status': analysis_state['status'],
                    'video_info': analysis_state['video_info'],
                    'violations': analysis_state['violations'],
                    'frames': analysis_state['frames'],
                    'elapsed': analysis_state['elapsed'],
                })

        elif path.startswith('/api/evidence/'):
            # 返回证据图片 /api/evidence/<violation_id>/<filename>
            parts = path.split('/api/evidence/')[1].split('/')
            if len(parts) == 2:
                v_id, fname = parts
                # 查找最近的输出目录
                for d in ['full_output_v2', 'full_output', 'dry_run_output']:
                    fpath = VIDEOS_DIR / '..' / '..' / d / f'violation_{v_id}' / fname
                    if fpath.exists():
                        data = fpath.read_bytes()
                        self.send_response(200)
                        ct = 'image/jpeg' if fname.endswith('.jpg') else 'application/octet-stream'
                        self.send_header('Content-Type', ct)
                        self.send_header('Content-Length', str(len(data)))
                        self.end_headers()
                        self.wfile.write(data)
                        return
            self.send_error(404)

        else:
            # 静态文件
            super().do_GET()

    def do_POST(self):
        parsed = urlparse(self.path)

        if parsed.path == '/api/analyze':
            content_len = int(self.headers.get('Content-Length', 0))
            body = json.loads(self.rfile.read(content_len)) if content_len > 0 else {}
            video_path = body.get('video_path', '')

            if not video_path or not os.path.exists(video_path):
                self._json_response({'error': '视频文件不存在'}, 400)
                return

            # 检查是否已在运行
            with state_lock:
                if analysis_state['status'] == 'analyzing' or analysis_state['status'] == 'loading':
                    self._json_response({'error': '已有分析任务在运行'}, 409)
                    return

            # 清空旧状态和缓存
            with state_lock:
                analysis_state['status'] = 'idle'
                analysis_state['frames'] = []
                analysis_state['violations'] = []
                analysis_state['violations_count'] = 0
                analysis_state['progress'] = 0
                analysis_state['error'] = None
            with frame_cache_lock:
                frame_cache.clear()

            # 启动后台分析
            t = threading.Thread(target=run_analysis, args=(video_path,), daemon=True)
            t.start()
            self._json_response({'message': '分析已启动', 'video': video_path})

        else:
            self.send_error(404)

    def _json_response(self, data, code=200):
        body = json.dumps(data, ensure_ascii=False).encode('utf-8')
        self.send_response(code)
        self.send_header('Content-Type', 'application/json; charset=utf-8')
        self.send_header('Content-Length', str(len(body)))
        self.send_header('Access-Control-Allow-Origin', '*')
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, format, *args):
        # 静默常规请求日志，只打印错误
        if '404' in str(args) or '500' in str(args):
            super().log_message(format, *args)


def main():
    port = WEB_PORT
    STATIC_DIR.mkdir(parents=True, exist_ok=True)

    server = HTTPServer(('0.0.0.0', port), APIHandler)
    print(f'\n🚨 Highway Reporter 分析服务器')
    print(f'   地址: http://localhost:{port}')
    print(f'   视频目录: {VIDEOS_DIR}')
    print(f'   Ctrl+C 停止\n')

    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print('\n服务器已停止')
        server.server_close()


if __name__ == '__main__':
    main()
