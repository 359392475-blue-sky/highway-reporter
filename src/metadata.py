"""
metadata.py - 视频元数据提取模块
从行车记录仪 MP4 文件中提取 GPS 坐标、时间信息

大多数记录仪（70迈、360、盯盯拍、特斯拉等）会在 MP4 文件中嵌入：
- GPS 坐标（经纬度）
- 录制时间
- 速度信息
- 设备信息

提取方式：
1. ffprobe 读取 metadata
2. Python 解析 MP4 atom（moov/udta）
3. 特殊格式解析（如 NMEA GPS 流）
"""
# CREATED_BY: highway-reporter MVP
# STATUS: active

import subprocess
import json
import re
import os
import struct
from pathlib import Path
from typing import Optional, Dict, List
from datetime import datetime

from .preflight import find_ffmpeg, find_ffprobe


def extract_with_ffprobe(video_path: str) -> Dict:
    """用 ffprobe 提取视频元数据"""
    result = {
        'creation_time': None,
        'gps_lat': None,
        'gps_lon': None,
        'duration': None,
        'width': None,
        'height': None,
        'fps': None,
        'encoder': None,
        'raw_metadata': {},
    }

    try:
        ffprobe = find_ffprobe()
        if not ffprobe:
            return result
        cmd = [
            ffprobe, '-v', 'quiet',
            '-print_format', 'json',
            '-show_format', '-show_streams',
            video_path
        ]
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=10)
        if proc.returncode != 0:
            return result

        data = json.loads(proc.stdout)

        # Format metadata
        fmt = data.get('format', {})
        tags = fmt.get('tags', {})
        result['raw_metadata'] = tags
        result['duration'] = float(fmt.get('duration', 0))

        # 常见的时间字段
        for key in ['creation_time', 'date', 'creation_date']:
            if key in tags:
                result['creation_time'] = tags[key]
                break

        # GPS 坐标 — 不同厂商用不同的 tag
        # 常见格式：
        #   location: +31.2345+121.4567/
        #   location-eng: +31.2345+121.4567/
        #   com.apple.quicktime.location.ISO6709: +31.2345+121.4567+023.456/
        for key in ['location', 'location-eng',
                    'com.apple.quicktime.location.ISO6709']:
            if key in tags:
                lat, lon = _parse_iso6709(tags[key])
                if lat is not None:
                    result['gps_lat'] = lat
                    result['gps_lon'] = lon
                break

        # 视频流信息
        for stream in data.get('streams', []):
            if stream.get('codec_type') == 'video':
                result['width'] = stream.get('width')
                result['height'] = stream.get('height')
                fps_str = stream.get('r_frame_rate', '0/1')
                if '/' in fps_str:
                    num, den = fps_str.split('/')
                    result['fps'] = round(int(num) / max(1, int(den)), 1)
                # 流级别的 tags
                stream_tags = stream.get('tags', {})
                if not result['creation_time'] and 'creation_time' in stream_tags:
                    result['creation_time'] = stream_tags['creation_time']
                break

        # Encoder
        result['encoder'] = tags.get('encoder', tags.get('handler_name', ''))

    except (subprocess.TimeoutExpired, FileNotFoundError, json.JSONDecodeError):
        pass

    return result


def _parse_iso6709(location_str: str):
    """
    解析 ISO 6709 格式的坐标
    格式：+31.2345+121.4567/ 或 +31.2345+121.4567+023.456/
    """
    pattern = re.compile(r'([+-]\d+\.?\d*)')
    matches = pattern.findall(location_str)
    if len(matches) >= 2:
        return float(matches[0]), float(matches[1])
    return None, None


def extract_gps_from_subtitle(video_path: str) -> List[Dict]:
    """
    一些记录仪把 GPS 数据存在字幕流（subtitle track）中
    格式通常是 NMEA 或自定义格式
    """
    gps_points = []

    try:
        # 检查是否有字幕流
        ffprobe = find_ffprobe()
        ffmpeg = find_ffmpeg()
        if not ffprobe or not ffmpeg:
            return gps_points
        cmd = [
            ffprobe, '-v', 'quiet',
            '-print_format', 'json',
            '-show_streams', video_path
        ]
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=10)
        data = json.loads(proc.stdout)

        sub_streams = [
            s for s in data.get('streams', [])
            if s.get('codec_type') == 'subtitle'
            or 'data' in s.get('codec_type', '')
        ]

        if not sub_streams:
            return gps_points

        # 提取字幕内容
        for sub in sub_streams:
            idx = sub.get('index', 0)
            cmd = [
                ffmpeg, '-y', '-v', 'quiet',
                '-i', video_path,
                '-map', '0:%d' % idx,
                '-f', 'srt', '/tmp/gps_sub.srt'
            ]
            subprocess.run(cmd, capture_output=True, timeout=10)

            if os.path.exists('/tmp/gps_sub.srt'):
                with open('/tmp/gps_sub.srt', 'r', errors='ignore') as f:
                    content = f.read()

                # 尝试解析 NMEA GPRMC
                for match in re.finditer(
                    r'\$GPRMC,(\d+\.?\d*),A,'
                    r'(\d+\.?\d*),(N|S),'
                    r'(\d+\.?\d*),(E|W)',
                    content
                ):
                    time_str, lat_raw, lat_dir, lon_raw, lon_dir = match.groups()
                    lat = _nmea_to_decimal(float(lat_raw), lat_dir)
                    lon = _nmea_to_decimal(float(lon_raw), lon_dir)
                    gps_points.append({'lat': lat, 'lon': lon, 'time': time_str})

                # 也试试常见的中文记录仪格式
                # 如: "N:31.2345 E:121.4567 V:60km/h"
                for match in re.finditer(
                    r'[NS]:?\s*(\d+\.?\d+)\s*[EW]:?\s*(\d+\.?\d+)',
                    content
                ):
                    lat, lon = float(match.group(1)), float(match.group(2))
                    gps_points.append({'lat': lat, 'lon': lon})

    except (subprocess.TimeoutExpired, FileNotFoundError, json.JSONDecodeError):
        pass

    return gps_points


def _nmea_to_decimal(raw: float, direction: str) -> float:
    """NMEA 格式转十进制度"""
    degrees = int(raw / 100)
    minutes = raw - degrees * 100
    decimal = degrees + minutes / 60
    if direction in ('S', 'W'):
        decimal = -decimal
    return round(decimal, 6)


def gps_to_address(lat: float, lon: float) -> Optional[str]:
    """GPS 坐标反查地址（需要网络）"""
    # 用高德/百度/Nominatim 反查
    try:
        import urllib.request
        url = (
            f'https://nominatim.openstreetmap.org/reverse'
            f'?lat={lat}&lon={lon}&format=json&zoom=16'
            f'&accept-language=zh'
        )
        req = urllib.request.Request(url, headers={
            'User-Agent': 'HighwayReporter/1.0'
        })
        resp = urllib.request.urlopen(req, timeout=10)
        data = json.loads(resp.read())
        return data.get('display_name', '')
    except Exception:
        return None


def extract_all_metadata(video_path: str) -> Dict:
    """提取视频的所有可用元数据"""
    result = extract_with_ffprobe(video_path)

    # 补充 GPS 数据
    if not result['gps_lat']:
        gps_points = extract_gps_from_subtitle(video_path)
        if gps_points:
            result['gps_lat'] = gps_points[0]['lat']
            result['gps_lon'] = gps_points[0]['lon']
            result['gps_track'] = gps_points

    # 如果有 GPS，尝试反查地址
    if result['gps_lat'] and result['gps_lon']:
        address = gps_to_address(result['gps_lat'], result['gps_lon'])
        if address:
            result['address'] = address

    return result


if __name__ == '__main__':
    import sys
    if len(sys.argv) < 2:
        print("Usage: python metadata.py <video_path>")
        sys.exit(1)

    path = sys.argv[1]
    meta = extract_all_metadata(path)

    print(json.dumps(meta, ensure_ascii=False, indent=2))
