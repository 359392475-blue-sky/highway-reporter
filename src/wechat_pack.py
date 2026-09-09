"""Pack report materials so a phone can open them inside WeChat.

Primary path: one self-contained HTML per violation → 文件传输助手.
Optional: serve the same folder on the LAN when the phone is on the
home router (PC may be wired; that still counts as the same LAN).
"""
from __future__ import annotations

import base64
import html
import io
import socket
import sys
import threading
from dataclasses import dataclass, field
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Iterable, List, Optional, Sequence

from PIL import Image

from .report_platforms import copy_all

GUIDE_NAME = '先看这里.txt'
MAX_IMAGES = 4
MAX_IMAGE_SIDE = 960
JPEG_QUALITY = 62

GUIDE_TEXT = """怎么把材料弄到手机上

1. 打开电脑上的微信
2. 点「文件传输助手」（就是自己跟自己聊天那个）
3. 把这个黄色说明旁边的网页文件，拖进对话框，点发送
   一次只发一个。交完一辆，再发下一辆
4. 拿出手机，打开微信，点文件传输助手，点开刚发来的文件
5. 按手机上的「复制这一栏」，贴到交警微信里
6. 手指按住照片不放，保存到相册。交的时候从相册选这张图

电脑没有无线网也没关系。手机用流量也能收到。
"""

_server_lock = threading.Lock()
_server: Optional[ThreadingHTTPServer] = None
_server_thread: Optional[threading.Thread] = None
_server_dir: Optional[Path] = None
_server_port: Optional[int] = None


@dataclass
class PackResult:
    folder: Path
    html_files: List[Path] = field(default_factory=list)
    txt_files: List[Path] = field(default_factory=list)
    guide: Optional[Path] = None


def evidence_dir(out_dir: Path, violation: dict) -> Path:
    return Path(out_dir) / f"violation_{violation.get('id')}"


def collect_images(out_dir: Path, violation: dict) -> List[Path]:
    folder = evidence_dir(out_dir, violation)
    found: List[Path] = []
    seen = set()

    def add(path: Path) -> None:
        if not path.is_file():
            return
        key = path.resolve()
        if key in seen:
            return
        seen.add(key)
        found.append(path)

    for name in violation.get('evidence_files') or []:
        add(folder / Path(name).name)
    if folder.is_dir():
        for path in sorted(folder.glob('scene_*.jpg')):
            add(path)
        for path in sorted(folder.glob('frame_*.jpg')):
            add(path)
        for path in sorted(folder.glob('*.jpg')):
            add(path)
    return found[:MAX_IMAGES]


def encode_image_data_uri(path: Path) -> str:
    img = Image.open(path).convert('RGB')
    w, h = img.size
    scale = MAX_IMAGE_SIDE / max(w, h)
    if scale < 1:
        img = img.resize((max(1, int(w * scale)), max(1, int(h * scale))), Image.LANCZOS)
    buf = io.BytesIO()
    img.save(buf, format='JPEG', quality=JPEG_QUALITY, optimize=True)
    b64 = base64.b64encode(buf.getvalue()).decode('ascii')
    return f'data:image/jpeg;base64,{b64}'


def _html_page(title: str, body: str) -> str:
    return f"""<!DOCTYPE html>
<html lang="zh-CN">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1, maximum-scale=1">
<title>{html.escape(title)}</title>
<style>
body {{ margin:0; background:#F4EFE4; color:#1C1914; font:16px/1.55 -apple-system,"PingFang SC","Noto Sans SC",sans-serif; }}
.wrap {{ padding:18px 16px 48px; max-width:560px; margin:0 auto; }}
h1 {{ font-size:20px; margin:0 0 8px; }}
.sub,.tip {{ color:#6B6458; font-size:14px; margin:0 0 12px; }}
.card {{ background:#FFFBF5; border:1px solid #D9D0C0; border-radius:10px; padding:14px; margin:12px 0; }}
.lab {{ color:#6B6458; font-size:13px; margin-bottom:6px; }}
textarea {{ width:100%; box-sizing:border-box; min-height:64px; padding:10px; font-size:16px; border:1px solid #D9D0C0; border-radius:8px; background:#fff; color:#1C1914; }}
button {{ margin-top:8px; width:100%; border:0; background:#C9A227; color:#1C1914; font-size:16px; padding:10px; border-radius:8px; }}
img {{ width:100%; border-radius:8px; display:block; }}
figcaption {{ color:#6B6458; font-size:13px; margin-top:6px; }}
.step {{ background:#2A2722; color:#FFFBF5; padding:12px 14px; border-radius:10px; margin-bottom:14px; font-size:14px; }}
</style>
</head>
<body>
<div class="wrap">
{body}
</div>
<script>
function cp(id){{
  var el=document.getElementById(id);
  el.focus(); el.select();
  try {{ document.execCommand('copy'); }} catch(e) {{}}
  alert('已抄下来。回到交警微信里，长按输入框，选粘贴。');
}}
</script>
</body>
</html>
"""


def render_violation_html(
    violation: dict,
    rows: Sequence[tuple],
    images: Sequence[Path],
    region: dict,
    index: int = 1,
    total: int = 1,
) -> str:
    plate = violation.get('plate') or '车牌没看清'
    place = f"{region.get('province', '')} {region.get('city', '')}".strip()
    how = region.get('channel') or '打开当地交警的微信'
    fields = []
    for i, (label, value) in enumerate(rows):
        text = html.escape(value or '还没有，请自己补')
        fields.append(
            f'<div class="card"><div class="lab">{html.escape(label)}</div>'
            f'<textarea id="f{i}" readonly>{text}</textarea>'
            f'<button type="button" onclick="cp(\'f{i}\')">复制这一栏</button></div>'
        )
    pics = []
    for n, path in enumerate(images, 1):
        try:
            uri = encode_image_data_uri(path)
        except OSError:
            continue
        pics.append(
            f'<figure class="card"><img src="{uri}" alt="照片{n}">'
            f'<figcaption>第 {n} 张照片 · 手指按住不放，保存到相册。交的时候从相册选这张。</figcaption></figure>'
        )
    if not pics:
        pics.append('<p class="tip">这辆车没有照片。请回电脑，把照片单独发到微信「文件传输助手」。</p>')
    body = f"""
<p class="sub">第 {index} 辆，一共 {total} 辆 · 交到 {html.escape(place)}</p>
<h1>占用应急车道 · {html.escape(str(plate))}</h1>
<div class="step">先打开交警的微信，再回到这里一栏一栏复制。<br>{html.escape(how)}</div>
{''.join(fields)}
<h1 style="margin-top:20px;font-size:18px;">照片（交的时候要上传）</h1>
{''.join(pics)}
"""
    return _html_page(f'举报 {plate}', body)


def write_wechat_pack(
    out_dir: Path,
    violations: Iterable[dict],
    region: dict,
) -> PackResult:
    out_dir = Path(out_dir)
    folder = out_dir / 'wechat'
    if folder.exists():
        for old in folder.glob('report-*.html'):
            old.unlink()
        for old in folder.glob('report-*.txt'):
            old.unlink()
    folder.mkdir(parents=True, exist_ok=True)
    items = list(violations)
    result = PackResult(folder=folder)
    guide = folder / GUIDE_NAME
    guide.write_text(GUIDE_TEXT, encoding='utf-8')
    result.guide = guide
    total = max(len(items), 1)
    for i, violation in enumerate(items, 1):
        vid = violation.get('id', i)
        rows = region['fields'](violation)
        images = collect_images(out_dir, violation)
        html_path = folder / f'report-{vid}.html'
        html_path.write_text(
            render_violation_html(violation, rows, images, region, i, total),
            encoding='utf-8',
        )
        txt_path = folder / f'report-{vid}.txt'
        header = (
            f"{region.get('title') or region.get('city')}\n"
            f"{region.get('channel')}\n"
            f"{region.get('hint')}\n\n"
        )
        txt_path.write_text(header + copy_all(rows), encoding='utf-8')
        result.html_files.append(html_path)
        result.txt_files.append(txt_path)
    return result


def lan_ips() -> List[str]:
    ips: List[str] = []
    try:
        sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        sock.connect(('223.5.5.5', 80))
        ip = sock.getsockname()[0]
        sock.close()
        if ip and not ip.startswith('127.'):
            ips.append(ip)
    except OSError:
        pass
    try:
        for info in socket.getaddrinfo(socket.gethostname(), None, socket.AF_INET):
            ip = info[4][0]
            if ip not in ips and not ip.startswith('127.'):
                ips.append(ip)
    except OSError:
        pass
    return ips


def lan_urls(filename: str, port: int) -> List[str]:
    return [f'http://{ip}:{port}/{filename}' for ip in lan_ips()]


def ensure_pack_server(folder: Path, port: int = 8765) -> int:
    """Serve the wechat folder. Safe to call repeatedly."""
    global _server, _server_thread, _server_dir, _server_port
    folder = Path(folder).resolve()
    with _server_lock:
        if _server and _server_dir == folder and _server_port:
            return _server_port
        if _server:
            try:
                _server.shutdown()
            except OSError:
                pass
            _server = None
        handler_dir = str(folder)

        class QuietHandler(SimpleHTTPRequestHandler):
            def __init__(self, *args, **kwargs):
                super().__init__(*args, directory=handler_dir, **kwargs)

            def log_message(self, format, *args):
                return

        last_error = None
        for candidate in range(port, port + 8):
            try:
                httpd = ThreadingHTTPServer(('0.0.0.0', candidate), QuietHandler)
            except OSError as exc:
                last_error = exc
                continue
            thread = threading.Thread(target=httpd.serve_forever, daemon=True)
            thread.start()
            _server = httpd
            _server_thread = thread
            _server_dir = folder
            _server_port = candidate
            return candidate
        raise OSError(last_error or '无法打开本地端口')


def reveal_path(path: Path) -> None:
    path = Path(path)
    if sys.platform == 'darwin':
        subprocess_open(['open', '-R', str(path)])
    elif sys.platform == 'win32':
        subprocess_open(['explorer', '/select,', str(path)])
    else:
        subprocess_open(['xdg-open', str(path.parent)])


def open_wechat() -> bool:
    if sys.platform == 'darwin':
        for name in ('WeChat', '微信', 'Weixin'):
            if subprocess_open(['open', '-a', name]) == 0:
                return True
        return False
    if sys.platform == 'win32':
        return False
    return False


def subprocess_open(cmd: list) -> int:
    import subprocess
    try:
        return subprocess.run(cmd, capture_output=True, check=False).returncode
    except OSError:
        return 1
