"""
report_helper.py - 举报辅助包生成模块
目标：让用户在手机上举报时，操作降到最简

生成物：
1. 一个 HTML 举报辅助页面（手机浏览器打开）
   - 每个字段旁边有"复制"按钮，点一下就复制到剪贴板
   - 证据图片直接显示，长按可保存到相册
   - 视频可直接下载/保存
   - 分步骤指引：第1步打开12123→第2步点XX→...
2. 证据文件打包（图片+视频）
3. 二维码（扫码直接打开辅助页面）

用户体验：
  电脑扫描完 → 手机扫二维码 → 打开页面 → 照着一步步来
  需要填的内容全部可以一键复制，图片长按保存
"""
# CREATED_BY: highway-reporter MVP
# STATUS: active

import json
import base64
import os
import http.server
import threading
import socket
from pathlib import Path
from typing import Optional, List
from datetime import datetime

from .config import REPORT_SERVER_PORT

try:
    import qrcode
    HAS_QRCODE = True
except ImportError:
    HAS_QRCODE = False


def get_local_ip():
    """获取本机局域网 IP"""
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(("8.8.8.8", 80))
        ip = s.getsockname()[0]
        s.close()
        return ip
    except Exception:
        return "127.0.0.1"


# 深圳随手e拍举报步骤
SHENZHEN_STEPS = [
    {
        'title': '打开微信',
        'desc': '搜索公众号"深圳交警"，点击底部菜单"星级用户"→"随手e拍"',
        'icon': '📱',
    },
    {
        'title': '选择举报类型',
        'desc': '选择"占用应急车道"',
        'icon': '📋',
    },
    {
        'title': '填写违法信息',
        'desc': '按下方信息逐项填写（点"复制"按钮可一键复制）',
        'icon': '✏️',
    },
    {
        'title': '上传证据',
        'desc': '从相册选择下方的证据图片和视频（先长按保存到相册）',
        'icon': '📸',
    },
    {
        'title': '确认提交',
        'desc': '检查信息无误后提交，等待审核（1-2个工作日）',
        'icon': '✅',
    },
]

# 12123 APP 举报步骤
APP_12123_STEPS = [
    {
        'title': '打开交管12123',
        'desc': '首页 → 更多 → 交通违法举报（部分城市可能在不同位置）',
        'icon': '📱',
    },
    {
        'title': '选择举报类型',
        'desc': '选择"机动车违法" → "占用应急车道"',
        'icon': '📋',
    },
    {
        'title': '填写违法信息',
        'desc': '按下方信息逐项填写（点"复制"按钮可一键复制）',
        'icon': '✏️',
    },
    {
        'title': '上传证据',
        'desc': '选择下方已保存到相册的证据图片（至少2张）和视频',
        'icon': '📸',
    },
    {
        'title': '确认提交',
        'desc': '检查无误后提交',
        'icon': '✅',
    },
]

# 广州随手拍举报步骤
GUANGZHOU_STEPS = [
    {
        'title': '打开微信',
        'desc': '搜索公众号"广州交警" → 底部菜单"违法举报" → "随手拍"',
        'icon': '📱',
    },
    {
        'title': '填写违法信息',
        'desc': '按下方信息逐项填写（点"复制"按钮可一键复制）',
        'icon': '✏️',
    },
    {
        'title': '上传证据',
        'desc': '选择下方已保存到相册的证据图片和视频',
        'icon': '📸',
    },
    {
        'title': '确认提交',
        'desc': '提交后等待审核（高速大队1-2天）',
        'icon': '✅',
    },
]

PLATFORM_STEPS = {
    'shenzhen': SHENZHEN_STEPS,
    '12123': APP_12123_STEPS,
    'guangzhou': GUANGZHOU_STEPS,
}


def _render_evidence_images(evidence_images):
    """Render evidence image tags (Python 3.9 compatible)."""
    parts = []
    for i, f in enumerate(evidence_images):
        alt = '证据' + str(i + 1)
        parts.append(
            '<img class="evidence-img" src="evidence/' + f
            + '" onclick="previewImg(this)" alt="' + alt + '">'
        )
    return ''.join(parts)


def _render_evidence_videos(evidence_videos):
    """Render evidence video sections (Python 3.9 compatible)."""
    parts = []
    for f in evidence_videos:
        parts.append(
            '<div class="section">\n'
            '    <h2>🎬 证据视频</h2>\n'
            '    <a class="video-link" href="evidence/' + f
            + '" download>📥 点击下载视频：' + f + '</a>\n'
            '    <div class="save-hint">💡 下载后在相册中找到视频，举报时上传</div>\n'
            '</div>\n'
        )
    return ''.join(parts)


def generate_report_html(violation_data: dict, evidence_dir: str,
                         platform: str = '12123') -> str:
    """
    生成举报辅助 HTML 页面

    Args:
        violation_data: {
            'plate_text': '粤B12345',
            'vehicle_class': 'car',
            'violation_type': '占用应急车道',
            'violation_time': '2026-03-12 14:23',
            'violation_location': '广深高速深圳段K52+300',
            'vehicle_color': '白色',  # 可选
            'vehicle_brand': '特斯拉',  # 可选
            'duration_sec': 12.5,
            'confidence': 0.85,
        }
        evidence_dir: 证据文件目录
        platform: 举报平台 ('shenzhen'|'12123'|'guangzhou')

    Returns:
        HTML 字符串
    """
    steps = PLATFORM_STEPS.get(platform, APP_12123_STEPS)
    data = violation_data

    # 收集证据文件
    evidence_images = []
    evidence_videos = []
    if os.path.isdir(evidence_dir):
        for f in sorted(os.listdir(evidence_dir)):
            fp = os.path.join(evidence_dir, f)
            if f.lower().endswith(('.jpg', '.jpeg', '.png')):
                evidence_images.append(f)
            elif f.lower().endswith(('.mp4', '.mov')):
                evidence_videos.append(f)

    # 需要填写的字段
    fields = [
        ('车牌号码', data.get('plate_text', '未识别'), True),
        ('违法类型', data.get('violation_type', '占用应急车道'), True),
        ('违法时间', data.get('violation_time', ''), True),
        ('违法地点', data.get('violation_location', '请根据GPS或路牌补充'), True),
    ]
    if data.get('vehicle_color'):
        fields.append(('车辆颜色', data['vehicle_color'], True))
    if data.get('vehicle_brand'):
        fields.append(('车辆品牌', data['vehicle_brand'], True))
    fields.append(('车辆类型', _vehicle_class_cn(data.get('vehicle_class', 'car')), True))

    # 构建 HTML
    html = f"""<!DOCTYPE html>
<html lang="zh-CN">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0, user-scalable=no">
<title>举报辅助 - {data.get('plate_text', '未知')}</title>
<style>
* {{ margin: 0; padding: 0; box-sizing: border-box; }}
body {{
    font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
    background: #f5f5f5;
    color: #333;
    padding: 16px;
    padding-bottom: 80px;
    max-width: 500px;
    margin: 0 auto;
}}
.header {{
    background: linear-gradient(135deg, #1a73e8, #0d47a1);
    color: white;
    padding: 20px;
    border-radius: 12px;
    margin-bottom: 16px;
    text-align: center;
}}
.header h1 {{ font-size: 20px; margin-bottom: 4px; }}
.header .plate {{ font-size: 32px; font-weight: bold; letter-spacing: 2px; margin: 8px 0; }}
.header .meta {{ font-size: 13px; opacity: 0.8; }}
.confidence {{
    display: inline-block;
    padding: 2px 10px;
    border-radius: 12px;
    font-size: 12px;
    margin-top: 6px;
}}
.conf-high {{ background: #4caf50; color: white; }}
.conf-mid {{ background: #ff9800; color: white; }}
.conf-low {{ background: #f44336; color: white; }}

.section {{
    background: white;
    border-radius: 12px;
    padding: 16px;
    margin-bottom: 12px;
    box-shadow: 0 1px 3px rgba(0,0,0,0.1);
}}
.section h2 {{
    font-size: 16px;
    margin-bottom: 12px;
    display: flex;
    align-items: center;
    gap: 8px;
}}

/* 步骤 */
.step {{
    display: flex;
    gap: 12px;
    padding: 10px 0;
    border-bottom: 1px solid #eee;
}}
.step:last-child {{ border-bottom: none; }}
.step-icon {{ font-size: 24px; flex-shrink: 0; }}
.step-num {{
    background: #1a73e8;
    color: white;
    width: 22px; height: 22px;
    border-radius: 50%;
    display: flex; align-items: center; justify-content: center;
    font-size: 12px; font-weight: bold;
    flex-shrink: 0;
    margin-top: 2px;
}}
.step-content h3 {{ font-size: 15px; margin-bottom: 2px; }}
.step-content p {{ font-size: 13px; color: #666; }}

/* 复制字段 */
.field {{
    display: flex;
    align-items: center;
    justify-content: space-between;
    padding: 10px 0;
    border-bottom: 1px solid #f0f0f0;
}}
.field:last-child {{ border-bottom: none; }}
.field-label {{ font-size: 13px; color: #888; min-width: 70px; }}
.field-value {{
    font-size: 15px;
    font-weight: 500;
    flex: 1;
    margin: 0 8px;
    word-break: break-all;
}}
.copy-btn {{
    background: #1a73e8;
    color: white;
    border: none;
    padding: 6px 14px;
    border-radius: 6px;
    font-size: 13px;
    cursor: pointer;
    white-space: nowrap;
    flex-shrink: 0;
    -webkit-tap-highlight-color: transparent;
}}
.copy-btn:active {{ background: #0d47a1; }}
.copy-btn.copied {{
    background: #4caf50;
}}

/* 证据 */
.evidence-grid {{
    display: grid;
    grid-template-columns: 1fr 1fr;
    gap: 8px;
    margin-top: 8px;
}}
.evidence-img {{
    width: 100%;
    border-radius: 8px;
    cursor: pointer;
}}
.evidence-img:active {{ opacity: 0.8; }}
.save-hint {{
    text-align: center;
    font-size: 12px;
    color: #999;
    margin-top: 8px;
}}
.video-link {{
    display: block;
    background: #f5f5f5;
    padding: 12px;
    border-radius: 8px;
    text-align: center;
    color: #1a73e8;
    text-decoration: none;
    margin-top: 8px;
    font-size: 14px;
}}
.video-link:active {{ background: #e0e0e0; }}

/* 全屏图片预览 */
.overlay {{
    display: none;
    position: fixed;
    top: 0; left: 0; right: 0; bottom: 0;
    background: rgba(0,0,0,0.9);
    z-index: 100;
    justify-content: center;
    align-items: center;
}}
.overlay.active {{ display: flex; }}
.overlay img {{
    max-width: 95%;
    max-height: 90vh;
    border-radius: 4px;
}}

.footer {{
    text-align: center;
    margin-top: 16px;
    font-size: 12px;
    color: #bbb;
}}
</style>
</head>
<body>

<div class="header">
    <h1>🚨 交通违法举报辅助</h1>
    <div class="plate">{data.get('plate_text', '未识别')}</div>
    <div class="meta">{data.get('violation_type', '占用应急车道')} · {data.get('violation_time', '')}</div>
    {_confidence_badge(data.get('confidence', 0))}
</div>

<div class="section">
    <h2>📝 操作步骤</h2>
    {''.join(_render_step(i+1, s) for i, s in enumerate(steps))}
</div>

<div class="section">
    <h2>📋 填写信息（点击复制）</h2>
    {''.join(_render_field(label, value) for label, value, _ in fields)}
</div>

<div class="section">
    <h2>📸 证据图片（长按保存到相册）</h2>
    <div class="evidence-grid">
        {_render_evidence_images(evidence_images)}
    </div>
    <div class="save-hint">💡 长按图片 → 保存到相册，举报时从相册选择</div>
</div>

{_render_evidence_videos(evidence_videos)}

<div class="section" style="background:#fff3e0;">
    <h2>⚠️ 注意事项</h2>
    <ul style="font-size:13px; color:#666; padding-left:20px; line-height:1.8;">
        <li>请在违法发生后 <b>30天内</b> 提交（部分城市要求当天）</li>
        <li>确保证据图片中 <b>车牌清晰可辨</b></li>
        <li>违法地点如不确定，可填写大致路段</li>
        <li>提交后 <b>1-7个工作日</b> 审核，请耐心等待</li>
        <li>请确认违法行为属实，虚假举报将承担法律责任</li>
    </ul>
</div>

<div class="footer">
    Highway Reporter · 生成时间 {datetime.now().strftime('%Y-%m-%d %H:%M')}
</div>

<!-- 图片预览 -->
<div class="overlay" id="overlay" onclick="this.classList.remove('active')">
    <img id="previewImage" src="">
</div>

<script>
function copyText(btn, text) {{
    if (navigator.clipboard) {{
        navigator.clipboard.writeText(text).then(() => {{
            btn.textContent = '已复制 ✓';
            btn.classList.add('copied');
            setTimeout(() => {{
                btn.textContent = '复制';
                btn.classList.remove('copied');
            }}, 1500);
        }});
    }} else {{
        // fallback
        const ta = document.createElement('textarea');
        ta.value = text;
        ta.style.position = 'fixed';
        ta.style.opacity = '0';
        document.body.appendChild(ta);
        ta.select();
        document.execCommand('copy');
        document.body.removeChild(ta);
        btn.textContent = '已复制 ✓';
        btn.classList.add('copied');
        setTimeout(() => {{
            btn.textContent = '复制';
            btn.classList.remove('copied');
        }}, 1500);
    }}
}}

function previewImg(img) {{
    const overlay = document.getElementById('overlay');
    document.getElementById('previewImage').src = img.src;
    overlay.classList.add('active');
}}
</script>

</body>
</html>"""

    return html


def _render_step(num, step):
    return f"""
    <div class="step">
        <div class="step-num">{num}</div>
        <div class="step-content">
            <h3>{step['icon']} {step['title']}</h3>
            <p>{step['desc']}</p>
        </div>
    </div>"""


def _render_field(label, value):
    escaped = value.replace("'", "\\'").replace('"', '&quot;')
    return f"""
    <div class="field">
        <span class="field-label">{label}</span>
        <span class="field-value">{value}</span>
        <button class="copy-btn" onclick="copyText(this, '{escaped}')">复制</button>
    </div>"""


def _confidence_badge(conf):
    if conf >= 0.8:
        cls = 'conf-high'
        text = f'置信度 {conf:.0%} · 高'
    elif conf >= 0.5:
        cls = 'conf-mid'
        text = f'置信度 {conf:.0%} · 中'
    else:
        cls = 'conf-low'
        text = f'置信度 {conf:.0%} · 低（请人工确认）'
    return f'<span class="confidence {cls}">{text}</span>'


def _vehicle_class_cn(cls):
    mapping = {
        'car': '小型轿车',
        'truck': '货车',
        'bus': '大型客车',
        'motorcycle': '摩托车',
    }
    return mapping.get(cls, '小型轿车')


class ReportServer:
    """
    本地 HTTP 服务器，提供举报辅助页面

    用户电脑开启后，手机扫二维码访问
    同一局域网即可
    """

    def __init__(self, port: int = REPORT_SERVER_PORT):
        self.port = port
        self.server = None
        self.thread = None

    def serve(self, output_dir: str):
        """
        启动 HTTP 服务

        Args:
            output_dir: 包含 index.html 和 evidence/ 的目录
        """
        os.chdir(output_dir)
        handler = http.server.SimpleHTTPRequestHandler
        self.server = http.server.HTTPServer(('0.0.0.0', self.port), handler)

        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)
        self.thread.start()

        ip = get_local_ip()
        url = f"http://{ip}:{self.port}"
        print(f"\n{'='*50}")
        print(f"📱 举报辅助页面已就绪！")
        print(f"   地址: {url}")
        print(f"{'='*50}")

        # 生成二维码
        if HAS_QRCODE:
            qr = qrcode.QRCode(box_size=2, border=1)
            qr.add_data(url)
            qr.make(fit=True)
            qr.print_ascii(invert=True)
            print(f"\n📱 用手机扫描上方二维码打开\n")
        else:
            print(f"   （安装 qrcode 库可显示二维码: pip install qrcode）\n")

        return url

    def stop(self):
        if self.server:
            self.server.shutdown()


def build_report_package(violation_data: dict, evidence_dir: str,
                         output_dir: str, platform: str = '12123'):
    """
    生成完整的举报辅助包

    Args:
        violation_data: 违法信息
        evidence_dir: 原始证据目录
        output_dir: 输出目录（将包含 index.html + evidence/）

    Returns:
        输出目录路径
    """
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)

    # 复制证据文件到 evidence 子目录
    ev_out = output / 'evidence'
    ev_out.mkdir(exist_ok=True)

    import shutil
    if os.path.isdir(evidence_dir):
        for f in os.listdir(evidence_dir):
            src = os.path.join(evidence_dir, f)
            if os.path.isfile(src) and f.lower().endswith(
                    ('.jpg', '.jpeg', '.png', '.mp4', '.mov')):
                shutil.copy2(src, ev_out / f)

    # 生成 HTML
    html = generate_report_html(violation_data, str(ev_out), platform)
    (output / 'index.html').write_text(html, encoding='utf-8')

    # 生成摘要 JSON（给其他模块用）
    summary = {
        **violation_data,
        'evidence_images': [f for f in os.listdir(ev_out)
                           if f.lower().endswith(('.jpg', '.jpeg', '.png'))],
        'evidence_videos': [f for f in os.listdir(ev_out)
                           if f.lower().endswith(('.mp4', '.mov'))],
        'generated_at': datetime.now().isoformat(),
    }
    (output / 'summary.json').write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding='utf-8')

    print(f"[ReportHelper] 举报辅助包已生成: {output}")
    print(f"  - HTML 页面: {output / 'index.html'}")
    print(f"  - 证据文件: {len(os.listdir(ev_out))} 个")

    return str(output)


def write_review_page(output_dir: str, report_data: dict, template_path: Optional[str] = None) -> Path:
    """Write review.html with report JSON embedded so file:// open works in Chrome."""
    from .config import PROJECT_ROOT

    template = Path(template_path) if template_path else PROJECT_ROOT / 'web' / 'review.html'
    html = template.read_text(encoding='utf-8')
    needle = '<script type="application/json" id="report-data"></script>'
    if needle not in html:
        raise ValueError('review.html 缺少 report-data 占位')
    payload = json.dumps(report_data, ensure_ascii=False).replace('<', '\\u003c')
    dest = Path(output_dir) / 'review.html'
    dest.write_text(
        html.replace(needle, f'<script type="application/json" id="report-data">{payload}</script>', 1),
        encoding='utf-8',
    )
    return dest
