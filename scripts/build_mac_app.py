#!/usr/bin/env python3
"""Build a self-contained Apple Silicon .app + .dmg for a brand-new Mac."""
import os
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DIST = ROOT / 'dist'
APP_NAME = 'HighwayReporter'
APP = DIST / f'{APP_NAME}.app'
CONTENTS = APP / 'Contents'
RESOURCES = CONTENTS / 'Resources'
PYTHON_SRC = Path.home() / '.local/share/uv/python/cpython-3.12-macos-aarch64-none'
INCLUDE_APP = [
    'full_pipeline.py',
    'mac_app.py',
    'server.py',
    'README.md',
    '测试说明.md',
    '.env.example',
    'src',
    'web',
    'data/yolov8n.pt',
    'data/report_catalog.json',
]


def run(cmd, **kw):
    print('+', ' '.join(map(str, cmd)))
    subprocess.check_call(cmd, **kw)


def make_venv_portable(venv_dir: Path, python_dir: Path) -> None:
    """Replace absolute python symlinks with paths relative to the .app."""
    bin_dir = venv_dir / 'bin'
    target = os.path.relpath(python_dir / 'bin' / 'python3.12', bin_dir)
    for name in ('python', 'python3', 'python3.12'):
        link = bin_dir / name
        if link.exists() or link.is_symlink():
            link.unlink()
        link.symlink_to(target)
    cfg = venv_dir / 'pyvenv.cfg'
    home_rel = os.path.relpath(python_dir / 'bin', venv_dir)
    lines = []
    for line in cfg.read_text(encoding='utf-8').splitlines():
        if line.startswith('home'):
            lines.append(f'home = {home_rel}')
        else:
            lines.append(line)
    cfg.write_text('\n'.join(lines) + '\n', encoding='utf-8')


def copy_app_files(dest: Path) -> None:
    dest.mkdir(parents=True, exist_ok=True)
    for name in INCLUDE_APP:
        src = ROOT / name
        target = dest / name
        if src.is_dir():
            if target.exists():
                shutil.rmtree(target)
            shutil.copytree(
                src, target,
                ignore=shutil.ignore_patterns('__pycache__', '*.pyc', 'test-videos'),
            )
        else:
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src, target)


def write_plist() -> None:
    (CONTENTS / 'Info.plist').write_text(f'''<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
  <key>CFBundleName</key><string>高速举报助手</string>
  <key>CFBundleDisplayName</key><string>高速举报助手</string>
  <key>CFBundleIdentifier</key><string>com.highwayreporter.app</string>
  <key>CFBundleVersion</key><string>1.0.0</string>
  <key>CFBundleShortVersionString</key><string>1.0</string>
  <key>CFBundleExecutable</key><string>{APP_NAME}</string>
  <key>CFBundlePackageType</key><string>APPL</string>
  <key>LSMinimumSystemVersion</key><string>13.0</string>
  <key>LSArchitecturePriority</key>
  <array>
    <string>arm64</string>
  </array>
  <key>NSHighResolutionCapable</key><true/>
</dict>
</plist>
''', encoding='utf-8')


def write_launcher() -> None:
    macos = CONTENTS / 'MacOS'
    macos.mkdir(parents=True, exist_ok=True)
    launcher = macos / APP_NAME
    c_src = ROOT / 'scripts' / 'mac_launcher.c'
    run([
        'clang', '-arch', 'arm64', '-O2',
        '-mmacosx-version-min=13.0',
        '-o', str(launcher), str(c_src),
    ])
    os.chmod(launcher, 0o755)


def write_dmg_readme(dmg_dir: Path) -> None:
    (dmg_dir / '使用说明.txt').write_text(
        '高速违法举报助手（Apple 芯片 Mac）\n'
        '\n'
        '1. 把「HighwayReporter」拖到「应用程序」文件夹\n'
        '2. 第一次打开：按住 Control 点图标 → 打开（绕过未签名提示）\n'
        '3. 选择记录仪视频，可先勾选「只做本地粗筛」\n'
        '4. 完整读车牌需要火山引擎 API Key，且账号已充值\n'
        '\n'
        '不需要另外安装 Python、ffmpeg 或 YOLO。\n'
        '仅支持 Apple 芯片（M1/M2/M3/M4）。\n',
        encoding='utf-8',
    )


def main() -> int:
    if not PYTHON_SRC.is_dir():
        print('缺少 uv 管理的 Python 3.12，正在安装…')
        run(['uv', 'python', 'install', '3.12'])
        if not PYTHON_SRC.is_dir():
            print('Python 3.12 安装失败', PYTHON_SRC)
            return 1

    if APP.exists():
        shutil.rmtree(APP)
    DIST.mkdir(exist_ok=True)
    RESOURCES.mkdir(parents=True)

    print('复制 Python 运行时…')
    py_dst = RESOURCES / 'python'
    shutil.copytree(PYTHON_SRC, py_dst, symlinks=True)

    py = py_dst / 'bin' / 'python3'
    venv_dir = RESOURCES / 'venv'
    print('创建可迁移 venv 并安装依赖（含 YOLO / torch，大约 1–2GB）…')
    run(['uv', 'venv', '--python', str(py), '--relocatable', '--link-mode', 'copy', str(venv_dir)])
    run([
        'uv', 'pip', 'install',
        '--python', str(venv_dir / 'bin' / 'python'),
        '--link-mode', 'copy',
        '-r', str(ROOT / 'requirements.txt'),
        'certifi',
    ])
    make_venv_portable(venv_dir, py_dst)

    print('复制应用文件…')
    copy_app_files(RESOURCES / 'app')

    print('打包 ffmpeg…')
    run([sys.executable, str(ROOT / 'scripts' / 'bundle_ffmpeg.py'), str(RESOURCES)])

    write_plist()
    write_launcher()

    print('ad-hoc 签名…')
    # 先签 ffmpeg/dylib，再签整个 .app。--deep 对 Resources/ 下二进制不可靠。
    for nested in [RESOURCES / 'bin' / 'ffmpeg', RESOURCES / 'bin' / 'ffprobe']:
        if nested.is_file():
            run(['codesign', '--force', '--sign', '-', str(nested)])
    lib_dir = RESOURCES / 'lib'
    if lib_dir.is_dir():
        for dylib in sorted(lib_dir.glob('*.dylib')):
            run(['codesign', '--force', '--sign', '-', str(dylib)])
    run(['codesign', '--force', '--deep', '--sign', '-', str(APP)])

    dmg_stage = DIST / 'dmg-stage'
    if dmg_stage.exists():
        shutil.rmtree(dmg_stage)
    dmg_stage.mkdir()
    shutil.copytree(APP, dmg_stage / f'{APP_NAME}.app', symlinks=True)
    write_dmg_readme(dmg_stage)
    try:
        os.symlink('/Applications', dmg_stage / 'Applications')
    except OSError:
        pass

    dmg = DIST / f'{APP_NAME}-mac-arm64.dmg'
    if dmg.exists():
        dmg.unlink()
    print('生成 DMG…')
    run([
        'hdiutil', 'create',
        '-volname', '高速举报助手',
        '-srcfolder', str(dmg_stage),
        '-ov', '-format', 'UDZO',
        str(dmg),
    ])

    size = dmg.stat().st_size / 1024 / 1024
    print(f'\n完成: {dmg}  ({size:.0f} MB)')
    print(f'App:  {APP}')
    return 0


if __name__ == '__main__':
    sys.exit(main())
