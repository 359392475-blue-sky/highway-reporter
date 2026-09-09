#!/usr/bin/env python3
"""Pack a clean zip for friend testers. Does not include venv, .env, or videos."""
import sys
import zipfile
from datetime import date
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
INCLUDE = [
    'full_pipeline.py',
    'server.py',
    'README.md',
    '测试说明.md',
    'requirements.txt',
    '.env.example',
    'src',
    'web',
    'data/README.md',
    'LICENSE',
]
SKIP_DIR_NAMES = {'__pycache__', 'venv', '.git'}
SKIP_SUFFIXES = {'.pyc', '.pyo'}


def should_skip(path: Path) -> bool:
    if any(part in SKIP_DIR_NAMES for part in path.parts):
        return True
    if path.suffix in SKIP_SUFFIXES:
        return True
    if path.name == '.env':
        return True
    return False


def add_path(zf: zipfile.ZipFile, path: Path) -> None:
    if should_skip(path):
        return
    if path.is_file():
        zf.write(path, path.relative_to(ROOT).as_posix())
        return
    if path.is_dir():
        for child in sorted(path.rglob('*')):
            if child.is_file() and not should_skip(child):
                zf.write(child, child.relative_to(ROOT).as_posix())


def main() -> int:
    out_name = f'highway-reporter-test-{date.today().isoformat()}.zip'
    out_path = ROOT / out_name
    missing = [name for name in INCLUDE if not (ROOT / name).exists()]
    if missing:
        print('缺少文件，无法打包: ' + ', '.join(missing))
        return 1

    with zipfile.ZipFile(out_path, 'w', zipfile.ZIP_DEFLATED) as zf:
        for name in INCLUDE:
            add_path(zf, ROOT / name)

    size_mb = out_path.stat().st_size / 1024 / 1024
    print(f'已生成 {out_path.name}（{size_mb:.1f} MB）')
    print('发给朋友时请附上 测试说明.md 里的说明；不要把 .env 或 API Key 打进压缩包。')
    return 0


if __name__ == '__main__':
    sys.exit(main())
