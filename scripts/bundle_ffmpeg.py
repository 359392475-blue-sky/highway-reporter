#!/usr/bin/env python3
"""Copy ffmpeg/ffprobe and Homebrew dylibs into an app-relative bin/lib layout."""
import os
import shutil
import subprocess
from pathlib import Path


SYSTEM_PREFIXES = (
    '/usr/lib/',
    '/System/Library/',
    '/Library/Apple/',
)


def _otool_libs(path: Path):
    out = subprocess.check_output(['otool', '-L', str(path)], text=True)
    libs = []
    for line in out.splitlines()[1:]:
        line = line.strip()
        if not line:
            continue
        lib = line.split(' (', 1)[0].strip()
        libs.append(lib)
    return libs


def _is_system(lib: str) -> bool:
    return lib.startswith(SYSTEM_PREFIXES) or lib.startswith('@')


def collect(binary: Path, lib_dir: Path, seen: set):
    for lib in _otool_libs(binary):
        if _is_system(lib) or lib == str(binary):
            continue
        src = Path(lib)
        if not src.is_file():
            continue
        dest = lib_dir / src.name
        if src.resolve() not in seen:
            seen.add(src.resolve())
            shutil.copy2(src, dest)
            collect(dest, lib_dir, seen)


def retarget(binary: Path, lib_dir: Path):
    for lib in _otool_libs(binary):
        if _is_system(lib):
            continue
        name = Path(lib).name
        if not (lib_dir / name).exists():
            continue
        new = f'@loader_path/../lib/{name}'
        if Path(binary).parent.name == 'lib':
            new = f'@loader_path/{name}'
        subprocess.check_call(
            ['install_name_tool', '-change', lib, new, str(binary)],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
    subprocess.check_call(
        ['install_name_tool', '-id', f'@loader_path/{binary.name}', str(binary)],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )


def bundle(dest_root: Path, ffmpeg='ffmpeg', ffprobe='ffprobe') -> None:
    bin_dir = dest_root / 'bin'
    lib_dir = dest_root / 'lib'
    bin_dir.mkdir(parents=True, exist_ok=True)
    lib_dir.mkdir(parents=True, exist_ok=True)

    ffmpeg_src = shutil.which(ffmpeg)
    ffprobe_src = shutil.which(ffprobe)
    if not ffmpeg_src or not ffprobe_src:
        raise SystemExit('本机找不到 ffmpeg/ffprobe，无法打进安装包')

    for src in (ffmpeg_src, ffprobe_src):
        dest = bin_dir / Path(src).name
        shutil.copy2(src, dest)
        os.chmod(dest, 0o755)

    seen = set()
    collect(bin_dir / 'ffmpeg', lib_dir, seen)
    collect(bin_dir / 'ffprobe', lib_dir, seen)

    for item in list(lib_dir.iterdir()) + [bin_dir / 'ffmpeg', bin_dir / 'ffprobe']:
        retarget(item, lib_dir)
        # install_name_tool 会破坏原签名；必须逐个重签，
        # codesign --deep 不会可靠地签 Resources/ 下的二进制。
        subprocess.check_call(
            ['codesign', '--force', '--sign', '-', str(item)],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )

    print(f'bundled ffmpeg -> {bin_dir} ({len(list(lib_dir.iterdir()))} dylibs)')


if __name__ == '__main__':
    import sys
    bundle(Path(sys.argv[1] if len(sys.argv) > 1 else 'dist/ffmpeg-bundle'))
