"""Environment checks and API key loading for friend testers."""
import os
import shutil
import subprocess
from pathlib import Path
from typing import List, Optional

from .config import API_KEY_ENV, API_KEY_PATH, PROJECT_ROOT, YOLO_MODEL

APP_SUPPORT_DIR = Path.home() / 'Library' / 'Application Support' / 'HighwayReporter'
APP_SUPPORT_ENV = APP_SUPPORT_DIR / '.env'


def _load_env_file(env_path: Path) -> None:
    if not env_path.is_file():
        return
    try:
        text = env_path.read_text(encoding='utf-8')
    except OSError:
        return
    for raw in text.splitlines():
        line = raw.strip()
        if not line or line.startswith('#') or '=' not in line:
            continue
        key, _, value = line.partition('=')
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        if key and key not in os.environ:
            os.environ[key] = value


def load_dotenv(path: Optional[Path] = None) -> None:
    """Load KEY=VALUE pairs from .env without overriding existing env vars."""
    if path is not None:
        _load_env_file(Path(path))
        return
    _load_env_file(APP_SUPPORT_ENV)
    _load_env_file(PROJECT_ROOT / '.env')


def _key_from_openclaw() -> str:
    """Read a string key or OpenClaw secret-file reference."""
    import json

    config_path = Path(os.path.expanduser(API_KEY_PATH))
    if not config_path.is_file():
        return ''
    try:
        config = json.loads(config_path.read_text(encoding='utf-8'))
        raw = config.get('models', {}).get('providers', {}).get(
            'volcengine', {}).get('apiKey', '')
    except (OSError, ValueError):
        return ''

    if isinstance(raw, str) and raw.strip():
        return raw.strip()
    if not isinstance(raw, dict):
        return ''

    ident = str(raw.get('id') or 'volcengine').rstrip('/').split('/')[-1] or 'volcengine'
    secrets_path = Path.home() / '.openclaw' / 'secrets.json'
    if not secrets_path.is_file():
        return ''
    try:
        secrets = json.loads(secrets_path.read_text(encoding='utf-8'))
    except (OSError, ValueError):
        return ''
    value = secrets.get('models', {}).get(ident) or secrets.get(ident) or ''
    return value.strip() if isinstance(value, str) else ''


def save_api_key(key: str) -> Path:
    """Persist the key for the Mac app (not inside the .app bundle)."""
    APP_SUPPORT_DIR.mkdir(parents=True, exist_ok=True)
    APP_SUPPORT_ENV.write_text(f'{API_KEY_ENV}={key.strip()}\n', encoding='utf-8')
    os.chmod(APP_SUPPORT_ENV, 0o600)
    return APP_SUPPORT_ENV


def load_api_key() -> str:
    """Resolve the Volcengine key: env var, then .env, then OpenClaw config."""
    load_dotenv()
    key = (os.environ.get(API_KEY_ENV) or '').strip()
    if key:
        return key

    key = _key_from_openclaw()
    if key:
        return key

    raise RuntimeError(
        '未找到火山引擎 API Key。请任选一种方式配置：\n'
        f'  1. 在应用里填写 Key，或复制 .env.example 为 .env，填入 {API_KEY_ENV}=你的key\n'
        f'  2. macOS/Linux: export {API_KEY_ENV}=你的key\n'
        f'  3. Windows PowerShell: $env:{API_KEY_ENV}="你的key"'
    )


def _binary_candidates(name: str) -> List[Path]:
    """PATH, then the Mac .app Resources/bin, then Homebrew."""
    found = []
    which = shutil.which(name)
    if which:
        found.append(Path(which))
    found.extend([
        PROJECT_ROOT.parent / 'bin' / name,  # HighwayReporter.app/.../Resources/app -> bin
        PROJECT_ROOT / 'bin' / name,
        Path('/opt/homebrew/bin') / name,
        Path('/usr/local/bin') / name,
    ])
    unique = []
    seen = set()
    for path in found:
        key = str(path)
        if key not in seen:
            seen.add(key)
            unique.append(path)
    return unique


def _binary_runs(path: Path) -> bool:
    if not path.is_file() or not os.access(path, os.X_OK):
        return False
    try:
        proc = subprocess.run(
            [str(path), '-version'],
            capture_output=True,
            timeout=8,
        )
    except (OSError, subprocess.TimeoutExpired):
        return False
    return proc.returncode == 0


def find_binary(name: str) -> Optional[str]:
    """Return a working ffmpeg/ffprobe path. Skip binaries killed by invalid signatures."""
    for path in _binary_candidates(name):
        if _binary_runs(path):
            return str(path)
    return None


def find_ffmpeg() -> Optional[str]:
    return find_binary('ffmpeg')


def find_ffprobe() -> Optional[str]:
    return find_binary('ffprobe')


def _missing_binaries() -> List[str]:
    missing = []
    for name in ('ffmpeg', 'ffprobe'):
        if find_binary(name) is None:
            missing.append(name)
    return missing


def run_preflight(
    dry_run: bool = False,
    check_cloud: Optional[bool] = None,
    check_model: bool = True,
    model_path: Optional[str] = None,
) -> List[str]:
    """Return human-readable error strings. Empty list means ready to run."""
    errors: List[str] = []
    if check_cloud is None:
        check_cloud = not dry_run

    missing = _missing_binaries()
    if missing:
        errors.append(
            '未找到系统工具: ' + '、'.join(missing) + '。'
            '安装包应已内置 ffmpeg；若你是源码运行，请 brew install ffmpeg。'
        )

    if check_model:
        path = Path(model_path or YOLO_MODEL)
        if not path.is_file():
            errors.append(f'未找到 YOLO 模型文件: {path}')

    if check_cloud:
        try:
            load_api_key()
        except RuntimeError as exc:
            errors.append(str(exc))

    return errors
