"""Live pipeline defaults. CLI flags may override these at runtime."""
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEV_CHECKOUT = Path.home() / 'projects' / 'highway-reporter'


def default_output_dir() -> Path:
    """Keep test artifacts inside the project, not Desktop / Movies."""
    if (PROJECT_ROOT / 'data' / 'test-videos').is_dir():
        return PROJECT_ROOT / 'output'
    if (DEV_CHECKOUT / 'full_pipeline.py').is_file():
        return DEV_CHECKOUT / 'output'
    return Path.home() / 'Movies' / 'HighwayReporter'

# L1 detection
YOLO_MODEL = str(PROJECT_ROOT / 'data' / 'yolov8n.pt')
YOLO_CONFIDENCE = 0.45
L1_FRAME_STRIDE = 2

# Emergency-lane heuristic (normalized coordinates)
EMERGENCY_X_THRESHOLD = 0.68
ROI_Y_RANGE = (0.30, 0.90)
MIN_AREA_RATIO = 0.003
MAX_AREA_RATIO = 0.45
IOU_THRESHOLD = 0.25
MAX_LOST_FRAMES = 15
MIN_VIOLATION_FRAMES = 15
EVIDENCE_INTERVAL = 10

# L1.5 frame picking
L15_FRAME_STRIDE = 3
L15_RIGHT_X_THRESHOLD = 0.55
L15_MIN_GAP_SEC = 0.3
FRAMES_PER_VIOLATION = 4
JPEG_QUALITY = 95

# L0 highway filter
HIGHWAY_SCORE_YES = 0.35
HIGHWAY_SCORE_MAYBE = 0.2
VIDEO_EXTENSIONS = ('.mp4', '.mov', '.avi', '.ts', '.mkv')

# L2 cloud reader
CLOUD_MODEL = 'doubao-seed-2-0-pro-260215'
CLOUD_FALLBACK_MODEL = 'doubao-1.5-vision-pro-32k-250115'
CLOUD_BASE_URL = 'https://ark.cn-beijing.volces.com/api/v3'
API_KEY_PATH = '~/.openclaw/openclaw.json'
API_KEY_ENV = 'VOLCENGINE_API_KEY'
PLATE_MIN_CONFIDENCE = 0.5
PLATE_MIN_VOTES = 2
MONTHLY_QUOTA = 10

# Local servers
WEB_PORT = 8199
REPORT_SERVER_PORT = 8122
