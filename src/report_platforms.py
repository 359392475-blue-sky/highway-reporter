"""Nationwide 占用应急车道 report channels and copy-ready field rows."""
from __future__ import annotations

import json
import urllib.error
import urllib.request
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from .config import PROJECT_ROOT
from .preflight import APP_SUPPORT_DIR

Field = Tuple[str, str]  # (label, value)

DEFAULT_REGION_ID = 'guangdong-shenzhen'
PREFS_PATH = APP_SUPPORT_DIR / 'prefs.json'
CATALOG_PATH = PROJECT_ROOT / 'data' / 'report_catalog.json'

LEGACY_IDS = {
    'shenzhen': 'guangdong-shenzhen',
    'guangzhou': 'guangdong-guangzhou',
    'zhuhai': 'guangdong-zhuhai',
    'foshan': 'guangdong-foshan',
    'dongguan': 'guangdong-dongguan',
    'zhongshan': 'guangdong-zhongshan',
    'huizhou': 'guangdong-huizhou',
    'jiangmen': 'guangdong-jiangmen',
}

_STRIP = (
    '维吾尔自治区', '壮族自治区', '回族自治区', '特别行政区',
    '自治区', '省', '市',
)


def _text(v: dict, *keys, default='') -> str:
    for key in keys:
        val = v.get(key)
        if val not in (None, ''):
            return str(val)
    return default


def describe(v: dict) -> str:
    color = _text(v, 'color', 'vehicle_color')
    vtype = _text(v, 'vehicle_type', default='小型轿车')
    dur = v.get('duration_sec')
    dur_s = f'，持续约{dur}秒' if dur else ''
    plate = _text(v, 'plate', 'plate_text', default='号牌未识别')
    return f'{color}{vtype}（{plate}）在高速公路应急车道行驶{dur_s}。'.replace('（号牌未识别）', '')


def _core(v: dict) -> Dict[str, str]:
    return {
        'type': _text(v, 'violation_type', default='占用应急车道'),
        'gz_type': '高（快）速公路上占用应急车道行驶的',
        'dg_type': '机动车非紧急情况时占用高速公路应急车道',
        'plate': _text(v, 'plate', 'plate_text', default='未识别'),
        'time': _text(v, 'time', 'violation_time', default='需补充'),
        'location': _text(v, 'location', 'violation_location', default='请根据路牌或导航补充'),
        'color': _text(v, 'color', 'vehicle_color', default=''),
        'vtype': _text(v, 'vehicle_type', default='小型轿车'),
        'road_kind': '高速公路',
        'desc': describe(v),
    }


def fields_shenzhen(v: dict) -> List[Field]:
    c = _core(v)
    rows = [
        ('违法行为', c['type']),
        ('号牌号码', c['plate']),
        ('违法时间', c['time']),
        ('违法地点', c['location']),
        ('补充说明', c['desc']),
    ]
    if c['color']:
        rows.insert(4, ('车辆颜色', c['color']))
    return rows


def fields_guangzhou(v: dict) -> List[Field]:
    c = _core(v)
    return [
        ('路段类型', c['road_kind']),
        ('事发路段名称', c['location']),
        ('违法行为', c['gz_type']),
        ('号牌号码', c['plate']),
        ('违法时间', c['time']),
        ('车辆信息', f"{c['color']}{c['vtype']}".strip() or c['vtype']),
    ]


def fields_zhuhai(v: dict) -> List[Field]:
    c = _core(v)
    blob = (
        f"时间：{c['time']}\n"
        f"地点：{c['location']}\n"
        f"车牌：{c['plate']}\n"
        f"违法行为：占用应急车道\n"
        f"说明：{c['desc']}"
    )
    return [
        ('私信/邮件全文', blob),
        ('时间', c['time']),
        ('地点', c['location']),
        ('车牌', c['plate']),
        ('违法行为', '占用应急车道'),
    ]


def fields_dongguan(v: dict) -> List[Field]:
    c = _core(v)
    return [
        ('举报类型', c['dg_type']),
        ('号牌号码', c['plate']),
        ('违法时间', c['time']),
        ('违法地点', c['location']),
        ('补充说明', c['desc']),
    ]


def fields_generic_wechat(v: dict) -> List[Field]:
    c = _core(v)
    return [
        ('违法行为', c['type']),
        ('号牌号码', c['plate']),
        ('违法时间', c['time']),
        ('违法地点', c['location']),
        ('补充说明', c['desc']),
    ]


def fields_message(v: dict) -> List[Field]:
    return fields_zhuhai(v)


def fields_beijing(v: dict) -> List[Field]:
    c = _core(v)
    return [
        ('违法类型', c['type']),
        ('号牌号码', c['plate']),
        ('违法时间', c['time']),
        ('违法地点', c['location']),
        ('证据要求', '5–20 秒视频，或 2 张能看出位移的照片'),
        ('补充说明', c['desc']),
    ]


def fields_shanghai(v: dict) -> List[Field]:
    c = _core(v)
    return [
        ('入口', '随申办 → 上海交警专区 → 违法视频举报'),
        ('违法类型', c['type']),
        ('号牌号码', c['plate']),
        ('违法时间', c['time']),
        ('违法地点', c['location']),
        ('补充说明', c['desc']),
    ]


def fields_hunan(v: dict) -> List[Field]:
    c = _core(v)
    return [
        ('违法类型', '占用应急车道'),
        ('号牌号码', c['plate']),
        ('违法时间', c['time']),
        ('地点/方向', c['location']),
        ('注意', '打开定位；提交与违法间隔通常不超过 12 小时；不要在驾驶中拍摄'),
        ('补充说明', c['desc']),
    ]


def fields_sichuan(v: dict) -> List[Field]:
    c = _core(v)
    return [
        ('违法类型', '占用应急车道'),
        ('号牌号码', c['plate']),
        ('违法时间', c['time']),
        ('违法地点', c['location']),
        ('注意', '针对高速公路占用应急车道等七类违法；月奖励上限以公告为准'),
        ('补充说明', c['desc']),
    ]


def fields_tianjin(v: dict) -> List[Field]:
    c = _core(v)
    return [
        ('违法类型', c['type']),
        ('号牌号码', c['plate']),
        ('违法时间', c['time']),
        ('违法地点', c['location']),
        ('补充说明', c['desc']),
    ]


def fields_xian(v: dict) -> List[Field]:
    c = _core(v)
    return [
        ('违法类型', c['type']),
        ('号牌号码', c['plate']),
        ('违法时间', c['time']),
        ('违法地点', c['location'] + '（请在小程序地图定位）'),
        ('车辆信息', f"{c['color']}{c['vtype']}".strip() or c['vtype']),
        ('补充说明', c['desc']),
    ]


def fields_chongqing(v: dict) -> List[Field]:
    c = _core(v)
    return [
        ('违法类型', '占用应急车道'),
        ('号牌号码', c['plate']),
        ('违法时间', c['time']),
        ('违法地点', c['location']),
        ('补充说明', c['desc']),
    ]


def fields_phone122(v: dict) -> List[Field]:
    c = _core(v)
    blob = (
        f"您好，我举报高速公路占用应急车道：\n"
        f"时间：{c['time']}\n"
        f"地点：{c['location']}\n"
        f"车牌：{c['plate']}\n"
        f"说明：{c['desc']}"
    )
    return [
        ('拨打', '122'),
        ('口述稿', blob),
        ('时间', c['time']),
        ('地点', c['location']),
        ('车牌', c['plate']),
    ]


FORM_BUILDERS = {
    'shenzhen': fields_shenzhen,
    'guangzhou': fields_guangzhou,
    'message': fields_message,
    'dongguan': fields_dongguan,
    'generic': fields_generic_wechat,
    'beijing': fields_beijing,
    'shanghai': fields_shanghai,
    'hunan_expressway': fields_hunan,
    'sichuan_expressway': fields_sichuan,
    'tianjin': fields_tianjin,
    'xian': fields_xian,
    'chongqing': fields_chongqing,
    'phone122': fields_phone122,
}


def _load_catalog() -> dict:
    if not CATALOG_PATH.is_file():
        return {'regions': []}
    return json.loads(CATALOG_PATH.read_text(encoding='utf-8'))


def _hydrate(raw: dict) -> dict:
    region = dict(raw)
    form = region.get('form') or 'generic'
    region['name'] = region.get('city') or region.get('id')
    region['title'] = f"{region.get('province', '')} · {region['name']}".strip(' ·')
    region['fields'] = FORM_BUILDERS.get(form, fields_generic_wechat)
    region['aliases'] = list(region.get('aliases') or [])
    region['reward'] = bool(region.get('reward'))
    region['reward_note'] = region.get('reward_note') or ''
    return region


_CATALOG = _load_catalog()
REGIONS: List[dict] = [_hydrate(r) for r in _CATALOG.get('regions') or []]
CITIES = REGIONS  # backward-compatible alias


def _norm(name: str) -> str:
    text = (name or '').strip()
    for suffix in _STRIP:
        if text.endswith(suffix) and len(text) > len(suffix):
            text = text[: -len(suffix)]
            break
    return text


def city_by_id(city_id: str) -> dict:
    wanted = LEGACY_IDS.get(city_id, city_id)
    for region in REGIONS:
        if region['id'] == wanted:
            return region
    needle = _norm(city_id)
    for region in REGIONS:
        names = [_norm(region['city']), _norm(region['id'])]
        names.extend(_norm(a) for a in region['aliases'])
        if needle in names:
            return region
    return city_by_id(DEFAULT_REGION_ID) if wanted != DEFAULT_REGION_ID else REGIONS[0]


def provinces(reward_only: bool = False) -> List[str]:
    seen = []
    for region in REGIONS:
        if reward_only and not region['reward']:
            continue
        name = region['province']
        if name not in seen:
            seen.append(name)
    return seen


def cities_in_province(province: str, reward_only: bool = False) -> List[dict]:
    rows = [r for r in REGIONS if r['province'] == province]
    if reward_only:
        rewarded = [r for r in rows if r['reward']]
        if rewarded:
            return rewarded
    return rows


def reward_regions() -> List[dict]:
    return [r for r in REGIONS if r['reward']]


def _is_expressway(region: dict) -> bool:
    return '高速' in (region.get('city') or '') or 'expressway' in region['id']


def match_region(province: str, city: str = '', *, highway: bool = True) -> dict:
    """Map IP/user place names onto a catalog entry. Highway reports prefer 省管高速."""
    p = _norm(province)
    c = _norm(city)
    in_province = [
        r for r in REGIONS
        if _norm(r['province']) == p or p in _norm(r['province']) or _norm(r['province']) in p
    ]
    if not in_province:
        for r in REGIONS:
            aliases = [_norm(a) for a in r['aliases']]
            if p in aliases or c in aliases:
                in_province.append(r)
    if not in_province:
        return city_by_id(DEFAULT_REGION_ID)

    city_hits = []
    if c:
        for r in in_province:
            names = [_norm(r['city'])] + [_norm(a) for a in r['aliases']]
            if c in names and r['city'] not in ('全省', '全区', '其他地市'):
                city_hits.append(r)

    if highway:
        express = [r for r in in_province if _is_expressway(r)]
        if express:
            return express[0]
    if city_hits:
        return city_hits[0]
    rewarded = [r for r in in_province if r['reward']]
    if rewarded:
        return rewarded[0]
    return in_province[0]


def load_preferred_region_id() -> Optional[str]:
    if not PREFS_PATH.is_file():
        return None
    try:
        data = json.loads(PREFS_PATH.read_text(encoding='utf-8'))
    except (OSError, ValueError):
        return None
    region_id = data.get('region_id')
    if not region_id:
        return None
    found = city_by_id(region_id)
    return found['id'] if found else None


def save_preferred_region_id(region_id: str) -> Path:
    APP_SUPPORT_DIR.mkdir(parents=True, exist_ok=True)
    payload = {'region_id': region_id}
    if PREFS_PATH.is_file():
        try:
            payload = json.loads(PREFS_PATH.read_text(encoding='utf-8'))
            payload['region_id'] = region_id
        except (OSError, ValueError):
            payload = {'region_id': region_id}
    PREFS_PATH.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding='utf-8')
    return PREFS_PATH


def _http_json(url: str, encoding_hints: Tuple[str, ...] = ('utf-8',), timeout: float = 4.0) -> dict:
    req = urllib.request.Request(url, headers={'User-Agent': 'HighwayReporter/1.0'})
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        raw = resp.read()
    last_error = None
    for enc in encoding_hints:
        try:
            return json.loads(raw.decode(enc))
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            last_error = exc
    raise ValueError(last_error or 'invalid json')


def locate_from_ip(timeout: float = 4.0) -> Tuple[Optional[dict], str]:
    """Return (region, status text). Uses public IP geolocation, not GPS."""
    try:
        data = _http_json(
            'https://whois.pconline.com.cn/ipJson.jsp?json=true',
            encoding_hints=('utf-8', 'gbk', 'gb2312'),
            timeout=timeout,
        )
        province = data.get('pro') or ''
        city = data.get('city') or data.get('region') or ''
        if province or city:
            region = match_region(province, city)
            return region, f"网络定位：{province}{city}"
    except (urllib.error.URLError, TimeoutError, ValueError, OSError):
        pass
    try:
        data = _http_json(
            'http://ip-api.com/json/?lang=zh-CN',
            encoding_hints=('utf-8',),
            timeout=timeout,
        )
        if data.get('status') == 'success' and data.get('country') in ('中国', 'China'):
            province = data.get('regionName') or ''
            city = data.get('city') or ''
            region = match_region(province, city)
            return region, f"网络定位：{province}{city}"
        if data.get('status') == 'success':
            return None, f"当前 IP 不在国内（{data.get('country')}），请手动选地点"
    except (urllib.error.URLError, TimeoutError, ValueError, OSError) as exc:
        return None, f'定位失败：{exc}'
    return None, '定位失败，请手动选择地点'


def copy_all(rows: List[Field]) -> str:
    return '\n'.join(f'{label}：{value}' for label, value in rows)
