#!/usr/bin/env python3
"""
cloud_ocr_test.py - 测试火山引擎视觉大模型读取车牌和时间戳
"""
import json
import base64
import sys
import os
import urllib.request
import ssl

# 清除代理
for k in ['ALL_PROXY','HTTPS_PROXY','HTTP_PROXY','all_proxy','https_proxy','http_proxy']:
    os.environ.pop(k, None)

# 读取 API key
config_path = os.path.expanduser('~/.openclaw/openclaw.json')
with open(config_path) as f:
    config = json.load(f)

api_key = config.get('models', {}).get('providers', {}).get('volcengine', {}).get('apiKey', '')
if not api_key:
    print("ERROR: volcengine apiKey not found in openclaw.json")
    sys.exit(1)

def ask_vision(image_path, question):
    """调用火山引擎 doubao-vision 模型"""
    with open(image_path, 'rb') as f:
        img_data = base64.b64encode(f.read()).decode()
    
    ext = image_path.lower().split('.')[-1]
    mime = 'image/jpeg' if ext in ('jpg', 'jpeg') else 'image/png'
    
    body = json.dumps({
        'model': 'doubao-seed-1-6-vision-250815',
        'messages': [{
            'role': 'user',
            'content': [
                {'type': 'text', 'text': question},
                {'type': 'image_url', 'image_url': {'url': f'data:{mime};base64,{img_data}'}}
            ]
        }],
        'max_tokens': 2000,
    }).encode()
    
    ctx = ssl.create_default_context()
    req = urllib.request.Request(
        'https://ark.cn-beijing.volces.com/api/v3/chat/completions',
        data=body,
        headers={
            'Authorization': f'Bearer {api_key}',
            'Content-Type': 'application/json',
        },
        method='POST',
    )
    
    resp = urllib.request.urlopen(req, context=ctx, timeout=60)
    result = json.loads(resp.read())
    return result['choices'][0]['message']['content']


if __name__ == '__main__':
    evidence_dir = 'data/test-videos/report_test'
    
    # 测试所有证据帧
    test_images = [
        'evidence_v22_1_t4.7s.jpg',
        'evidence_v22_2_t8.1s.jpg', 
        'evidence_v36_1_t11.1s.jpg',
        'evidence_v36_2_t14.6s.jpg',
    ]
    
    prompt = """这是中国高速公路行车记录仪截图。请精确提取以下信息：

1. **车牌号码**：列出画面中所有可见的车牌（包括模糊的，尽量猜测每个字符）
2. **时间戳**：记录仪水印中的日期、时间、车速
3. **记录仪品牌**：左下角或右下角的品牌名
4. **应急车道车辆**：最右侧车道（靠护栏）的车辆颜色和类型
5. **路标/地点信息**：任何可见的路牌、指示牌文字

请逐项回答，车牌号尽量精确到每个字符，不确定的字符用?标注。"""

    for img_name in test_images:
        img_path = os.path.join(evidence_dir, img_name)
        if not os.path.exists(img_path):
            print(f"[SKIP] {img_name} not found")
            continue
        
        print(f"\n{'='*60}")
        print(f"📷 {img_name}")
        print('='*60)
        
        try:
            result = ask_vision(img_path, prompt)
            print(result)
        except Exception as e:
            print(f"ERROR: {e}")
        
        print()
