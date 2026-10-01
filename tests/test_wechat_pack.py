#!/usr/bin/env python3
import tempfile
import unittest
from pathlib import Path

from PIL import Image

from src.report_platforms import city_by_id
from src.wechat_pack import collect_images, write_wechat_pack


# All vehicle identifiers and dates in these fixtures are synthetic placeholders.

class WechatPackTests(unittest.TestCase):
    def setUp(self):
        self.v = {
            'id': 32,
            'can_report': True,
            'plate': '粤A00000',
            'time': '2000/01/01 00:00:00',
            'location': '广深沿江高速',
            'color': '灰色',
            'vehicle_type': '小型轿车',
            'duration_sec': 0.5,
            'evidence_files': ['frame_1.jpg'],
        }

    def _jpeg(self, path: Path):
        Image.new('RGB', (40, 30), (80, 80, 80)).save(path, format='JPEG')

    def test_collects_named_evidence_frames(self):
        with tempfile.TemporaryDirectory() as tmp:
            folder = Path(tmp) / 'violation_32'
            folder.mkdir()
            self._jpeg(folder / 'frame_1.jpg')
            self._jpeg(folder / 'scene_1_frame2.jpg')
            paths = collect_images(Path(tmp), self.v)
            names = [p.name for p in paths]
            self.assertEqual(names[0], 'frame_1.jpg')
            self.assertIn('scene_1_frame2.jpg', names)

    def test_html_is_one_file_with_fields_and_inline_image(self):
        region = city_by_id('guangdong-shenzhen')
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            evid = root / 'violation_32'
            evid.mkdir()
            self._jpeg(evid / 'frame_1.jpg')
            pack = write_wechat_pack(root, [self.v], region)
            html_path = pack.html_files[0]
            text = html_path.read_text(encoding='utf-8')
            self.assertEqual(html_path.name, 'report-32.html')
            self.assertIn('粤A00000', text)
            self.assertIn('data:image/jpeg;base64,', text)
            self.assertIn('复制这一栏', text)
            self.assertIn('随手e拍', text)
            self.assertTrue(pack.guide.is_file())
            self.assertIn('文件传输助手', pack.guide.read_text(encoding='utf-8'))
            self.assertIn('号牌号码：粤A00000', pack.txt_files[0].read_text(encoding='utf-8'))


if __name__ == '__main__':
    unittest.main()
