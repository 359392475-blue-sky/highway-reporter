#!/usr/bin/env python3
import unittest
from unittest.mock import patch

from src.report_platforms import (
    city_by_id,
    copy_all,
    fields_guangzhou,
    fields_hunan,
    fields_shenzhen,
    match_region,
    provinces,
    reward_regions,
)


# All vehicle identifiers and dates in these fixtures are synthetic placeholders.

class PlatformFieldTests(unittest.TestCase):
    def setUp(self):
        self.v = {
            'id': 32,
            'can_report': True,
            'plate': '粤A00000',
            'time': '2000/01/01 00:00:00',
            'location': None,
            'color': '灰色',
            'vehicle_type': '小型轿车',
            'duration_sec': 0.5,
        }

    def test_shenzhen_uses_miniprogram_labels(self):
        labels = [row[0] for row in fields_shenzhen(self.v)]
        self.assertEqual(labels[:4], ['违法行为', '号牌号码', '违法时间', '违法地点'])
        self.assertIn('粤A00000', dict(fields_shenzhen(self.v))['号牌号码'])

    def test_guangzhou_offline_upload_labels(self):
        labels = [row[0] for row in fields_guangzhou(self.v)]
        self.assertIn('路段类型', labels)
        self.assertIn('事发路段名称', labels)
        self.assertIn('高（快）速公路上占用应急车道行驶的', dict(fields_guangzhou(self.v))['违法行为'])

    def test_copy_all_is_pasteable(self):
        text = copy_all(fields_shenzhen(self.v))
        self.assertIn('号牌号码：粤A00000', text)

    def test_city_lookup(self):
        self.assertEqual(city_by_id('zhuhai')['name'], '珠海')
        self.assertEqual(city_by_id('guangdong-zhuhai')['id'], 'guangdong-zhuhai')

    def test_catalog_covers_mainland_provinces(self):
        self.assertGreaterEqual(len(provinces()), 31)
        self.assertGreaterEqual(len(reward_regions()), 8)

    def test_match_ip_place_names(self):
        self.assertEqual(match_region('广东省', '深圳市')['id'], 'guangdong-shenzhen')
        self.assertEqual(match_region('湖南', '长沙')['id'], 'hunan-expressway')
        self.assertEqual(match_region('四川', '成都')['id'], 'sichuan-expressway')
        self.assertEqual(match_region('北京市', '北京')['id'], 'beijing')

    def test_hunan_expressway_fields(self):
        rows = dict(fields_hunan(self.v))
        self.assertEqual(rows['违法类型'], '占用应急车道')
        self.assertIn('12 小时', rows['注意'])
        shenzhen = city_by_id('guangdong-shenzhen')
        self.assertTrue(shenzhen['reward'])
        self.assertEqual(shenzhen['fields'](self.v)[1][1], '粤A00000')

    def test_locate_from_ip_pconline(self):
        from src.report_platforms import locate_from_ip
        payload = {'pro': '广东省', 'city': '深圳市'}
        with patch('src.report_platforms._http_json', return_value=payload):
            region, status = locate_from_ip()
        self.assertEqual(region['id'], 'guangdong-shenzhen')
        self.assertIn('深圳', status)


if __name__ == '__main__':
    unittest.main()
