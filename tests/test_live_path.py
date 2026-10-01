#!/usr/bin/env python3
"""Live-path tests for L2/L3 and local violation rules."""
import json
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.cloud_reader import evaluate_can_report
from src.evidence import EvidenceBuilder, EvidencePackage
from src.highway_filter import classify_highway_score
from src.report_helper import build_report_package
from src.violation_tracker import ViolationEvent, ViolationTracker


# All vehicle identifiers and dates in these fixtures are synthetic placeholders.

class ReportPackageTests(unittest.TestCase):
    def test_html_references_copied_evidence_files(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            evidence_dir = tmp / 'violation_1'
            evidence_dir.mkdir()
            (evidence_dir / 'scene_1_frame10.jpg').write_bytes(b'\xff\xd8fake')
            (evidence_dir / 'clip.mp4').write_bytes(b'fake-mp4')

            output_dir = tmp / 'report_1'
            build_report_package(
                {
                    'plate_text': '粤A00000',
                    'violation_type': '占用应急车道',
                    'violation_time': '2026-03-15 10:00:00',
                },
                str(evidence_dir),
                str(output_dir),
                platform='shenzhen',
            )

            html = (output_dir / 'index.html').read_text(encoding='utf-8')
            self.assertIn('evidence/scene_1_frame10.jpg', html)
            self.assertIn('evidence/clip.mp4', html)
            self.assertTrue((output_dir / 'evidence' / 'scene_1_frame10.jpg').exists())


class ReviewPageTests(unittest.TestCase):
    def test_embedded_report_is_readable_without_fetch(self):
        from src.report_helper import write_review_page
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            data = {
                'video': 'demo.mp4',
                'video_info': {'width': 1920, 'height': 1080, 'duration': 12},
                'pipeline': {'cloud_model': 'test', 'l2_api_calls': 1},
                'violations': [{'id': 1, 'can_report': True, 'plate': '粤A00000'}],
            }
            path = write_review_page(str(tmp), data)
            html = path.read_text(encoding='utf-8')
            self.assertIn('粤A00000', html)
            start = html.index('id="report-data">') + len('id="report-data">')
            end = html.index('</script>', start)
            parsed = json.loads(html[start:end].replace('\\u003c', '<'))
            self.assertEqual(parsed['violations'][0]['plate'], '粤A00000')


class EvidenceQualityTests(unittest.TestCase):
    def _event(self, plate_text=None, plate_confidence=0.0, duration_frames=60):
        return ViolationEvent(
            track_id=32,
            plate_text=plate_text,
            plate_confidence=plate_confidence,
            vehicle_class='car',
            start_frame=0,
            end_frame=duration_frames,
            duration_frames=duration_frames,
            evidence_frame_indices=[],
            confidence=0.8,
        )

    def test_cloud_plate_raises_quality_when_l1_plate_is_empty(self):
        violation = self._event(plate_text=None, plate_confidence=0.0)
        pkg = EvidencePackage(
            violation_id=32,
            plate_text='粤A00000',
            vehicle_class='car',
            plate_confidence=0.95,
        )
        pkg.scene_screenshots = ['a.jpg', 'b.jpg']
        pkg.video_clip = 'clip.mp4'

        score = EvidenceBuilder()._assess_quality(pkg, violation)
        self.assertGreaterEqual(score, 1.0)

    def test_l1_empty_plate_does_not_get_full_plate_credit(self):
        violation = self._event(plate_text=None, plate_confidence=0.0)
        pkg = EvidencePackage(
            violation_id=32,
            plate_text=None,
            vehicle_class='car',
        )
        pkg.scene_screenshots = ['a.jpg', 'b.jpg']
        pkg.video_clip = 'clip.mp4'

        score = EvidenceBuilder()._assess_quality(pkg, violation)
        self.assertLess(score, 0.7)


class CanReportTests(unittest.TestCase):
    def test_valid_plate_can_report(self):
        self.assertTrue(evaluate_can_report('粤A00000', 0.75))

    def test_missing_plate_cannot_report(self):
        self.assertFalse(evaluate_can_report(None, 1.0))

    def test_uncertain_char_cannot_report(self):
        self.assertFalse(evaluate_can_report('粤A0000?', 0.9))

    def test_low_confidence_cannot_report(self):
        self.assertFalse(evaluate_can_report('粤A00000', 0.49))


class HighwayScoreTests(unittest.TestCase):
    def test_high_score_is_highway(self):
        self.assertIs(classify_highway_score(0.35), True)
        self.assertIs(classify_highway_score(0.9), True)

    def test_mid_score_is_uncertain(self):
        self.assertIsNone(classify_highway_score(0.2))
        self.assertIsNone(classify_highway_score(0.34))

    def test_low_score_is_not_highway(self):
        self.assertIs(classify_highway_score(0.19), False)


class TrackerRuleTests(unittest.TestCase):
    def _det(self, cx_norm, cy_norm=0.55, area=40000, w=1280, h=720):
        x1n, x2n = cx_norm - 0.08, cx_norm + 0.08
        return {
            'bbox': [int(x1n * w), int(cy_norm * h) - 100,
                     int(x2n * w), int(cy_norm * h) + 100],
            'bbox_norm': [x1n, cy_norm - 0.14, x2n, cy_norm + 0.14],
            'class': 'car',
            'confidence': 0.9,
            'center': (cx_norm * w, cy_norm * h),
            'area': area,
        }

    def test_right_side_vehicle_confirms_after_enough_frames(self):
        tracker = ViolationTracker(
            emergency_x_threshold=0.68,
            min_violation_frames=5,
        )
        frame_shape = (720, 1280, 3)
        found = []
        for i in range(12):
            det = self._det(0.78)
            det['bbox'] = [900 + i, 350, 1100 + i, 550]
            found.extend(tracker.update([det], frame_shape))
        self.assertEqual(len(found), 1)

    def test_left_side_vehicle_is_not_a_violation(self):
        tracker = ViolationTracker(
            emergency_x_threshold=0.68,
            min_violation_frames=5,
        )
        frame_shape = (720, 1280, 3)
        found = []
        for i in range(20):
            det = self._det(0.40)
            det['bbox'] = [400 + i, 350, 600 + i, 550]
            found.extend(tracker.update([det], frame_shape))
        self.assertEqual(found, [])


class FfmpegClipTests(unittest.TestCase):
    def test_extract_clip_from_sample_video(self):
        from src.preflight import find_ffmpeg
        if not find_ffmpeg():
            self.skipTest('no working ffmpeg')
        video = ROOT / 'data' / 'test-videos' / 'violation_short_39s.mp4'
        if not video.is_file():
            self.skipTest('sample video missing')
        with tempfile.TemporaryDirectory() as tmp:
            clip = Path(tmp) / 'clip.mp4'
            result = EvidenceBuilder().extract_video_clip(
                str(video), 2.0, 5.0, str(clip),
            )
            self.assertEqual(result, str(clip))
            self.assertGreater(clip.stat().st_size, 1000)


if __name__ == '__main__':
    unittest.main()
