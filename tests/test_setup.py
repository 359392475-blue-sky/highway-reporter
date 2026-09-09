#!/usr/bin/env python3
"""Setup/preflight tests for friend-facing install."""
import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.preflight import load_api_key, load_dotenv, run_preflight


def _no_support_env(tmp):
    return patch('src.preflight.APP_SUPPORT_ENV', Path(tmp) / 'no-app-support.env')


class ApiKeyTests(unittest.TestCase):
    def test_env_var_is_used(self):
        with patch.dict(os.environ, {'VOLCENGINE_API_KEY': 'from-env'}, clear=False):
            self.assertEqual(load_api_key(), 'from-env')

    def test_dotenv_is_used_when_env_missing(self):
        with tempfile.TemporaryDirectory() as tmp:
            env_file = Path(tmp) / '.env'
            env_file.write_text('VOLCENGINE_API_KEY=from-dotenv\n', encoding='utf-8')
            env = os.environ.copy()
            env.pop('VOLCENGINE_API_KEY', None)
            with patch.dict(os.environ, env, clear=True):
                with patch('src.preflight.PROJECT_ROOT', Path(tmp)):
                    with patch('src.preflight.API_KEY_PATH', '/no/such/openclaw.json'):
                        with _no_support_env(tmp):
                            load_dotenv(env_file)
                            self.assertEqual(load_api_key(), 'from-dotenv')

    def test_missing_key_explains_env_var(self):
        env = os.environ.copy()
        env.pop('VOLCENGINE_API_KEY', None)
        with tempfile.TemporaryDirectory() as tmp:
            with patch.dict(os.environ, env, clear=True):
                with patch('src.preflight.PROJECT_ROOT', Path(tmp)):
                    with patch('src.preflight.API_KEY_PATH', '/no/such/openclaw.json'):
                        with _no_support_env(tmp):
                            with self.assertRaises(RuntimeError) as ctx:
                                load_api_key()
        msg = str(ctx.exception)
        self.assertIn('VOLCENGINE_API_KEY', msg)
        self.assertIn('.env', msg)


class PreflightTests(unittest.TestCase):
    def test_dry_run_does_not_require_api_key(self):
        env = os.environ.copy()
        env.pop('VOLCENGINE_API_KEY', None)
        with tempfile.TemporaryDirectory() as tmp:
            model = Path(tmp) / 'yolov8n.pt'
            model.write_bytes(b'fake')
            with patch.dict(os.environ, env, clear=True):
                with patch('src.preflight.PROJECT_ROOT', Path(tmp)):
                    with patch('src.preflight.YOLO_MODEL', str(model)):
                        with patch('src.preflight.API_KEY_PATH', '/no/such/openclaw.json'):
                            with _no_support_env(tmp):
                                with patch('src.preflight.find_binary',
                                           return_value='/usr/bin/ffmpeg'):
                                    errors = run_preflight(dry_run=True)
        self.assertEqual(errors, [])

    def test_full_run_requires_api_key(self):
        env = os.environ.copy()
        env.pop('VOLCENGINE_API_KEY', None)
        with tempfile.TemporaryDirectory() as tmp:
            model = Path(tmp) / 'yolov8n.pt'
            model.write_bytes(b'fake')
            with patch.dict(os.environ, env, clear=True):
                with patch('src.preflight.PROJECT_ROOT', Path(tmp)):
                    with patch('src.preflight.YOLO_MODEL', str(model)):
                        with patch('src.preflight.API_KEY_PATH', '/no/such/openclaw.json'):
                            with _no_support_env(tmp):
                                with patch('src.preflight.find_binary',
                                           return_value='/usr/bin/ffmpeg'):
                                    errors = run_preflight(dry_run=False)
        self.assertTrue(any('VOLCENGINE_API_KEY' in e for e in errors))

    def test_missing_ffmpeg_is_reported(self):
        with patch('src.preflight.find_binary', return_value=None):
            errors = run_preflight(dry_run=True, check_cloud=False, check_model=False)
        self.assertTrue(any('ffmpeg' in e.lower() for e in errors))

    def test_missing_model_is_reported(self):
        with patch('src.preflight.find_binary', return_value='/usr/bin/ffmpeg'):
            with patch('src.preflight.YOLO_MODEL', '/no/such/yolov8n.pt'):
                errors = run_preflight(dry_run=True, check_cloud=False)
        self.assertTrue(any('yolov8n' in e.lower() or '模型' in e for e in errors))


if __name__ == '__main__':
    unittest.main()
