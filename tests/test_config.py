"""Tests for Config — loading, platform detection, auto-detection."""
from __future__ import annotations

import json
import os
from pathlib import Path
from unittest.mock import patch

import pytest

from src.config import Config


class TestDetectPlatforms:
    def test_telegram_enabled_when_token_set(self):
        with patch.dict(os.environ, {"TELEGRAM_BOT_TOKEN": "123:abc"}):
            platforms = Config.detect_platforms()
            assert platforms["telegram"] is True

    def test_telegram_disabled_when_no_token(self):
        with patch.dict(os.environ, {}, clear=True):
            platforms = Config.detect_platforms()
            assert platforms["telegram"] is False

    def test_telegram_disabled_when_token_empty(self):
        with patch.dict(os.environ, {"TELEGRAM_BOT_TOKEN": ""}):
            platforms = Config.detect_platforms()
            assert platforms["telegram"] is False


class TestLogPlatformStatus:
    def test_logs_enabled_and_disabled(self, capsys):
        platforms = {"telegram": True, "api": False}
        Config.log_platform_status(platforms)
        captured = capsys.readouterr()
        assert "telegram" in captured.out
        assert "ENABLED" in captured.out
        assert "api" in captured.out
        assert "disabled" in captured.out

    def test_logs_no_interfaces(self, capsys):
        platforms = {"telegram": False}
        Config.log_platform_status(platforms)
        captured = capsys.readouterr()
        assert "No interface enabled" in captured.out


class TestLoad:
    def test_loads_defaults_when_no_config(self, tmp_path):
        with patch.object(Path, "exists", return_value=False), \
             patch.object(Path, "read_text", return_value="{}"):
            cfg = Config.load()
            assert cfg.bot_name == "rk-agent"
            assert cfg.access_mode == "allowlist"

    def test_loads_from_config_json(self, tmp_path):
        config_file = tmp_path / "config.json"
        config_file.write_text(json.dumps({"bot_name": "test-bot", "access_mode": "public"}))
        with patch.object(Config, "load", wraps=Config.load) as mock_load:
            cfg = Config.load(str(config_file))
            assert cfg.bot_name == "test-bot"
            assert cfg.access_mode == "public"

    def test_auto_detects_nvidia_when_key_present(self, tmp_path):
        with patch.object(Path, "exists", return_value=False), \
             patch.object(Path, "read_text", return_value="{}"), \
             patch.dict(os.environ, {"NVIDIA_API_KEY": "nvkey123"}):
            cfg = Config.load()
            assert cfg.nvidia_enabled is True
            assert "nvidia" in cfg.default_provider_priority

    def test_vercel_auto_detect(self, tmp_path):
        with patch.object(Path, "exists", return_value=False), \
             patch.object(Path, "read_text", return_value="{}"), \
             patch.dict(os.environ, {"VERCEL": "1", "VERCEL_API_KEY": "vkey"}):
            cfg = Config.load()
            assert cfg.vercel_enabled is True
            assert "vercel" in cfg.default_provider_priority

    def test_loads_soul_md_identity(self, tmp_path):
        config_file = tmp_path / "config.json"
        config_file.write_text("{}")
        # Write soul.md in the actual CWD (where Config.load() looks for it)
        import os
        orig_cwd = os.getcwd()
        os.chdir(tmp_path)
        try:
            (tmp_path / "soul.md").write_text("Custom identity")
            cfg = Config.load(str(config_file))
            assert "Custom identity" in cfg.system_prompt
        finally:
            os.chdir(orig_cwd)


class TestGetAndReload:
    def test_get_returns_cached(self):
        Config.invalidate()
        cfg1 = Config.get()
        cfg2 = Config.get()
        assert cfg1 is cfg2

    def test_reload_returns_new_instance(self):
        Config.invalidate()
        cfg1 = Config.get()
        cfg2 = Config.reload()
        assert cfg1 is not cfg2

    def test_reload_updates_cache(self):
        Config.invalidate()
        Config.get()
        cfg2 = Config.reload()
        assert Config.get() is cfg2
