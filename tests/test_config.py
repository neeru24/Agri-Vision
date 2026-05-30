import importlib
import os
import sys
import pytest


def test_missing_secret_key_aborts_import(monkeypatch):
    """Importing `app` in production without SECRET_KEY must abort startup (SystemExit)."""
    monkeypatch.delenv("SECRET_KEY", raising=False)
    monkeypatch.delenv("AGRI_VISION_ALLOW_DEV_SECRET", raising=False)
    monkeypatch.setenv("FLASK_ENV", "production")

    if "app" in sys.modules:
        del sys.modules["app"]

    with pytest.raises(SystemExit):
        importlib.import_module("app")
