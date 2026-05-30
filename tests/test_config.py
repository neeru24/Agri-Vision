import os
import subprocess
import sys
import pytest


def test_missing_secret_key_aborts_import():
    """Importing `app` in production without SECRET_KEY must abort startup (SystemExit)."""
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            (
                "import os; "
                "os.environ.pop('SECRET_KEY', None); "
                "os.environ.pop('AGRI_VISION_ALLOW_DEV_SECRET', None); "
                "os.environ['FLASK_ENV'] = 'production'; "
                "import importlib; "
                "importlib.import_module('app')"
            ),
        ],
        capture_output=True,
        text=True,
        cwd=os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    )
    assert result.returncode != 0, "Expected app to exit with non-zero code"
    assert "Missing required SECRET_KEY" in result.stderr or "Missing required SECRET_KEY" in result.stdout
