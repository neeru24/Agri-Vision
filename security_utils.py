"""Security helpers for upload validation and secret handling."""

from __future__ import annotations

import os
import re
import tempfile
import unicodedata
from typing import Mapping, Tuple

try:
    import magic  # type: ignore
except ImportError:  # pragma: no cover - handled by UploadValidationError
    magic = None


class UploadValidationError(ValueError):
    """Raised when an uploaded file fails validation."""

    def __init__(self, message: str, status_code: int = 400) -> None:
        super().__init__(message)
        self.status_code = status_code


def is_production_env(env: Mapping[str, str]) -> bool:
    """Detect production mode from common environment variables."""
    return env.get("FLASK_ENV") == "production" or env.get("ENV") == "production" or env.get("APP_ENV") == "production"


def resolve_secret_key(env: Mapping[str, str]) -> str:
    """Resolve SECRET_KEY and fail fast when missing in production."""
    secret_key = env.get("SECRET_KEY")

    # Tests expect SystemExit on import-time for production.
    # This function raises RuntimeError; app.py catches it and aborts via SystemExit.
    if secret_key is None or secret_key == "":
        if is_production_env(env):
            # app.py converts this RuntimeError to SystemExit.
            raise RuntimeError("SECRET_KEY must be configured in production.")

        # In non-production, provide a deterministic dev key.
        return "dev_secret_123"


    return secret_key



def sanitize_filename(filename: str, max_length: int = 120) -> str:
    """Return a safe ASCII-only filename with a bounded length."""
    normalized = unicodedata.normalize("NFKD", filename or "")
    ascii_name = normalized.encode("ascii", "ignore").decode("ascii")
    ascii_name = ascii_name.strip().replace(" ", "_")
    ascii_name = re.sub(r"[^A-Za-z0-9._-]", "", ascii_name)

    if not ascii_name or ascii_name in {".", ".."}:
        ascii_name = "upload"

    base, ext = os.path.splitext(ascii_name)
    base = base[:max_length].rstrip("._-") or "upload"
    ext = ext[:10].lower()
    return f"{base}{ext}"


def _read_stream_limited(stream, max_bytes: int, chunk_size: int = 1024 * 1024) -> bytes:
    chunks = []
    total = 0
    while True:
        chunk = stream.read(chunk_size)
        if not chunk:
            break
        total += len(chunk)
        if total > max_bytes:
            raise UploadValidationError("File exceeds maximum upload size.", status_code=413)
        chunks.append(chunk)
    return b"".join(chunks)


def detect_mime_type(sample: bytes) -> str:
    """Detect MIME type.

    Uses python-magic (libmagic) when available; on failure or if missing,
    falls back to signature-based detection from ``services.file_validator``
    (e.g. CI/dev without libmagic, or Windows).
    """
    if magic is not None:
        try:
            return magic.from_buffer(sample, mime=True)
        except Exception:
            pass
    from services.file_validator import detect_image_type

    detected = detect_image_type(sample)
    if detected is None:
        raise UploadValidationError("Invalid image content.", status_code=400)
    return detected[1]


def validate_image_upload(
    file_storage,
    allowed_extensions: set[str],
    allowed_mime_types: set[str],
    max_bytes: int,
) -> Tuple[str, bytes, str]:
    """Validate extension, size, and magic bytes for an upload."""
    if file_storage is None or not getattr(file_storage, "filename", ""):
        raise UploadValidationError("No file selected.", status_code=400)

    sanitized_name = sanitize_filename(file_storage.filename)
    if "." not in sanitized_name:
        raise UploadValidationError("Invalid file name.", status_code=400)

    ext = sanitized_name.rsplit(".", 1)[1].lower()
    if ext not in allowed_extensions:
        raise UploadValidationError("Invalid file type. Please upload PNG, JPG, JPEG, or GIF images.", status_code=400)

    try:
        file_storage.stream.seek(0)
    except Exception:
        pass

    file_bytes = _read_stream_limited(file_storage.stream, max_bytes)

    try:
        file_storage.stream.seek(0)
    except Exception:
        pass

    # Prefer signature-based validation from services/file_validator when available.
    try:
        from services.file_validator import validate_upload as validate_upload_magicbytes

        claimed_mime = getattr(file_storage, "content_type", "") or ""
        ok, reason = validate_upload_magicbytes(
            file_bytes=file_bytes,
            claimed_mime=claimed_mime,
        )
        if not ok:
            raise UploadValidationError(reason, status_code=400)
        detected = detect_mime_type(file_bytes[:2048])
        if detected not in allowed_mime_types:
            raise UploadValidationError("Invalid image content.", status_code=400)
        return sanitized_name, file_bytes, detected
    except UploadValidationError:
        raise
    except Exception:
        # Fallback to detect_mime_type only.
        mime_type = detect_mime_type(file_bytes[:2048])
        if mime_type not in allowed_mime_types:
            raise UploadValidationError("Invalid image content.", status_code=400)
        return sanitized_name, file_bytes, mime_type



def save_temp_upload(file_bytes: bytes, upload_dir: str, filename: str) -> str:
    """Persist upload to a temp file so it can be explicitly cleaned up."""
    os.makedirs(upload_dir, exist_ok=True)
    ext = os.path.splitext(filename)[1].lower()
    fd, path = tempfile.mkstemp(prefix="upload_", suffix=ext, dir=upload_dir)
    with os.fdopen(fd, "wb") as handle:
        handle.write(file_bytes)
    return path


def cleanup_temp_upload(path: str | None) -> None:
    if not path:
        return
    try:
        os.remove(path)
    except FileNotFoundError:
        return
