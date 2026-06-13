"""
File type validation using magic bytes to prevent DoS via malicious uploads.

Validates uploaded files by reading their binary signatures rather than
trusting the caller-supplied MIME type, blocking executables, archives,
and other non-image content from reaching the model inference pipeline.
"""

import logging
from typing import Optional, Tuple

logger = logging.getLogger(__name__)

# Magic-byte signatures mapped to (format_name, canonical_mime_type).
# Longer signatures are checked first to avoid false matches on short prefixes.
_MAGIC_BYTES: list[tuple[bytes, str, str]] = [
    (b"\x89PNG\r\n\x1a\n", "PNG",  "image/png"),
    (b"GIF89a",            "GIF",  "image/gif"),
    (b"GIF87a",            "GIF",  "image/gif"),
    (b"II\x2a\x00",        "TIFF", "image/tiff"),   # little-endian
    (b"MM\x00\x2a",        "TIFF", "image/tiff"),   # big-endian
    (b"\xFF\xD8\xFF",      "JPEG", "image/jpeg"),
    (b"BM",                "BMP",  "image/bmp"),
]

ALLOWED_MIME_TYPES: frozenset[str] = frozenset(
    mime for _, _, mime in _MAGIC_BYTES
)

# Reject files larger than this to prevent memory exhaustion.
MAX_FILE_SIZE_BYTES: int = 10 * 1024 * 1024  # 10 MB


def detect_image_type(file_bytes: bytes) -> Optional[Tuple[str, str]]:
    """Return (format_name, mime_type) detected from magic bytes, or None."""
    if not file_bytes:
        return None
    header = file_bytes[:12]
    for magic, fmt, mime in _MAGIC_BYTES:
        if header.startswith(magic):
            return fmt, mime
    return None


def validate_upload(
    file_bytes: bytes,
    claimed_mime: str,
) -> Tuple[bool, str]:
    """Validate an uploaded file before passing it to the model.

    Checks (in order):
      1. File is non-empty and within the size limit.
      2. Claimed MIME type is on the allowlist.
      3. Actual magic bytes match an allowed image format.
      4. Detected MIME type matches the claimed MIME type.

    Args:
        file_bytes:   Raw bytes of the uploaded file.
        claimed_mime: Content-Type header value supplied by the client.

    Returns:
        (True, "") on success.
        (False, human-readable reason) on failure.
    """
    if not file_bytes:
        return False, "Uploaded file is empty."

    size_mb = len(file_bytes) / (1024 * 1024)
    if len(file_bytes) > MAX_FILE_SIZE_BYTES:
        return False, f"File size ({size_mb:.1f} MB) exceeds the {MAX_FILE_SIZE_BYTES // (1024 * 1024)} MB limit."

    if claimed_mime not in ALLOWED_MIME_TYPES:
        return False, (
            f"Content-Type '{claimed_mime}' is not supported. "
            f"Accepted types: {', '.join(sorted(ALLOWED_MIME_TYPES))}."
        )

    detected = detect_image_type(file_bytes)
    if detected is None:
        logger.warning("Unrecognised file signature for claimed type '%s'.", claimed_mime)
        return False, (
            "Invalid image: file content does not match any supported image format."
        )

    detected_fmt, detected_mime = detected
    if detected_mime != claimed_mime:
        logger.warning(
            "MIME mismatch: claimed '%s', detected '%s' (%s).",
            claimed_mime, detected_mime, detected_fmt,
        )
        return False, (
            f"File content appears to be {detected_fmt} ({detected_mime}), "
            f"not the declared '{claimed_mime}'."
        )

    logger.debug("File validation passed: %s (%s), %.2f KB.", detected_fmt, detected_mime, len(file_bytes) / 1024)
    return True, ""


def get_supported_formats() -> dict[str, str]:
    """Return a mapping of format name to MIME type for all accepted formats."""
    seen: dict[str, str] = {}
    for _, fmt, mime in _MAGIC_BYTES:
        seen.setdefault(fmt, mime)
    return seen
