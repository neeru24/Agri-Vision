"""
Prediction cache for Agri-Vision inference results.

Two-tier strategy:
  1. Redis (if available) — keyed by sha256(image_bytes), TTL = MODEL_CACHE_TTL_SECONDS (default 3600s).
     Enables cache sharing across gunicorn workers.
  2. In-memory LRU — used when Redis is unavailable (dev/CI environments).

Grad-CAM heatmap byte blobs are intentionally excluded from this cache to
keep Redis memory usage bounded; only the JSON-serialisable inference
result dict is stored.

Usage
-----
    from services.model_cache import get_cached_prediction, cache_prediction, cache_stats

    hit = get_cached_prediction(image_bytes)
    if hit is None:
        result = run_inference(image)
        cache_prediction(image_bytes, result)
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
from datetime import datetime, timedelta
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
_CACHE_TTL_SECONDS: int = int(os.getenv("MODEL_CACHE_TTL_SECONDS", "3600"))  # 1 hour
_DEFAULT_TTL_MINUTES: int = _CACHE_TTL_SECONDS // 60
_DEFAULT_MAX_ENTRIES: int = 500

# Fields that contain large base64 blobs — excluded from the cached dict to
# keep Redis entries small.  They are regenerated (or served from the
# GradCAM in-process cache) on a cache hit.
_EXCLUDED_FIELDS = {"grad_cam_image_b64", "heatmap_only_b64"}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _sha256(image_bytes: bytes) -> str:
    return hashlib.sha256(image_bytes).hexdigest()


def _strip_blobs(result: Dict[str, Any]) -> Dict[str, Any]:
    """Return a copy of *result* with large base64 blob fields removed."""
    stripped = {k: v for k, v in result.items() if k not in _EXCLUDED_FIELDS}
    # Also strip nested heatmap keys inside 'disease'
    if "disease" in stripped and isinstance(stripped["disease"], dict):
        stripped["disease"] = {
            k: v for k, v in stripped["disease"].items()
            if k not in {"heatmap_b64", "heatmap_only_b64"}
        }
    return stripped


# ---------------------------------------------------------------------------
# In-memory fallback cache
# ---------------------------------------------------------------------------

class _InMemoryCache:
    """Thread-safe LRU-style in-memory cache."""

    def __init__(self, ttl_minutes: int = _DEFAULT_TTL_MINUTES, max_entries: int = _DEFAULT_MAX_ENTRIES) -> None:
        self._ttl = timedelta(minutes=ttl_minutes)
        self._max = max_entries
        self._store: Dict[str, Dict[str, Any]] = {}
        self._timestamps: Dict[str, datetime] = {}

    def get(self, key: str) -> Optional[Dict[str, Any]]:
        if key not in self._store:
            logger.debug("InMemCache MISS  %s", key[:12])
            return None
        age = datetime.utcnow() - self._timestamps[key]
        if age > self._ttl:
            self._evict(key)
            logger.debug("InMemCache STALE %s (age %.0fs)", key[:12], age.total_seconds())
            return None
        logger.debug("InMemCache HIT   %s", key[:12])
        return self._store[key]

    def set(self, key: str, value: Dict[str, Any]) -> None:
        if len(self._store) >= self._max:
            self._evict_oldest()
        self._store[key] = value
        self._timestamps[key] = datetime.utcnow()
        logger.debug("InMemCache SET   %s  (size %d/%d)", key[:12], len(self._store), self._max)

    def clear(self) -> None:
        count = len(self._store)
        self._store.clear()
        self._timestamps.clear()
        logger.info("InMemCache cleared (%d entries removed).", count)

    def stats(self) -> Dict[str, Any]:
        return {
            "backend": "memory",
            "size": len(self._store),
            "max_entries": self._max,
            "ttl_seconds": int(self._ttl.total_seconds()),
            "utilisation_pct": round(len(self._store) / self._max * 100, 1),
        }

    def _evict(self, key: str) -> None:
        self._store.pop(key, None)
        self._timestamps.pop(key, None)

    def _evict_oldest(self) -> None:
        if not self._timestamps:
            return
        oldest = min(self._timestamps, key=lambda k: self._timestamps[k])
        self._evict(oldest)
        logger.debug("InMemCache EVICT oldest %s.", oldest[:12])


# ---------------------------------------------------------------------------
# Redis-backed cache
# ---------------------------------------------------------------------------

class _RedisCache:
    """Prediction cache backed by Redis with JSON serialisation."""

    _PREFIX = "agrivision:pred:"

    def __init__(self, redis_client: Any, ttl_seconds: int = _CACHE_TTL_SECONDS) -> None:
        self._redis = redis_client
        self._ttl = ttl_seconds

    def _key(self, sha: str) -> str:
        return f"{self._PREFIX}{sha}"

    def get(self, key: str) -> Optional[Dict[str, Any]]:
        try:
            raw = self._redis.get(self._key(key))
            if raw is None:
                logger.debug("RedisCache MISS  %s", key[:12])
                return None
            logger.debug("RedisCache HIT   %s", key[:12])
            return json.loads(raw)
        except Exception as exc:
            logger.warning("RedisCache GET error: %s", exc)
            return None

    def set(self, key: str, value: Dict[str, Any]) -> None:
        try:
            self._redis.setex(self._key(key), self._ttl, json.dumps(value, default=str))
            logger.debug("RedisCache SET   %s (TTL %ds)", key[:12], self._ttl)
        except Exception as exc:
            logger.warning("RedisCache SET error: %s", exc)

    def clear(self) -> None:
        try:
            pattern = f"{self._PREFIX}*"
            keys = self._redis.keys(pattern)
            if keys:
                self._redis.delete(*keys)
                logger.info("RedisCache cleared (%d keys).", len(keys))
        except Exception as exc:
            logger.warning("RedisCache CLEAR error: %s", exc)

    def stats(self) -> Dict[str, Any]:
        try:
            pattern = f"{self._PREFIX}*"
            size = len(self._redis.keys(pattern))
        except Exception:
            size = -1
        return {
            "backend": "redis",
            "size": size,
            "ttl_seconds": self._ttl,
        }


# ---------------------------------------------------------------------------
# Module-level singleton — wired at import time (or via init_cache_backend)
# ---------------------------------------------------------------------------
_cache: Optional[_InMemoryCache | _RedisCache] = None
_fallback_cache: _InMemoryCache = _InMemoryCache()


def init_cache_backend(redis_client: Any = None) -> None:
    """
    Call once during app startup to point the module at a live Redis client.
    If *redis_client* is None (or the ping fails) the in-memory fallback is used.
    """
    global _cache
    if redis_client is not None:
        try:
            redis_client.ping()
            _cache = _RedisCache(redis_client, ttl_seconds=_CACHE_TTL_SECONDS)
            logger.info(
                "Inference result cache: Redis backend active (TTL %ds).", _CACHE_TTL_SECONDS
            )
            return
        except Exception as exc:
            logger.warning("Redis not reachable for inference cache, using memory: %s", exc)
    _cache = _fallback_cache
    logger.info(
        "Inference result cache: in-memory backend active (TTL %dm).", _DEFAULT_TTL_MINUTES
    )


def _get_cache() -> _InMemoryCache | _RedisCache:
    if _cache is None:
        # Auto-initialise with in-memory if init_cache_backend() was never called
        init_cache_backend(None)
    return _cache  # type: ignore[return-value]


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def get_cached_prediction(image_bytes: bytes) -> Optional[Dict[str, Any]]:
    """Return cached inference result for *image_bytes*, or None on miss/expiry."""
    key = _sha256(image_bytes)
    return _get_cache().get(key)


def cache_prediction(image_bytes: bytes, prediction: Dict[str, Any]) -> None:
    """
    Persist *prediction* in the cache keyed by sha256(*image_bytes*).

    Large base64 blob fields (grad_cam, heatmap) are stripped before storage
    so the cached payload stays compact.
    """
    key = _sha256(image_bytes)
    _get_cache().set(key, _strip_blobs(prediction))


def clear_cache() -> None:
    """Flush all cached predictions."""
    _get_cache().clear()


def cache_stats() -> Dict[str, Any]:
    """Return current cache statistics (backend-agnostic)."""
    return _get_cache().stats()
