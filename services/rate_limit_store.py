from __future__ import annotations

import threading
import time
from dataclasses import dataclass
from typing import Dict, Optional, Tuple

try:
    import redis  # type: ignore
except Exception:  # pragma: no cover
    redis = None  # type: ignore


@dataclass(frozen=True)
class RateLimitResult:
    allowed: bool
    limit: int
    remaining: int
    reset_after_seconds: int


class RateLimitStoreError(Exception):
    pass


class RateLimitStore:
    def incr_with_expiry(
        self,
        *,
        key: str,
        window_seconds: int,
        limit: int,
        increment: int = 1,
    ) -> RateLimitResult:
        raise NotImplementedError

    def get_ttl_seconds(self, *, key: str) -> Optional[int]:
        """Return remaining TTL for key in seconds.

        If key does not exist, return None.
        """
        raise NotImplementedError


class MemoryRateLimitStore(RateLimitStore):
    def __init__(self) -> None:
        self._lock = threading.Lock()
        # key -> (count, expires_at_epoch)
        self._counters: Dict[str, Tuple[int, float]] = {}

    def incr_with_expiry(
        self,
        *,
        key: str,
        window_seconds: int,
        limit: int,
        increment: int = 1,
    ) -> RateLimitResult:
        now = time.time()
        with self._lock:
            current = self._counters.get(key)
            if current is None:
                expires_at = now + float(window_seconds)
                count = increment
            else:
                count, expires_at = current
                if now >= expires_at:
                    expires_at = now + float(window_seconds)
                    count = increment
                else:
                    count = count + increment

            self._counters[key] = (count, expires_at)
            remaining = max(0, limit - count)
            reset_after = max(0, int(expires_at - now))
            allowed = count <= limit
            return RateLimitResult(
                allowed=allowed,
                limit=limit,
                remaining=remaining,
                reset_after_seconds=reset_after,
            )

    def get_ttl_seconds(self, *, key: str) -> Optional[int]:
        now = time.time()
        with self._lock:
            current = self._counters.get(key)
            if current is None:
                return None
            _, expires_at = current
            ttl = int(expires_at - now)
            return ttl if ttl > 0 else None


class RedisRateLimitStore(RateLimitStore):
    def __init__(self, redis_url: str) -> None:
        if redis is None:
            raise RateLimitStoreError("redis package not available")
        self._client = redis.Redis.from_url(redis_url, decode_responses=True)
        self._client.ping()

        # Lua for atomic incr + expiry + TTL
        self._lua_incr = self._client.register_script(
            """
            local key = KEYS[1]
            local window = tonumber(ARGV[1])
            local increment = tonumber(ARGV[2])
            local limit = tonumber(ARGV[3])

            local current = redis.call('INCRBY', key, increment)
            if current == increment then
              redis.call('EXPIRE', key, window)
            end

            local ttl = redis.call('TTL', key)
            if ttl < 0 then
              ttl = window
              redis.call('EXPIRE', key, window)
            end

            local remaining = limit - current
            if remaining < 0 then remaining = 0 end

            if current <= limit then
              return {1, limit, remaining, ttl}
            else
              return {0, limit, remaining, ttl}
            end
            """
        )

    def incr_with_expiry(
        self,
        *,
        key: str,
        window_seconds: int,
        limit: int,
        increment: int = 1,
    ) -> RateLimitResult:
        try:
            res = self._lua_incr(
                keys=[key],
                args=[window_seconds, increment, limit],
            )
            allowed = bool(int(res[0]))
            return RateLimitResult(
                allowed=allowed,
                limit=int(res[1]),
                remaining=int(res[2]),
                reset_after_seconds=int(res[3]),
            )
        except Exception as exc:
            raise RateLimitStoreError(str(exc)) from exc

    def get_ttl_seconds(self, *, key: str) -> Optional[int]:
        try:
            ttl = self._client.ttl(key)
            if ttl is None or int(ttl) <= 0:
                return None
            return int(ttl)
        except Exception as exc:
            raise RateLimitStoreError(str(exc)) from exc


def build_rate_limit_store(redis_url: Optional[str]) -> RateLimitStore:
    if redis_url:
        try:
            return RedisRateLimitStore(redis_url)
        except Exception:
            # caller is expected to fall back
            pass
    return MemoryRateLimitStore()

