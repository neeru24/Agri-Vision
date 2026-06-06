from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Optional


def _env_bool(name: str, default: bool) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "t", "yes", "y", "on"}


def _env_int(name: str, default: int) -> int:
    raw = os.getenv(name)
    if raw is None or raw.strip() == "":
        return default
    return int(raw)


def _env_str(name: str, default: Optional[str] = None) -> Optional[str]:
    raw = os.getenv(name)
    if raw is None:
        return default
    return raw


@dataclass(frozen=True)
class RateLimitConfig:
    enabled: bool

    global_limit_window_seconds: int
    global_limit_max: int

    auth_limit_window_seconds: int
    auth_limit_max_per_ip: int
    auth_limit_max_per_account: int

    prediction_limit_window_seconds: int
    prediction_limit_max_per_ip: int
    prediction_limit_max_per_user: int

    admin_limit_window_seconds: int
    admin_limit_max_per_user: int

    lockout_failed_attempts: int
    lockout_cooldown_seconds: int

    # Abuse scoring thresholds
    abuse_score_lock_threshold: int
    abuse_score_decay_seconds: int

    # Headers
    headers_enabled: bool

    # Redis
    redis_url: Optional[str]


def load_rate_limit_config() -> RateLimitConfig:
    # Safe defaults aimed at minimizing false positives.
    enabled = _env_bool("RATE_LIMIT_ENABLED", True)

    global_limit_window_seconds = _env_int("GLOBAL_LIMIT_WINDOW", 60)
    global_limit_max = _env_int("GLOBAL_LIMIT_MAX", 100)

    auth_limit_window_seconds = _env_int("AUTH_LIMIT_WINDOW", 60)
    auth_limit_max_per_ip = _env_int("AUTH_LIMIT_MAX", 5)
    auth_limit_max_per_account = _env_int("AUTH_ACCOUNT_LIMIT_MAX", 5)

    prediction_limit_window_seconds = _env_int("PREDICTION_LIMIT_WINDOW", 60)
    prediction_limit_max_per_ip = _env_int("PREDICTION_LIMIT_MAX", 20)
    prediction_limit_max_per_user = _env_int("PREDICTION_USER_LIMIT_MAX", 40)

    admin_limit_window_seconds = _env_int("ADMIN_LIMIT_WINDOW", 60)
    admin_limit_max_per_user = _env_int("ADMIN_LIMIT_MAX", 300)

    lockout_failed_attempts = _env_int("LOCKOUT_FAILED_ATTEMPTS", 5)
    lockout_cooldown_seconds = _env_int("LOCKOUT_COOLDOWN_SECONDS", 900)  # 15 minutes

    abuse_score_lock_threshold = _env_int("ABUSE_SCORE_LOCK_THRESHOLD", 100)
    abuse_score_decay_seconds = _env_int("ABUSE_SCORE_DECAY_SECONDS", 300)

    headers_enabled = _env_bool("RATE_LIMIT_HEADERS_ENABLED", True)

    redis_url = _env_str("REDIS_URL")

    return RateLimitConfig(
        enabled=enabled,
        global_limit_window_seconds=global_limit_window_seconds,
        global_limit_max=global_limit_max,
        auth_limit_window_seconds=auth_limit_window_seconds,
        auth_limit_max_per_ip=auth_limit_max_per_ip,
        auth_limit_max_per_account=auth_limit_max_per_account,
        prediction_limit_window_seconds=prediction_limit_window_seconds,
        prediction_limit_max_per_ip=prediction_limit_max_per_ip,
        prediction_limit_max_per_user=prediction_limit_max_per_user,
        admin_limit_window_seconds=admin_limit_window_seconds,
        admin_limit_max_per_user=admin_limit_max_per_user,
        lockout_failed_attempts=lockout_failed_attempts,
        lockout_cooldown_seconds=lockout_cooldown_seconds,
        abuse_score_lock_threshold=abuse_score_lock_threshold,
        abuse_score_decay_seconds=abuse_score_decay_seconds,
        headers_enabled=headers_enabled,
        redis_url=redis_url,
    )

