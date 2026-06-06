from __future__ import annotations

import hashlib
import time
from dataclasses import dataclass
from typing import Optional

from services.rate_limit_store import RateLimitStore



def _sha256_hex(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class LockoutState:
    locked: bool
    lockout_until_epoch: Optional[int]


class LockoutService:
    def __init__(
        self,
        *,
        store: RateLimitStore,
        failed_attempts: int,
        cooldown_seconds: int,
    ) -> None:
        self._store = store
        self._failed_attempts = failed_attempts
        self._cooldown_seconds = cooldown_seconds

    def _key_ip(self, ip: str) -> str:
        return f"lock:auth:login:ip:{ip}"

    def _key_user(self, email: str) -> str:
        email_hash = _sha256_hex(email.strip().lower())
        return f"lock:auth:login:user:{email_hash}"

    def is_locked(self, *, ip: Optional[str], email: Optional[str]) -> LockoutState:
        now = int(time.time())

        if ip:
            ttl_ip = self._store.get_ttl_seconds(key=self._key_ip(ip))
            if ttl_ip is not None and ttl_ip > 0:
                return LockoutState(locked=True, lockout_until_epoch=now + ttl_ip)

        if email:
            ttl_user = self._store.get_ttl_seconds(key=self._key_user(email))
            if ttl_user is not None and ttl_user > 0:
                return LockoutState(locked=True, lockout_until_epoch=now + ttl_user)

        return LockoutState(locked=False, lockout_until_epoch=None)

    def record_failed_login(self, *, ip: str, email: Optional[str]) -> None:
        # Implement lockout by tracking attempts and setting a lock key when threshold reached.
        # We do this with a two-step strategy:
        #  - increment fail counter in a rolling window = cooldown_seconds
        #  - when it crosses failed_attempts, set a lock key with expiry=cooldown_seconds
        # This avoids needing a delete() operation.

        if email:
            fail_key_user = f"fail:auth:login:user:{_sha256_hex(email.strip().lower())}"
            res_user = self._store.incr_with_expiry(
                key=fail_key_user,
                window_seconds=self._cooldown_seconds,
                limit=self._failed_attempts,
                increment=1,
            )
            if not res_user.allowed:
                self._store.incr_with_expiry(
                    key=self._key_user(email),
                    window_seconds=self._cooldown_seconds,
                    limit=1,
                    increment=1,
                )

        fail_key_ip = f"fail:auth:login:ip:{ip}"
        res_ip = self._store.incr_with_expiry(
            key=fail_key_ip,
            window_seconds=self._cooldown_seconds,
            limit=self._failed_attempts,
            increment=1,
        )
        if not res_ip.allowed:
            self._store.incr_with_expiry(
                key=self._key_ip(ip),
                window_seconds=self._cooldown_seconds,
                limit=1,
                increment=1,
            )

    def reset_on_success(self, *, ip: Optional[str], email: Optional[str]) -> None:
        # We don't have a delete primitive; counters naturally expire.
        # For usability, we still record a success event (handled in middleware) and do nothing here.
        return


