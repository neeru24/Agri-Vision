"""Account lockout checks and auth audit hooks used by ``app.login``."""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import TYPE_CHECKING, Optional

from config.security_config import AccountLockoutConfig, load_account_lockout_config
from services.security_audit_service import log_auth_security_event

if TYPE_CHECKING:
    from models import User


@dataclass(frozen=True)
class AccountLockoutState:
    locked: bool
    unlocked_expired_lock: bool
    locked_until: Optional[datetime]


def get_client_ip() -> Optional[str]:
    try:
        from flask import request

        forwarded_for = request.headers.get("X-Forwarded-For", "")
        if forwarded_for:
            return forwarded_for.split(",", 1)[0].strip()[:64]
        return (request.remote_addr or "")[:64] or None
    except RuntimeError:
        return None


def get_user_agent() -> Optional[str]:
    try:
        from flask import request

        return (request.headers.get("User-Agent") or "")[:255] or None
    except RuntimeError:
        return None


class AccountLockoutService:
    def __init__(self, config: Optional[AccountLockoutConfig] = None) -> None:
        self.config = config or load_account_lockout_config()

    def check_lockout(self, user: "User", now: Optional[datetime] = None) -> AccountLockoutState:
        if not self.config.enabled:
            return AccountLockoutState(False, False, None)

        current_time = now or datetime.utcnow()
        locked_until = user.account_locked_until
        if locked_until is None:
            return AccountLockoutState(False, False, None)

        if locked_until > current_time:
            return AccountLockoutState(True, False, locked_until)

        user.account_locked_until = None
        user.failed_login_attempts = 0
        return AccountLockoutState(False, True, None)

    def record_failed_login(
        self,
        user: "User",
        *,
        ip: Optional[str],
        user_agent: Optional[str],
        now: Optional[datetime] = None,
    ) -> AccountLockoutState:
        current_time = now or datetime.utcnow()
        user.failed_login_attempts = int(user.failed_login_attempts or 0) + 1
        user.last_failed_login_at = current_time
        user.last_failed_ip = ip

        log_auth_security_event(
            action="AUTH_FAILED",
            severity="medium",
            user_id=user.id,
            email=user.email,
            ip=ip,
            user_agent=user_agent,
            metadata={"failedAttempts": user.failed_login_attempts},
        )

        if self.config.enabled and user.failed_login_attempts >= self.config.max_failed_attempts:
            user.account_locked_until = current_time + timedelta(
                minutes=self.config.lockout_duration_minutes
            )
            log_auth_security_event(
                action="ACCOUNT_LOCKED",
                severity="high",
                user_id=user.id,
                email=user.email,
                ip=ip,
                user_agent=user_agent,
                metadata={
                    "failedAttempts": user.failed_login_attempts,
                    "lockedUntil": user.account_locked_until.isoformat(),
                },
            )
            return AccountLockoutState(True, False, user.account_locked_until)

        return AccountLockoutState(False, False, None)

    def record_successful_login(
        self,
        user: "User",
        *,
        ip: Optional[str],
        user_agent: Optional[str],
        now: Optional[datetime] = None,
    ) -> None:
        current_time = now or datetime.utcnow()
        user.failed_login_attempts = 0
        user.last_failed_login_at = None
        user.account_locked_until = None
        user.last_successful_login_at = current_time
        user.last_successful_ip = ip

        log_auth_security_event(
            action="AUTH_SUCCESS",
            severity="low",
            user_id=user.id,
            email=user.email,
            ip=ip,
            user_agent=user_agent,
            metadata={"lastSuccessfulLoginAt": current_time.isoformat()},
        )

    def record_unlock(
        self,
        user: "User",
        *,
        ip: Optional[str],
        user_agent: Optional[str],
    ) -> None:
        log_auth_security_event(
            action="ACCOUNT_UNLOCKED",
            severity="medium",
            user_id=user.id,
            email=user.email,
            ip=ip,
            user_agent=user_agent,
            metadata={"reason": "lockout_expired"},
        )
