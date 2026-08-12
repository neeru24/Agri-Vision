from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List, Optional, Tuple

from auth.device_detection import parse_device_metadata
from auth.token_crypto import sha256_hex
from models import DeviceSession, RefreshToken, RefreshTokenFamily, db


@dataclass(frozen=True)
class SessionContext:
    user_id: str
    session_id: str
    refresh_token_family_id: str
    jti: str
    ip_address: Optional[str]
    user_agent: Optional[str]


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


def _throttle_should_update(prev: Optional[datetime], *, throttle_seconds: int) -> bool:
    if prev is None:
        return True
    return (_utcnow() - prev) >= timedelta(seconds=throttle_seconds)


def create_session_or_update_current(
    *,
    user_id: str,
    session_id: str,
    refresh_token_family_id: str,
    ip_address: Optional[str],
    user_agent: Optional[str],
    is_current: bool = True,
    expires_at: Optional[datetime] = None,
    device_write_throttle_seconds: int = 60,
) -> DeviceSession:
    """Create a DeviceSession row if missing, and mark it current.

    Designed to be safe for initial token issuance and refresh-linked session updates.
    """

    now = _utcnow()
    # best-effort device parsing
    ua = user_agent or ""
    device_name, browser_name, operating_system, device_type = parse_device_metadata(ua)
    country = None
    city = None

    # Ownership/current handling: for the same user and session scope, only one current session.
    # We rely on session_id uniqueness coming from JWT generation.
    # If already exists, update metadata with throttling.

    existing = (
        DeviceSession.query.filter_by(user_id=user_id, session_id=session_id, refresh_token_family_id=refresh_token_family_id)
        .one_or_none()
    )

    if existing is None:
        row = DeviceSession(
            user_id=user_id,
            session_id=session_id,
            refresh_token_family_id=refresh_token_family_id,
            device_name=device_name,
            browser_name=browser_name,
            operating_system=operating_system,
            device_type=device_type,
            user_agent=user_agent,
            ip_address=ip_address,
            country=country,
            city=city,
            is_current=is_current,
            is_active=True,
            created_at=now,
            updated_at=now,
            last_activity_at=now,
            revoked_at=None,
            expires_at=expires_at,
        )
        db.session.add(row)
        db.session.flush()
        if is_current:
            _unset_other_current(user_id=user_id, session_id=session_id, refresh_token_family_id=refresh_token_family_id)
        db.session.commit()
        return row

    # Update current flag and metadata (throttled last_activity)
    if is_current and not existing.is_current:
        _unset_other_current(user_id=user_id, session_id=session_id, refresh_token_family_id=refresh_token_family_id)
        existing.is_current = True

    # Update device/IP/UA
    existing.device_name = device_name
    existing.browser_name = browser_name
    existing.operating_system = operating_system
    existing.device_type = device_type
    existing.user_agent = user_agent
    existing.ip_address = ip_address

    if expires_at is not None:
        existing.expires_at = expires_at

    if _throttle_should_update(existing.last_activity_at, throttle_seconds=device_write_throttle_seconds):
        existing.last_activity_at = now

    db.session.add(existing)
    db.session.commit()
    return existing


def _unset_other_current(*, user_id: str, session_id: str, refresh_token_family_id: str) -> None:
    DeviceSession.query.filter(
        DeviceSession.user_id == user_id,
        DeviceSession.session_id != session_id,
        DeviceSession.refresh_token_family_id == refresh_token_family_id,
    ).update({"is_current": False}, synchronize_session=False)


def touch_session_activity(
    *,
    user_id: str,
    session_id: str,
    refresh_token_family_id: str,
    ip_address: Optional[str],
    user_agent: Optional[str],
    throttle_seconds: int = 60,
) -> None:
    now = _utcnow()
    row = (
        DeviceSession.query.filter_by(
            user_id=user_id,
            session_id=session_id,
            refresh_token_family_id=refresh_token_family_id,
        ).one_or_none()
    )
    if row is None:
        # Best-effort: create it if missing.
        create_session_or_update_current(
            user_id=user_id,
            session_id=session_id,
            refresh_token_family_id=refresh_token_family_id,
            ip_address=ip_address,
            user_agent=user_agent,
            is_current=False,
            expires_at=None,
        )
        return


    if row.is_active and _throttle_should_update(row.last_activity_at, throttle_seconds=throttle_seconds):
        row.last_activity_at = now
        row.ip_address = ip_address
        row.user_agent = user_agent
        db.session.add(row)
        db.session.commit()


def list_active_sessions(
    *,
    user_id: str,
    current_session_id: Optional[str],
    limit: int = 50,
) -> List[DeviceSession]:
    q = DeviceSession.query.filter(DeviceSession.user_id == user_id, DeviceSession.is_active == True)
    if current_session_id is not None:
        q = q.order_by(DeviceSession.session_id == current_session_id)
    q = q.order_by(DeviceSession.last_activity_at.desc())
    rows = q.limit(limit).all()
    return rows


def revoke_session(*, user_id: str, session_id: str, ip_address: Optional[str], user_agent: Optional[str]) -> bool:
    now = _utcnow()
    row = DeviceSession.query.filter_by(user_id=user_id, session_id=session_id).one_or_none()
    if row is None:
        return False

    if row.is_active:
        row.is_active = False
        row.revoked_at = now
        row.is_current = False
        row.last_activity_at = row.last_activity_at
        db.session.add(row)

    # revoke entire refresh token family
    family = RefreshTokenFamily.query.filter_by(id=row.refresh_token_family_id, user_id=user_id).one_or_none()
    if family is not None:
        family.is_compromised = True
        family.compromised_at = now
        db.session.add(family)

        # Also mark existing refresh tokens inactive/consumed
        RefreshToken.query.filter_by(family_id=family.id, user_id=user_id).update(
            {"revoked_at": now, "is_compromised": True}, synchronize_session=False
        )

    db.session.commit()
    return True


def revoke_others(*, user_id: str, keep_session_id: str, ip_address: Optional[str], user_agent: Optional[str]) -> int:
    now = _utcnow()
    q = DeviceSession.query.filter(DeviceSession.user_id == user_id, DeviceSession.session_id != keep_session_id, DeviceSession.is_active == True)
    rows = q.all()
    revoked = 0
    for r in rows:
        revoked += 1
        revoke_session(user_id=user_id, session_id=r.session_id, ip_address=ip_address, user_agent=user_agent)
    return revoked

