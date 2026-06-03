from __future__ import annotations

from datetime import datetime, timedelta
from typing import Any, Dict, Optional, Tuple

from auth.jwt_utils import create_access_token, create_refresh_token, decode_token, new_jti
from auth.token_crypto import sha256_hex
from models import RefreshToken, RefreshTokenFamily, db


class RefreshRotationError(Exception):
    def __init__(self, code: str, message: str) -> None:
        super().__init__(message)
        self.code = code
        self.message = message


def _utcnow() -> datetime:
    return datetime.utcnow()


def rotate_refresh_token(
    *,
    raw_refresh_token: str,
    request_id: str,
    ip: Optional[str],
    user_agent: Optional[str],
) -> Tuple[str, str]:

    """Rotate a refresh token with replay detection.

    Returns: (access_token, new_refresh_token)
    Raises RefreshRotationError on failure.
    """

    claims, err = decode_token(raw_refresh_token)
    if err is not None or claims is None:
        raise RefreshRotationError("unauthorized", "Invalid or expired refresh token")

    if claims.get("type") != "refresh":
        raise RefreshRotationError("unauthorized", "Invalid token type")

    user_id = str(claims.get("sub"))
    session_id = str(claims.get("session_id"))
    family_id = str(claims.get("family_id"))
    token_id = str(claims.get("jti"))

    token_hash = sha256_hex(raw_refresh_token)

    now = _utcnow()

    # Concurrency safety: single transaction + conditional revoke of the current token.
    # We treat "revoked_at is null" as the one-time-use gate.
    with db.session.begin():
        token_row = (
            RefreshToken.query.filter(
                RefreshToken.family_id == family_id,
                RefreshToken.user_id == user_id,
                RefreshToken.session_id == session_id,
                RefreshToken.token_hash == token_hash,
                RefreshToken.id == token_id,
            )
            .with_for_update()
            .one_or_none()
        )

        if token_row is None:
            # Could be tampered token or never issued.
            raise RefreshRotationError("unauthorized", "Invalid refresh token")

        # Expiration check (defense in depth; JWT exp already validated).
        if token_row.expires_at < now:
            raise RefreshRotationError("unauthorized", "Expired refresh token")

        family = RefreshTokenFamily.query.filter(
            RefreshTokenFamily.id == family_id,
            RefreshTokenFamily.user_id == user_id,
        ).one_or_none()

        if family is None:
            raise RefreshRotationError("unauthorized", "Invalid token family")

        # If family already compromised, block refresh and require reauth.
        if family.is_compromised:
            raise RefreshRotationError("compromised", "Refresh token family compromised")

        # Replay detection: if token already revoked/replaced => reuse.
        if token_row.revoked_at is not None or token_row.replaced_by_token_id is not None:
            # Mark family compromised and reject.
            if not family.is_compromised:
                family.is_compromised = True
                family.compromised_at = now
                # Mark current token compromised as well.
                token_row.is_compromised = True
            db.session.add(family)
            db.session.add(token_row)
            raise RefreshRotationError("reuse", "Refresh token reuse detected")

        # If the token is active, rotate:
        new_family_id = family_id  # same family chain
        new_token_id = new_jti()

        # Revoke old token and link replacement.
        token_row.revoked_at = now
        token_row.replaced_by_token_id = new_token_id
        token_row.last_used_at = now
        token_row.created_ip = ip
        token_row.created_user_agent = user_agent

        # Create new refresh token row.
        new_refresh_raw = create_refresh_token(
            user_id=user_id,
            session_id=session_id,
            family_id=new_family_id,
            jti=new_token_id,
        )
        new_refresh_hash = sha256_hex(new_refresh_raw)

        new_exp = now + timedelta(seconds=int(claims.get("exp") - claims.get("iat", now.timestamp())))
        # The above is best-effort; token JWT exp already contains actual TTL.
        # We'll also set expires_at from the decoded claims exp.
        decoded_exp = claims.get("exp")
        if decoded_exp is not None:
            try:
                new_exp = datetime.utcfromtimestamp(int(decoded_exp) - (int(claims.get("iat", now.timestamp())) - int(claims.get("iat", now.timestamp()))))
            except Exception:
                pass

        # Prefer JWT exp to avoid mistakes
        refresh_cfg_exp_seconds = None
        try:
            refresh_cfg_exp_seconds = int(new_refresh_raw and 0)
        except Exception:
            refresh_cfg_exp_seconds = None

        # We'll compute expires_at from JWT by decoding again cheaply.
        new_claims, _ = decode_token(new_refresh_raw)
        expires_at = now + timedelta(seconds=int(new_claims.get("exp") - new_claims.get("iat"))) if new_claims else (now + timedelta(days=14))

        new_row = RefreshToken(
            id=new_token_id,
            family_id=new_family_id,
            user_id=user_id,
            session_id=session_id,
            token_hash=new_refresh_hash,
            created_at=now,
            expires_at=expires_at,
            revoked_at=None,
            replaced_by_token_id=None,
            is_compromised=False,
            last_used_at=None,
            created_ip=ip,
            created_user_agent=user_agent,
        )
        db.session.add(new_row)

        # Issue access token.
        access_jti = new_jti()
        access_token = create_access_token(
            user_id=user_id,
            session_id=session_id,
            family_id=new_family_id,
            jti=access_jti,
        )

    return access_token, new_refresh_raw

