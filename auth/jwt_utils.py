from __future__ import annotations

import time
import uuid
from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple

import jwt


@dataclass(frozen=True)
class JwtConfig:
    issuer: str
    access_ttl_seconds: int
    refresh_ttl_seconds: int
    audience: Optional[str] = None


def get_jwt_secret() -> str:
    # Required for production. For local dev we fall back to existing SECRET_KEY.
    # Keep it consistent with app.py.
    import os

    secret = os.getenv("JWT_SECRET") or os.getenv("SECRET_KEY")
    if not secret:
        raise RuntimeError("JWT_SECRET (or SECRET_KEY) must be configured")
    return secret


def get_jwt_config() -> JwtConfig:
    import os

    issuer = os.getenv("JWT_ISSUER", "agri-vision")
    access_ttl_seconds = int(os.getenv("ACCESS_TOKEN_TTL_SECONDS", "900"))
    refresh_ttl_seconds = int(os.getenv("REFRESH_TOKEN_TTL_SECONDS", "1209600"))  # 14 days
    audience = os.getenv("JWT_AUDIENCE")

    return JwtConfig(
        issuer=issuer,
        access_ttl_seconds=access_ttl_seconds,
        refresh_ttl_seconds=refresh_ttl_seconds,
        audience=audience if audience else None,
    )


def _now() -> int:
    return int(time.time())


def create_access_token(*, user_id: str, session_id: str, family_id: str, jti: str) -> str:
    cfg = get_jwt_config()
    secret = get_jwt_secret()

    now = _now()
    payload: Dict[str, Any] = {
        "iss": cfg.issuer,
        "sub": user_id,
        "aud": cfg.audience,
        "iat": now,
        "exp": now + cfg.access_ttl_seconds,
        "jti": jti,
        "type": "access",
        "session_id": session_id,
        "family_id": family_id,
    }
    return jwt.encode(payload, secret, algorithm="HS256")


def create_refresh_token(*, user_id: str, session_id: str, family_id: str, jti: str) -> str:
    cfg = get_jwt_config()
    secret = get_jwt_secret()

    now = _now()
    payload: Dict[str, Any] = {
        "iss": cfg.issuer,
        "sub": user_id,
        "aud": cfg.audience,
        "iat": now,
        "exp": now + cfg.refresh_ttl_seconds,
        "jti": jti,
        "type": "refresh",
        "session_id": session_id,
        "family_id": family_id,
    }
    return jwt.encode(payload, secret, algorithm="HS256")


def decode_token(token: str) -> Tuple[Optional[Dict[str, Any]], Optional[str]]:
    """Returns (claims, error_code)."""
    cfg = get_jwt_config()
    secret = get_jwt_secret()
    try:
        claims = jwt.decode(
            token,
            secret,
            algorithms=["HS256"],
            audience=cfg.audience,
            issuer=cfg.issuer,
        )
        return claims, None
    except jwt.ExpiredSignatureError:
        return None, "expired"
    except jwt.InvalidTokenError:
        return None, "invalid"


def new_jti() -> str:
    return str(uuid.uuid4())

