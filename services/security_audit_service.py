from __future__ import annotations

from datetime import datetime, timezone
from typing import Optional

from auth.audit_log import log_security_event
from config.security_config import load_account_lockout_config


def log_auth_security_event(
    *,
    action: str,
    severity: str,
    user_id: Optional[str],
    email: Optional[str],
    ip: Optional[str],
    user_agent: Optional[str],
    metadata: Optional[dict[str, object]] = None,
) -> None:
    if not load_account_lockout_config().audit_enabled:
        return

    details: dict[str, object] = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "action": action,
        "severity": severity,
        "email": email,
        "metadata": metadata or {},
    }
    log_security_event(
        event=action,
        user_id=user_id,
        session_id=None,
        family_id=None,
        request_id=None,
        ip=ip,
        user_agent=user_agent,
        details=details,
    )
