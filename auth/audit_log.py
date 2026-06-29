from __future__ import annotations

import logging
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)


def log_security_event(*, event: str, user_id: Optional[str], session_id: Optional[str], family_id: Optional[str], request_id: Optional[str], ip: Optional[str], user_agent: Optional[str], details: Optional[Dict[str, Any]] = None) -> None:
    payload: Dict[str, Any] = {
        "event": event,
        "userId": user_id,
        "sessionId": session_id,
        "familyId": family_id,
        "requestId": request_id,
        "ip": ip,
        "userAgent": user_agent,
        "details": details or {},
    }

    logger.warning(payload)


def log_audit_event(event_type: str, message: str, user_id: Optional[str] = None) -> None:
    log_security_event(
        event=event_type,
        user_id=user_id,
        session_id=None,
        family_id=None,
        request_id=None,
        ip=None,
        user_agent=None,
        details={"message": message}
    )


