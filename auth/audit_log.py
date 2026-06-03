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

