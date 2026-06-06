from __future__ import annotations

from datetime import datetime, timedelta
import threading

import pytest

from auth.rotation_service import RefreshRotationError, rotate_refresh_token
from auth.jwt_utils import create_access_token, create_refresh_token
from auth.token_crypto import sha256_hex
from models import db, User, RefreshTokenFamily, RefreshToken


@pytest.fixture()
def app_with_db(monkeypatch):
    # Uses the existing pytest configuration; relies on app.py db.init_app already.
    # Import app to create application context.
    import app as flask_app

    flask_app.app.config.update(
        {"TESTING": True, "SQLALCHEMY_DATABASE_URI": "sqlite:///:memory:"}
    )

    with flask_app.app.app_context():
        db.create_all()
        yield flask_app.app
        db.session.remove()
        db.drop_all()


def _seed_user_and_family(db_session):
    user = User(email="u@example.com", full_name="User", role="farmer")
    user.set_password("password123")
    db_session.add(user)
    db_session.commit()

    family = RefreshTokenFamily(user_id=user.id, session_id="sessA")
    db_session.add(family)
    db_session.commit()
    return user, family


def test_successful_rotation_rejects_old_token(app_with_db):
    from auth.jwt_utils import new_jti

    with app_with_db.app_context():
        user, family = _seed_user_and_family(db.session)

        # Create initial refresh token row.
        token_id = new_jti()
        refresh_raw = create_refresh_token(user_id=user.id, session_id="sessA", family_id=family.id, jti=token_id)
        refresh_hash = sha256_hex(refresh_raw)

        now = datetime.utcnow()
        expires_at = now + timedelta(days=14)

        row = RefreshToken(
            id=token_id,
            family_id=family.id,
            user_id=user.id,
            session_id="sessA",
            token_hash=refresh_hash,
            created_at=now,
            expires_at=expires_at,
            revoked_at=None,
            replaced_by_token_id=None,
            is_compromised=False,
            last_used_at=None,
            created_ip="127.0.0.1",
            created_user_agent="pytest",
        )
        db.session.add(row)
        db.session.commit()

        # First refresh should succeed.
        access1, refresh2 = rotate_refresh_token(
            raw_refresh_token=refresh_raw,
            request_id="req1",
            ip="127.0.0.1",
            user_agent="pytest",
        )
        assert access1
        assert refresh2

        # Old refresh token should now be rejected as reuse.
        with pytest.raises(RefreshRotationError) as excinfo:
            rotate_refresh_token(
                raw_refresh_token=refresh_raw,
                request_id="req2",
                ip="127.0.0.1",
                user_agent="pytest",
            )
        assert excinfo.value.code in {"reuse", "compromised"}


def test_concurrent_refresh_only_one_succeeds(app_with_db):
    from auth.jwt_utils import new_jti

    with app_with_db.app_context():
        user, family = _seed_user_and_family(db.session)

        token_id = new_jti()
        refresh_raw = create_refresh_token(user_id=user.id, session_id="sessA", family_id=family.id, jti=token_id)
        refresh_hash = sha256_hex(refresh_raw)

        now = datetime.utcnow()
        expires_at = now + timedelta(days=14)

        row = RefreshToken(
            id=token_id,
            family_id=family.id,
            user_id=user.id,
            session_id="sessA",
            token_hash=refresh_hash,
            created_at=now,
            expires_at=expires_at,
            revoked_at=None,
            replaced_by_token_id=None,
            is_compromised=False,
            last_used_at=None,
            created_ip="127.0.0.1",
            created_user_agent="pytest",
        )
        db.session.add(row)
        db.session.commit()

        results = {"ok": [], "err": []}

        def worker(idx: int, app) -> None:
            with app.app_context():
                try:
                    access, refresh2 = rotate_refresh_token(
                        raw_refresh_token=refresh_raw,
                        request_id=f"req{idx}",
                        ip="127.0.0.1",
                        user_agent="pytest",
                    )
                    results["ok"].append((access, refresh2))
                except RefreshRotationError as e:
                    results["err"].append((e.code, str(e)))

        t1 = threading.Thread(target=worker, args=(1, app_with_db))
        t2 = threading.Thread(target=worker, args=(2, app_with_db))
        t1.start()
        import time; time.sleep(0.05)
        t2.start()
        t1.join()
        t2.join()

        assert len(results["ok"]) == 1
        assert len(results["err"]) == 1

