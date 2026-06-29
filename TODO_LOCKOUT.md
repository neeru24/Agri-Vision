# Account Lockout Protection - Implementation Checklist

## Phase 1 (Discovery / Integration)
- [x] Verify login flow: `app.py` route `/login` (password-based via Flask-Login)
- [x] Verify existing lockout logic: `services/lockout_service.py`
- [x] Verify existing audit logging: `auth/audit_log.py`
- [x] Verify existing env config: `config/rate_limit_config.py`

## Phase 2 (Code changes)
- [x] Wire `LockoutService` into `POST /login`:

- [x] Check `is_locked()` before password verification

  - [x] On locked: return HTTP 423 + flash generic message

  - [x] On failed password: `record_failed_login()` + audit `AUTH_FAILED`
  - [x] On successful password: audit `AUTH_SUCCESS` and ensure counters reset behavior
- [x] Add `ACCOUNT_LOCKED` audit event when lock becomes active
  - [x] Keep responses non-enumerating


## Phase 3 (Tests)
- [x] Unit tests for `AccountLockoutService`:
  - [x] counter increment / threshold
  - [x] lock TTL / cooldown

- [x] Integration tests for Flask `/login`:
  - [x] configured failures -> locked -> 423
  - [x] cooldown expiry -> successful login allowed

## Phase 4 (Quality / CI)
- [x] Run `pytest` locally
- [x] Fix any lint/test regressions
