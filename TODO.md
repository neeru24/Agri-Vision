# TODO - Account Lockout Protection (Agri-Vision)

## Phase 1: Discovery & Architecture
- [x] Read `app.py` login flow
- [x] Read `models.py` User model
- [x] Read `services/lockout_service.py` (existing skeleton)
- [x] Read `services/rate_limit_store.py` (memory/redis)
- [x] Read `config/rate_limit_config.py` (existing lockout envs)
- [x] Read `auth/audit_log.py`
- [x] Produce full internal architecture + integration mapping

## Phase 2: Implementation
- [x] Extend `services/lockout_service.py` to expose lock transition result and cooldown info
- [x] Add lockout policy config/env validation (reuse `config/rate_limit_config.py` or extend)
- [x] Update `models.py` User model with lockout fields (failed attempts, timestamps, locked until, IPs)
- [x] Update `app.py` `/login` route:
  - [x] check lockout status before password verification
  - [x] record failed attempts on wrong password
  - [x] reset counters + unlock fields on successful login
  - [x] add security audit events for auth success/fail/locked/unlocked
  - [x] keep response/message consistent (no user enumeration)
- [x] Add migration strategy/tooling (or ensure `db.create_all()` covers new fields in tests)

## Phase 3: Testing
- [x] Unit tests for `services/lockout_service.py`
- [x] Unit tests for lockout policy/env validation
- [x] Integration tests for Flask `/login` lockout behavior
- [x] Security tests for response consistency / no enumeration
- [x] Regression: ensure existing tests still pass

## Phase 4: Build/CI/Vercel
- [x] Run `pytest`
- [x] Run lint/type-check/build (repo scripts)
- [x] Ensure CI (GitHub Actions) passes
- [x] Ensure Vercel deployment is safe (no server-only imports at module scope)

## Phase 5: Documentation & Changelog
- [x] Update README/security docs + env var guide
- [x] Add CHANGELOG entry for lockout feature
