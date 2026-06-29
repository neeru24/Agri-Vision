# TODO: Enterprise RBAC for Agri-Vision

## Step 1: Repo analysis (completed baseline)
- [x] Identify current auth/role patterns in `app.py` and `models.py`.
- [x] Confirm existing JWT/refresh + audit utilities under `auth/`.

## Step 2: RBAC schema & models
- [x] Extend `models.py` with normalized RBAC tables: `roles`, `permissions`, `role_permissions`, `user_roles`.

- [x] Implement `User` helpers to support both legacy `User.role` and RBAC (backward compatible but RBAC authoritative).


## Step 3: Seed roles/permissions/mappings
- [x] Add `auth/rbac_seed.py` to seed: Admin/Farmer/Researcher/Moderator.
- [x] Add idempotent default permissions + role-permission mappings.

## Step 4: Permission resolution engine
- [x] Add `auth/rbac.py` with a centralized resolution function returning effective permissions for a user (handles multiple roles).
- [x] Optimize queries to avoid N+1.

## Step 5: Authorization middleware / decorators
- [x] Add `auth/authorization.py` with decorators: `requireRole`, `requirePermission`, `requireAnyRole`, `requireAnyPermission`, `requireAllPermissions`.
- [x] Enforce deny-by-default and ensure 401/403 semantics.
- [x] Emit audit events for denied access.

## Step 6: Integrate authorization into Flask routes
- [x] Update `app.py` to protect privileged endpoints using the new decorators.
- [x] Remove/replace ad-hoc `current_user.is_researcher()` checks where appropriate.

## Step 7: Admin-only RBAC management APIs
- [x] Implement Admin-only APIs for role/permission CRUD and role assignment/removal.

## Step 8: Audit logging integration
- [x] Ensure role/permission changes emit audit events.
- [x] Ensure authorization denials emit audit events.

## Step 9: Tests
- [x] Add unit tests for permission resolution.
- [x] Add tests for decorators/endpoint protection.
- [x] Add security tests for privilege escalation attempts.
- [x] Add integration tests ensuring login + authorized access.

## Step 10: Documentation
- [x] Update `README.md` or `architecture.md` with RBAC design + usage.

## Step 11: Verification
- [x] Run `pytest`.
- [x] Run lint/type checks as applicable.
- [x] Run build/deploy checks as applicable.
