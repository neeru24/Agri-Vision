from functools import wraps
from flask import abort, jsonify
from flask_login import current_user
from auth.rbac import has_role, has_permission, has_any_permission, has_all_permissions
from auth.audit_log import log_audit_event

def require_role(role_slug: str):
    def decorator(f):
        @wraps(f)
        def decorated_function(*args, **kwargs):
            if not current_user.is_authenticated:
                return abort(401)
            if not has_role(current_user, role_slug):
                log_audit_event(
                    "AUTHZ_DENIED",
                    f"User {current_user.id} denied access to {f.__name__} (requires role {role_slug})",
                    user_id=current_user.id
                )
                return abort(403)
            return f(*args, **kwargs)
        return decorated_function
    return decorator

def require_any_role(role_slugs: list[str]):
    def decorator(f):
        @wraps(f)
        def decorated_function(*args, **kwargs):
            if not current_user.is_authenticated:
                return abort(401)
            if not any(has_role(current_user, role) for role in role_slugs):
                log_audit_event(
                    "AUTHZ_DENIED",
                    f"User {current_user.id} denied access to {f.__name__} (requires any of {role_slugs})",
                    user_id=current_user.id
                )
                return abort(403)
            return f(*args, **kwargs)
        return decorated_function
    return decorator

def require_permission(permission_slug: str):
    def decorator(f):
        @wraps(f)
        def decorated_function(*args, **kwargs):
            if not current_user.is_authenticated:
                return abort(401)
            if not has_permission(current_user, permission_slug):
                log_audit_event(
                    "AUTHZ_DENIED",
                    f"User {current_user.id} denied access to {f.__name__} (requires permission {permission_slug})",
                    user_id=current_user.id
                )
                return abort(403)
            return f(*args, **kwargs)
        return decorated_function
    return decorator

def require_any_permission(permission_slugs: list[str]):
    def decorator(f):
        @wraps(f)
        def decorated_function(*args, **kwargs):
            if not current_user.is_authenticated:
                return abort(401)
            if not has_any_permission(current_user, permission_slugs):
                log_audit_event(
                    "AUTHZ_DENIED",
                    f"User {current_user.id} denied access to {f.__name__} (requires any of {permission_slugs})",
                    user_id=current_user.id
                )
                return abort(403)
            return f(*args, **kwargs)
        return decorated_function
    return decorator

def require_all_permissions(permission_slugs: list[str]):
    def decorator(f):
        @wraps(f)
        def decorated_function(*args, **kwargs):
            if not current_user.is_authenticated:
                return abort(401)
            if not has_all_permissions(current_user, permission_slugs):
                log_audit_event(
                    "AUTHZ_DENIED",
                    f"User {current_user.id} denied access to {f.__name__} (requires all of {permission_slugs})",
                    user_id=current_user.id
                )
                return abort(403)
            return f(*args, **kwargs)
        return decorated_function
    return decorator
