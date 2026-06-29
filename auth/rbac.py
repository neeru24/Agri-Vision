from models import db, User, Role, Permission, RolePermission, UserRole

def get_effective_permissions(user: User) -> set[str]:
    """
    Returns a set of permission slugs that the user has.
    """
    if not user or not user.is_active:
        return set()

    # Query using joins to avoid N+1
    perms = db.session.query(Permission.slug).join(
        RolePermission, Permission.id == RolePermission.permission_id
    ).join(
        Role, Role.id == RolePermission.role_id
    ).join(
        UserRole, UserRole.role_id == Role.id
    ).filter(
        UserRole.user_id == user.id,
        Role.deleted_at.is_(None)
    ).all()

    permissions = {p[0] for p in perms}

    # Fallback for legacy roles if no UserRole mappings exist
    if not permissions and user.role:
        # Get permissions mapped to the user's legacy role string
        legacy_role = Role.query.filter_by(slug=user.role, deleted_at=None).first()
        if legacy_role:
            legacy_perms = db.session.query(Permission.slug).join(
                RolePermission, Permission.id == RolePermission.permission_id
            ).filter(
                RolePermission.role_id == legacy_role.id
            ).all()
            permissions = {p[0] for p in legacy_perms}

    return permissions

def has_permission(user: User, permission_slug: str) -> bool:
    """Check if user has a specific permission."""
    return permission_slug in get_effective_permissions(user)

def has_any_permission(user: User, permission_slugs: list[str]) -> bool:
    user_perms = get_effective_permissions(user)
    return any(p in user_perms for p in permission_slugs)

def has_all_permissions(user: User, permission_slugs: list[str]) -> bool:
    user_perms = get_effective_permissions(user)
    return all(p in user_perms for p in permission_slugs)

def has_role(user: User, role_slug: str) -> bool:
    """Check if user has a specific role (by slug)."""
    if not user or not user.is_active:
        return False
    
    # Check new RBAC tables
    has_new_role = db.session.query(UserRole).join(
        Role, Role.id == UserRole.role_id
    ).filter(
        UserRole.user_id == user.id,
        Role.slug == role_slug,
        Role.deleted_at.is_(None)
    ).first() is not None

    if has_new_role:
        return True

    # Fallback to legacy
    return user.role == role_slug

def has_any_role(user: User, role_slugs: list[str]) -> bool:
    return any(has_role(user, r) for r in role_slugs)

