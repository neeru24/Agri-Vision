def test_admin_models_requires_login(app):
    """Unauthenticated requests to admin endpoints should be redirected/denied."""
    # Use a fresh client without any session to ensure it's unauthenticated
    client = app.test_client()
    resp = client.get("/admin/models")
    assert resp.status_code in (302, 401, 403)


def test_admin_export_requires_login(app):
    client = app.test_client()
    resp = client.get("/admin/models/export/pdf")
    assert resp.status_code in (302, 401, 403)
