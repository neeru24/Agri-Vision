def test_admin_models_requires_login(client, app):
    """Unauthenticated requests to admin endpoints should be redirected/denied."""
    app.config["LOGIN_DISABLED"] = False
    try:
        resp = client.get("/admin/models")
        assert resp.status_code in (302, 401, 403)
    finally:
        app.config["LOGIN_DISABLED"] = True


def test_admin_export_requires_login(client, app):
    app.config["LOGIN_DISABLED"] = False
    try:
        resp = client.get("/admin/models/export/pdf")
        assert resp.status_code in (302, 401, 403)
    finally:
        app.config["LOGIN_DISABLED"] = True
