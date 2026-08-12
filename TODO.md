# TODO - Dev auth bypass

- [x] Add a dev-only configuration flag (based on app.config["DEBUG"] or an env var) that disables auth checks.

- [ ] Implement bypass by short-circuiting auth decorators in auth/authorization.py.
- [ ] Ensure current_user is treated as authenticated (without altering prod behavior).
- [ ] Update app.py to set the bypass flag from DEBUG.
- [ ] Run minimal checks: start Flask and verify routes like /dashboard, /reports, /profile, /model-admin are not redirected/aborted.
- [ ] Document changes in-file with comments explaining why.
