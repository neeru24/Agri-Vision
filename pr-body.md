## Summary
- Configure SQLite transactions with BEGIN IMMEDIATE so concurrent refresh rotation serializes like production row locking.
- Harden refresh rotation tests: shared file DB, per-thread Flask app context, docstring and assertions.

## Test plan
- [ ] `pytest tests/test_refresh_rotation.py -v`
- [ ] (Optional) `pytest tests/test_refresh_rotation.py::test_concurrent_refresh_only_one_succeeds --count=50 -v` if pytest-repeat is installed
