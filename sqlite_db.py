"""
SQLite-specific engine hooks so concurrent writers serialize like production Postgres.

SQLite's default deferred transactions let two connections both observe an unchanged
row before either commits; ``SELECT ... FOR UPDATE`` does not match Postgres row
locks. ``BEGIN IMMEDIATE`` reserves a write lock at transaction start.

See: https://docs.sqlalchemy.org/en/20/dialects/sqlite.html#serializable-isolation-savepoints-transactional-ddl
"""

from __future__ import annotations

import weakref

from sqlalchemy import event
from sqlalchemy.engine import Engine

# Avoid stacking duplicate listeners if ``app`` is imported more than once (e.g. tools, reload).
_configured: weakref.WeakKeyDictionary[Engine, bool] = weakref.WeakKeyDictionary()


def configure_sqlite_immediate_transactions(engine: Engine) -> None:
    """Register connect/begin hooks so each transaction opens with BEGIN IMMEDIATE."""
    if engine.dialect.name != "sqlite":
        return
    if _configured.get(engine):
        return

    def _on_connect(dbapi_conn, connection_record):
        dbapi_conn.isolation_level = None

    def _on_begin(conn):
        conn.exec_driver_sql("BEGIN IMMEDIATE")

    event.listen(engine, "connect", _on_connect, insert=True)
    event.listen(engine, "begin", _on_begin)
    _configured[engine] = True
