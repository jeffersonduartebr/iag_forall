# Objective: Regression tests for schema drift and lazy DDL found in the end-to-end run.
"""db_init must create every query_log column the workers query; calibration DDL runs once."""

import re
from contextlib import contextmanager
from types import SimpleNamespace

from app.services import judge_calibration

from app import db_manager, query_service


def test_db_init_query_log_has_every_column_the_api_adds():
    # O worker do NSGA depende só do db_init e consultava quality_source antes de a API
    # acrescentá-la ("Unknown column 'quality_source' in 'WHERE'").
    import inspect

    api_columns = set(re.findall(r"ADD COLUMN IF NOT EXISTS (\w+)", inspect.getsource(query_service.ensure_query_log)))
    db_init_columns = set(db_manager.SCHEMA_DEFINITIONS["query_log"]["columns"])
    assert api_columns, "ensure_query_log deveria declarar colunas via ADD COLUMN"
    assert api_columns <= db_init_columns, sorted(api_columns - db_init_columns)


def _recording_engine(statements):
    class _Conn:
        def execute(self, stmt, params=None):
            statements.append(str(stmt).split()[0].upper())

    @contextmanager
    def begin():
        yield _Conn()

    return SimpleNamespace(begin=begin)


def test_calibration_table_is_created_once_and_before_the_cache_update(monkeypatch):
    statements = []
    monkeypatch.setattr(judge_calibration, "_get_engine", lambda: _recording_engine(statements))
    monkeypatch.setattr(judge_calibration, "settings", SimpleNamespace(JUDGE_CALIBRATION_ENABLED=True))
    monkeypatch.setattr(judge_calibration, "_table_ready", False)

    judge_calibration.update_calibration_cache_status("q")  # cache antes de qualquer julgamento
    judge_calibration.record_judge_calibration("j1", "q", 7.0)
    judge_calibration.record_judge_calibration("j2", "q", 8.0)

    assert statements == ["CREATE", "UPDATE", "INSERT", "INSERT"]
