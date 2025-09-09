import pytest

from sgr.sql_guard import SQLGuard, GuardConfig, SQLGuardError

guard = SQLGuard(
    GuardConfig(
        allowed_schemas={"public"},
        allowed_tables={"users"},
        allowed_columns={"id", "name", "created_at"},
    )
)


def test_select_ok():
    sql = "SELECT id, name FROM public.users WHERE created_at >= '2025-01-01'"
    assert guard.validate(sql) is True


def test_block_comments():
    sql = "SELECT id FROM public.users -- comment"
    with pytest.raises(SQLGuardError):
        guard.validate(sql)


def test_block_multi_stmt():
    sql = "SELECT id FROM public.users; SELECT 1"
    with pytest.raises(SQLGuardError):
        guard.validate(sql)


def test_block_table():
    sql = "SELECT * FROM secret.admins"
    with pytest.raises(SQLGuardError):
        guard.validate(sql)
