"""Tests for pg_compat SQL translation layer."""
from __future__ import annotations

import pytest

from src.db.pg_compat import (
    translate_sql,
    _replace_placeholders,
    _replace_datetime,
    _translate_insert_or_replace,
    _safe_ident,
)


class TestSafeIdent:
    def test_accepts_valid_identifiers(self):
        assert _safe_ident("users") == "users"
        assert _safe_ident("my_table") == "my_table"
        assert _safe_ident("_private") == "_private"
        assert _safe_ident("a1") == "a1"

    def test_rejects_invalid_identifiers(self):
        with pytest.raises(ValueError, match="Unsafe SQL identifier"):
            _safe_ident("users; drop table")
        with pytest.raises(ValueError, match="Unsafe SQL identifier"):
            _safe_ident("user name")
        with pytest.raises(ValueError, match="Unsafe SQL identifier"):
            _safe_ident("table--")


class TestReplacePlaceholders:
    def test_replace_simple(self):
        sql = "SELECT * FROM users WHERE id = ?"
        assert _replace_placeholders(sql) == "SELECT * FROM users WHERE id = $1"

    def test_replace_multiple(self):
        sql = "INSERT INTO t VALUES (?, ?, ?)"
        assert _replace_placeholders(sql) == "INSERT INTO t VALUES ($1, $2, $3)"

    def test_no_placeholders(self):
        sql = "SELECT 1"
        assert _replace_placeholders(sql) == "SELECT 1"

    def test_empty_string(self):
        assert _replace_placeholders("") == ""


class TestReplaceDatetime:
    def test_datetime_now(self):
        sql = "SELECT datetime('now')"
        assert _replace_datetime(sql) == "SELECT NOW()"

    def test_date_now(self):
        sql = "SELECT date('now')"
        assert _replace_datetime(sql) == "SELECT CURRENT_DATE"

    def test_case_insensitive(self):
        sql = "SELECT DATETIME('now')"
        assert _replace_datetime(sql) == "SELECT NOW()"

    def test_no_match(self):
        sql = "SELECT 1"
        assert _replace_datetime(sql) == "SELECT 1"


class TestTranslateInsertOrReplace:
    def test_basic_translation(self):
        sql = "INSERT OR REPLACE INTO chat_summaries (user_id, summary) VALUES (?, ?)"
        result = _translate_insert_or_replace(sql)
        assert "ON CONFLICT (user_id) DO UPDATE SET" in result
        assert "summary = EXCLUDED.summary" in result
        assert result.count("$") == 0  # placeholders not replaced yet

    def test_unknown_table_falls_back_to_do_nothing(self):
        sql = "INSERT OR REPLACE INTO unknown_table (id, val) VALUES (?, ?)"
        result = _translate_insert_or_replace(sql)
        assert "ON CONFLICT DO NOTHING" in result

    def test_all_conflict_columns_no_update(self):
        sql = "INSERT OR REPLACE INTO chat_summaries (user_id) VALUES (?)"
        result = _translate_insert_or_replace(sql)
        assert "ON CONFLICT (user_id) DO NOTHING" in result

    def test_not_insert_or_replace(self):
        sql = "SELECT * FROM users"
        assert _translate_insert_or_replace(sql) == sql


class TestTranslateSql:
    def test_full_pipeline(self):
        sql = "INSERT OR REPLACE INTO chat_summaries (user_id, summary) VALUES (?, ?)"
        result = translate_sql(sql)
        assert "ON CONFLICT (user_id) DO UPDATE SET" in result
        assert "summary = EXCLUDED.summary" in result
        assert "$1" in result and "$2" in result

    def test_select_with_placeholders(self):
        sql = "SELECT * FROM users WHERE id = ? AND name = ?"
        result = translate_sql(sql)
        assert result == "SELECT * FROM users WHERE id = $1 AND name = $2"

    def test_round_trip_datetime(self):
        sql = "UPDATE t SET ts = datetime('now') WHERE id = ?"
        result = translate_sql(sql)
        assert "NOW()" in result
        assert "$1" in result
