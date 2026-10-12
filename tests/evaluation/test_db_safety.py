"""Evaluation runs model-generated SQL against a shared DB.

The statements come out of the model output, so they must not be able to change
the database (see issue #96): one destructive candidate would otherwise corrupt
the cached ``sqlite_tables.db`` for every later run. A pathological query must
not be able to hang the run either.
"""

import sqlite3
import time

import pytest

from llmsql.evaluation.evaluate import evaluate
from llmsql.utils.evaluation_utils import (
    DEFAULT_QUERY_TIMEOUT,
    connect_sqlite,
    evaluate_sample,
    execute_sql,
)
from llmsql.utils.regex_extractor import find_sql

# A single statement that never terminates on its own (unbounded recursive CTE).
# It has to start with SELECT so the extractor picks it up like a real
# prediction would.
RUNAWAY_SQL = (
    "SELECT (WITH RECURSIVE c(n) AS "
    "(SELECT 1 UNION ALL SELECT n+1 FROM c) SELECT count(*) FROM c)"
)


@pytest.fixture
def db_file(tmp_path):
    """A small SQLite DB with one table, copied next to the evaluation files."""
    db_path = tmp_path / "sqlite_tables.db"
    conn = sqlite3.connect(db_path)
    conn.execute("CREATE TABLE test (id INTEGER PRIMARY KEY, name TEXT)")
    conn.executemany("INSERT INTO test (name) VALUES (?)", [("Alice",), ("Bob",)])
    conn.commit()
    conn.close()
    return str(db_path)


class TestReadOnlyConnection:
    """connect_sqlite must not hand out a writable connection"""

    @pytest.mark.parametrize(
        "statement",
        [
            "DROP TABLE test",
            "DELETE FROM test",
            "UPDATE test SET name = 'x'",
            "INSERT INTO test (name) VALUES ('Carol')",
            "ALTER TABLE test ADD COLUMN extra TEXT",
        ],
    )
    def test_write_attempts_fail_and_leave_db_intact(self, db_file, statement):
        conn = connect_sqlite(db_file)
        try:
            assert execute_sql(conn, statement) is None
        finally:
            conn.close()

        # The data has to survive: the DB is cached in workdir_path and reused
        # by every later evaluation.
        check = sqlite3.connect(db_file)
        try:
            assert check.execute("SELECT name FROM test ORDER BY id").fetchall() == [
                ("Alice",),
                ("Bob",),
            ]
        finally:
            check.close()

    def test_reads_still_work(self, db_file):
        conn = connect_sqlite(db_file)
        try:
            assert execute_sql(conn, "SELECT name FROM test") == [
                ("Alice",),
                ("Bob",),
            ]
        finally:
            conn.close()

    def test_missing_db_still_raises(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            connect_sqlite(str(tmp_path / "nope.db"))


class TestQueryTimeout:
    """A runaway statement must be cut off instead of hanging the run"""

    def test_runaway_query_is_aborted_within_budget(self, db_file):
        conn = connect_sqlite(db_file)
        try:
            start = time.monotonic()
            assert execute_sql(conn, RUNAWAY_SQL, query_timeout=1.0) is None
            elapsed = time.monotonic() - start
        finally:
            conn.close()

        # Some slack for slow CI machines, but nowhere near an unbounded run.
        assert elapsed < 10.0

    def test_connection_is_usable_after_an_abort(self, db_file):
        """Aborting one statement must not poison the connection"""
        conn = connect_sqlite(db_file)
        try:
            assert execute_sql(conn, RUNAWAY_SQL, query_timeout=0.5) is None
            assert execute_sql(conn, "SELECT name FROM test") == [
                ("Alice",),
                ("Bob",),
            ]
        finally:
            conn.close()

    def test_fast_queries_are_not_affected(self, db_file):
        conn = connect_sqlite(db_file)
        try:
            assert execute_sql(conn, "SELECT name FROM test", query_timeout=0.05) == [
                ("Alice",),
                ("Bob",),
            ]
        finally:
            conn.close()

    def test_timeout_is_counted_as_sql_error(self, db_file):
        """
        An overrunning prediction is a SQL error, not a silent wrong answer
        """
        questions = {
            1: {
                "table_id": "test",
                "sql": "SELECT name FROM test",
                "question": "List names",
            }
        }
        item = {"question_id": 1, "completion": RUNAWAY_SQL}

        conn = connect_sqlite(db_file)
        try:
            is_match, _, metrics = evaluate_sample(
                item, questions, conn, query_timeout=1.0
            )
        finally:
            conn.close()

        # The extractor yields one candidate per SELECT keyword, and every one
        # of them overruns the budget, so each is counted as a SQL error.
        assert is_match == 0
        assert metrics["sql_error"] == len(find_sql(RUNAWAY_SQL))

    def test_default_budget_is_applied(self):
        """The default is a real budget, not an unbounded run"""
        assert DEFAULT_QUERY_TIMEOUT > 0


class StubConnection:
    """Records progress-handler calls instead of running SQL"""

    def __init__(self, fail=False):
        self.installed = 0
        self.cleared = 0
        self.fail = fail

    def set_progress_handler(self, handler, n):
        if handler is None:
            self.cleared += 1
        else:
            self.installed += 1

    def cursor(self):
        class _Cursor:
            def __init__(self, fail):
                self.fail = fail

            def execute(self, sql):
                if self.fail:
                    raise sqlite3.OperationalError("boom")

            def fetchall(self):
                return [(1,)]

        return _Cursor(self.fail)


class TestProgressHandlerLifecycle:
    """The handler lives on the connection, so it must be installed and removed"""

    def test_installed_and_cleared_on_success(self):
        conn = StubConnection()
        assert execute_sql(conn, "SELECT 1", query_timeout=1.0) == [(1,)]
        assert (conn.installed, conn.cleared) == (1, 1)

    def test_cleared_after_a_failed_statement(self):
        conn = StubConnection(fail=True)
        assert execute_sql(conn, "SELECT 1", query_timeout=1.0) is None
        assert (conn.installed, conn.cleared) == (1, 1)

    def test_not_installed_when_budget_is_disabled(self):
        conn = StubConnection()
        assert execute_sql(conn, "SELECT 1", query_timeout=0) == [(1,)]
        assert (conn.installed, conn.cleared) == (0, 0)


class TestEvaluatePassesTimeout:
    """evaluate() has to forward the budget down to evaluate_sample"""

    def test_query_timeout_reaches_evaluate_sample(self, monkeypatch, tmp_path):
        # evaluate() only needs the file to exist here: evaluate_sample is
        # replaced below, so no query is ever run against it.
        db_path = tmp_path / "sqlite_tables.db"
        sqlite3.connect(db_path).close()

        (tmp_path / "questions.jsonl").write_text(
            '{"question_id": 1, "table_id": "test", "question": "Q", '
            '"sql": "SELECT 1"}'
        )
        (tmp_path / "outputs.jsonl").write_text(
            '{"question_id": 1, "completion": "SELECT 1"}'
        )

        captured = {}

        def fake_evaluate_sample(item, questions, conn, **kwargs):
            captured.update(kwargs)
            return (
                1,
                None,
                {
                    "pred_none": 0,
                    "gold_none": 0,
                    "sql_error": 0,
                    "exact_string_match": 1,
                },
            )

        monkeypatch.setattr(
            "llmsql.evaluation.evaluate.evaluate_sample", fake_evaluate_sample
        )
        monkeypatch.setattr(
            "llmsql.evaluation.evaluate.print_summary", lambda *a, **k: None
        )
        monkeypatch.setattr(
            "llmsql.evaluation.evaluate.log_mismatch", lambda **k: None
        )
        monkeypatch.setattr(
            "llmsql.evaluation.evaluate.save_json_report", lambda path, report: None
        )

        evaluate(
            outputs=str(tmp_path / "outputs.jsonl"),
            workdir_path=str(tmp_path),
            show_mismatches=False,
            query_timeout=1.5,
        )

        assert captured["query_timeout"] == 1.5
