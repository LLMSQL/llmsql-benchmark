"""Tests for llmsql.utils.evaluation_utils module."""

import sqlite3

import pytest

from llmsql.utils.evaluation_utils import (
    evaluate_sample,
    execute_sql,
    fix_table_name,
    normalize_question_id,
    normalize_sql,
    resolve_prediction_coverage,
)


@pytest.fixture
def test_db(tmp_path):
    """Create a test SQLite database."""
    db_path = tmp_path / "test.db"
    conn = sqlite3.connect(str(db_path))

    # Create test table
    conn.execute("""
        CREATE TABLE users (
            id INTEGER PRIMARY KEY,
            name TEXT,
            age INTEGER
        )
    """)
    conn.execute("INSERT INTO users VALUES (1, 'Alice', 30)")
    conn.execute("INSERT INTO users VALUES (2, 'Bob', 25)")
    conn.execute("INSERT INTO users VALUES (3, 'Charlie', 35)")
    conn.commit()

    yield conn

    conn.close()


class TestExecuteSQL:
    """Test cases for execute_sql function."""

    def test_successful_query(self, test_db: sqlite3.Connection) -> None:
        """Test successful query execution with results."""
        result = execute_sql(test_db, "SELECT name FROM users WHERE id = 1")
        assert result == [("Alice",)]

    def test_query_returns_multiple_rows(self, test_db: sqlite3.Connection) -> None:
        """Test query returning multiple rows."""
        result = execute_sql(test_db, "SELECT name FROM users ORDER BY name")
        assert result is not None
        assert len(result) == 3
        # Results should be sorted
        assert result == [("Alice",), ("Bob",), ("Charlie",)]

    def test_query_returns_empty_list(self, test_db: sqlite3.Connection) -> None:
        """Test query returning no results."""
        result = execute_sql(test_db, "SELECT * FROM users WHERE id = 999")
        assert result == []

    def test_query_returns_null(self, test_db: sqlite3.Connection) -> None:
        """Test query returning NULL value."""
        test_db.execute("INSERT INTO users VALUES (4, NULL, NULL)")
        test_db.commit()
        result = execute_sql(test_db, "SELECT name FROM users WHERE id = 4")
        assert result == [(None,)]

    def test_invalid_sql_syntax(self, test_db: sqlite3.Connection) -> None:
        """Test that invalid SQL returns None."""
        result = execute_sql(test_db, "SELECT * FORM users")  # typo: FORM
        assert result is None

    def test_sql_execution_exception(self, test_db: sqlite3.Connection) -> None:
        """Test that SQL errors return None."""
        result = execute_sql(test_db, "SELECT * FROM nonexistent_table")
        assert result is None

    def test_result_sorting(self, test_db: sqlite3.Connection) -> None:
        """Test that results are sorted."""
        # Insert in different order
        test_db.execute("DELETE FROM users")
        test_db.execute("INSERT INTO users VALUES (3, 'Charlie', 35)")
        test_db.execute("INSERT INTO users VALUES (1, 'Alice', 30)")
        test_db.execute("INSERT INTO users VALUES (2, 'Bob', 25)")
        test_db.commit()

        result = execute_sql(test_db, "SELECT id, name FROM users")
        assert result is not None
        # Should be sorted regardless of insertion order
        assert result == [(1, "Alice"), (2, "Bob"), (3, "Charlie")]


class TestNormalizeSql:
    def test_strips_semicolon_and_collapses_whitespace(self) -> None:
        assert normalize_sql("  SELECT a\n  FROM t ;  ") == "SELECT a FROM t"

    def test_preserves_case_and_literals(self) -> None:
        assert normalize_sql("SELECT a FROM t WHERE b = 'X  y'") == (
            "SELECT a FROM t WHERE b = 'X y'"
        )


class TestFixTableName:
    """Test cases for fix_table_name function."""

    def test_replace_single_quoted_table(self) -> None:
        """Test replacement of FROM 'Table' with actual table ID."""
        sql = "SELECT * FROM 'Table' WHERE id = 1"
        result = fix_table_name(sql, "actual_table_123")
        assert result == 'SELECT * FROM "actual_table_123" WHERE id = 1'

    def test_replace_double_quoted_table(self) -> None:
        """Test replacement of FROM \"Table\" with actual table ID."""
        sql = 'SELECT * FROM "Table" WHERE id = 1'
        result = fix_table_name(sql, "actual_table_123")
        assert result == 'SELECT * FROM "actual_table_123" WHERE id = 1'

    def test_replace_unquoted_table(self) -> None:
        """Test replacement of FROM Table with actual table ID."""
        sql = "SELECT * FROM Table WHERE id = 1"
        result = fix_table_name(sql, "actual_table_123")
        assert result == 'SELECT * FROM "actual_table_123" WHERE id = 1'

    def test_multiple_table_references(self) -> None:
        """Test SQL with multiple table references."""
        sql = "SELECT * FROM 'Table' JOIN 'Table' ON Table.id = Table.parent_id"
        result = fix_table_name(sql, "my_table")
        # Placeholder in FROM should be swapped out for the real table name
        assert "FROM Table" not in result
        assert "FROM 'Table'" not in result
        assert 'FROM "Table"' not in result
        assert 'FROM "my_table"' in result

    def test_table_id_with_special_characters(self) -> None:
        """Test table ID containing special characters."""
        sql = "SELECT * FROM Table"
        result = fix_table_name(sql, "table-with-dashes_123")
        assert result == 'SELECT * FROM "table-with-dashes_123"'

    def test_whitespace_handling(self) -> None:
        """Test that leading/trailing whitespace is stripped."""
        sql = "  SELECT * FROM Table  "
        result = fix_table_name(sql, "my_table")
        # Should be stripped
        assert result == 'SELECT * FROM "my_table"'

    def test_case_sensitive_table_keyword(self) -> None:
        """Test that only exact 'Table' placeholder is replaced."""
        sql = "SELECT * FROM Table WHERE table_name = 'other_table'"
        result = fix_table_name(sql, "my_table")
        # Should only replace the FROM Table, not 'other_table'
        assert result == "SELECT * FROM \"my_table\" WHERE table_name = 'other_table'"

    def test_complex_query(self) -> None:
        """Test complex query with joins and subqueries."""
        sql = """
        SELECT t1.id, t2.name
        FROM 'Table' t1
        JOIN "Table" t2 ON t1.id = t2.parent_id
        WHERE EXISTS (SELECT 1 FROM Table t3 WHERE t3.id = t1.id)
        """
        result = fix_table_name(sql, "users")
        assert '"users"' in result
        # All Table placeholders should be replaced
        assert "'Table'" not in result

    def test_lowercase_from_keyword(self) -> None:
        """Issue #125: lowercase 'from' keyword should match."""
        result = fix_table_name('select * from "Table" where 1', "t")
        assert 'FROM "t"' in result

    def test_lowercase_table_placeholder(self) -> None:
        """Issue #125: FROM 'table' (lowercase name) should match."""
        assert fix_table_name('SELECT * FROM "table" WHERE 1', "t") == 'SELECT * FROM "t" WHERE 1'

    def test_backtick_quoted_table(self) -> None:
        """Issue #125: FROM `Table` (backtick quotes) should match."""
        assert fix_table_name("SELECT * FROM `Table` WHERE 1", "t") == 'SELECT * FROM "t" WHERE 1'

    def test_extra_whitespace_between_from_and_table(self) -> None:
        """Issue #125: extra spaces between FROM and Table should match."""
        assert fix_table_name('SELECT * FROM  "Table" WHERE 1', "t") == 'SELECT * FROM "t" WHERE 1'

    def test_newline_between_from_and_table(self) -> None:
        """Issue #125: newline between FROM and Table should match."""
        assert fix_table_name('SELECT *\nFROM\n"Table"\nWHERE 1', "t") == 'SELECT *\nFROM "t"\nWHERE 1'

    def test_column_named_table_not_replaced(self) -> None:
        """Issue #125: a column or value containing 'Table' should not be touched."""
        sql = "SELECT * FROM \"Table\" WHERE name = 'Table'"
        result = fix_table_name(sql, "t")
        assert "name = 'Table'" in result


class TestEvaluateSample:
    """Test cases for evaluate_sample function."""

    @pytest.fixture
    def questions_dict(self):
        """Sample questions dictionary."""
        return {
            1: {
                "table_id": "users",
                "sql": "SELECT name FROM users WHERE id = 1",
                "question": "What is the name of user 1?",
            },
            2: {
                "table_id": "users",
                "sql": "SELECT COUNT(*) FROM users",
                "question": "How many users are there?",
            },
            3: {
                "table_id": "users",
                "sql": "SELECT age FROM users WHERE name = 'NonExistent'",
                "question": "What is the age of NonExistent?",
            },
        }

    @pytest.fixture
    def eval_db(self, tmp_path):
        """Create evaluation test database."""
        db_path = tmp_path / "eval.db"
        conn = sqlite3.connect(str(db_path))
        conn.execute("CREATE TABLE users (id INTEGER, name TEXT, age INTEGER)")
        conn.execute("INSERT INTO users VALUES (1, 'Alice', 30)")
        conn.execute("INSERT INTO users VALUES (2, 'Bob', 25)")
        conn.commit()
        yield conn
        conn.close()

    def test_matching_prediction(self, eval_db, questions_dict) -> None:
        """Test when prediction matches gold SQL."""
        item = {
            "question_id": 1,
            "completion": "SELECT name FROM Table WHERE id = 1",
        }
        is_match, mismatch_info, metrics = evaluate_sample(
            item, questions_dict, eval_db
        )
        assert is_match == 1
        assert mismatch_info is None
        assert metrics["pred_none"] == 0
        assert metrics["gold_none"] == 0
        assert metrics["sql_error"] == 0
        assert metrics["exact_string_match"] == 0

    def test_non_matching_prediction(self, eval_db, questions_dict) -> None:
        """Test when prediction does not match gold SQL."""
        item = {
            "question_id": 1,
            "completion": "SELECT name FROM Table WHERE id = 2",  # Wrong id
        }
        is_match, mismatch_info, metrics = evaluate_sample(
            item, questions_dict, eval_db
        )
        assert is_match == 0
        assert mismatch_info is not None
        assert mismatch_info["question_id"] == 1
        assert "question" in mismatch_info
        assert "gold_sql" in mismatch_info
        assert "model_output" in mismatch_info

    def test_gold_query_returns_empty(self, eval_db, questions_dict) -> None:
        """Test when gold query returns empty result."""
        item = {
            "question_id": 3,
            "completion": "SELECT age FROM Table WHERE name = 'NonExistent'",
        }
        is_match, mismatch_info, metrics = evaluate_sample(
            item, questions_dict, eval_db
        )
        # Both return empty, should match
        assert is_match == 1

    def test_predicted_query_sql_error(self, eval_db, questions_dict) -> None:
        """Test when predicted query has SQL error."""
        item = {
            "question_id": 1,
            "completion": "SELECT INVALID SYNTAX",
        }
        is_match, mismatch_info, metrics = evaluate_sample(
            item, questions_dict, eval_db
        )
        assert is_match == 0
        assert metrics["sql_error"] >= 1

    def test_multiple_predictions_one_matches(self, eval_db, questions_dict) -> None:
        """Test when multiple predictions exist and one matches."""
        item = {
            "question_id": 1,
            "completion": "SELECT name FROM Table WHERE id = 999; SELECT name FROM Table WHERE id = 1",
        }
        is_match, mismatch_info, metrics = evaluate_sample(
            item, questions_dict, eval_db
        )
        # Second query should match
        assert is_match == 1

    def test_multiple_predictions_none_match(self, eval_db, questions_dict) -> None:
        """Test when multiple predictions exist but none match."""
        item = {
            "question_id": 1,
            "completion": "SELECT name FROM Table WHERE id = 999; SELECT name FROM Table WHERE id = 888",
        }
        is_match, mismatch_info, metrics = evaluate_sample(
            item, questions_dict, eval_db
        )
        assert is_match == 0

    def test_int_like_string_question_id_is_accepted(
        self, eval_db, questions_dict
    ) -> None:
        """Other tools serialise ids as strings; int-like strings must work."""
        item = {
            "question_id": "1",  # int-like string instead of int
            "completion": "SELECT name FROM Table WHERE id = 1",
        }
        is_match, _, _ = evaluate_sample(item, questions_dict, eval_db)
        assert is_match == 1

    def test_invalid_question_id_type(self, eval_db, questions_dict) -> None:
        """A non-int-like question_id is reported by name, not by a bare assert."""
        item = {
            "question_id": "not-an-id",
            "completion": "SELECT 1",
        }
        with pytest.raises(ValueError, match="not-an-id"):
            evaluate_sample(item, questions_dict, eval_db)

    def test_unknown_question_id_is_reported(self, eval_db, questions_dict) -> None:
        """An id outside the benchmark raises ValueError instead of KeyError."""
        item = {
            "question_id": 99999,
            "completion": "SELECT 1",
        }
        with pytest.raises(ValueError, match="99999"):
            evaluate_sample(item, questions_dict, eval_db)

    def test_invalid_completion_type(self, eval_db, questions_dict) -> None:
        """Test error when completion is not string."""
        item = {
            "question_id": 1,
            "completion": ["SELECT 1"],  # List instead of string
        }
        with pytest.raises(TypeError, match="completion"):
            evaluate_sample(item, questions_dict, eval_db)

    def test_metrics_counters(self, eval_db, questions_dict) -> None:
        """Test that metrics counters are accurate."""
        item = {
            "question_id": 2,
            "completion": "SELECT COUNT(*) FROM Table",
        }
        is_match, mismatch_info, metrics = evaluate_sample(
            item, questions_dict, eval_db
        )
        assert is_match == 1
        # Verify metrics structure
        assert "pred_none" in metrics
        assert "gold_none" in metrics
        assert "sql_error" in metrics
        assert "exact_string_match" in metrics
        assert isinstance(metrics["pred_none"], int)
        assert isinstance(metrics["gold_none"], int)
        assert isinstance(metrics["sql_error"], int)
        assert isinstance(metrics["exact_string_match"], int)

    def test_mismatch_info_structure(self, eval_db, questions_dict) -> None:
        """Test structure of mismatch_info when prediction fails."""
        item = {
            "question_id": 1,
            "completion": "SELECT name FROM Table WHERE id = 999",
        }
        is_match, mismatch_info, metrics = evaluate_sample(
            item, questions_dict, eval_db
        )
        assert is_match == 0
        assert mismatch_info is not None
        # Check all required fields
        assert "question_id" in mismatch_info
        assert "question" in mismatch_info
        assert "gold_sql" in mismatch_info
        assert "model_output" in mismatch_info
        assert "gold_results" in mismatch_info
        assert "prediction_results" in mismatch_info

    def test_exact_string_match_ignores_semicolon_and_whitespace(
        self, eval_db
    ) -> None:
        """Gold SQL ends with ';' but extracted predictions never do."""
        questions = {
            1: {
                "table_id": "users",
                "sql": 'SELECT "name" FROM "users" WHERE "id" = 1;',
                "question": "What is the name of user 1?",
            }
        }
        item = {
            "question_id": 1,
            "completion": 'SELECT  "name"\nFROM "Table" WHERE "id" = 1;',
        }

        is_match, _, metrics = evaluate_sample(item, questions, eval_db)

        assert is_match == 1
        assert metrics["exact_string_match"] == 1

    def test_exact_string_match_zero_when_sql_differs(self, eval_db) -> None:
        """Execution match does not imply exact string match."""
        questions = {
            1: {
                "table_id": "users",
                "sql": 'SELECT "name" FROM "users" WHERE "id" = 1;',
                "question": "What is the name of user 1?",
            }
        }
        item = {
            "question_id": 1,
            "completion": "SELECT name FROM Table WHERE age = 30",
        }

        is_match, _, metrics = evaluate_sample(item, questions, eval_db)

        assert is_match == 1
        assert metrics["exact_string_match"] == 0

    def test_null_results_metrics(self, eval_db) -> None:
        """Test metrics when both gold and prediction return NULL results."""
        questions = {
            1: {
                "table_id": "users",
                "sql": "SELECT NULL",
                "question": "Returns NULL",
            }
        }
        item = {
            "question_id": 1,
            "completion": "SELECT NULL",
        }

        is_match, mismatch_info, metrics = evaluate_sample(item, questions, eval_db)

        assert is_match == 1
        assert mismatch_info is None
        assert metrics["gold_none"] == 1
        assert metrics["pred_none"] == 1
        assert metrics["sql_error"] == 0
        assert metrics["exact_string_match"] == 1


class TestNormalizeQuestionId:
    """question_id coercion"""

    def test_int_passes_through(self):
        assert normalize_question_id(42) == 42

    def test_int_like_string_is_accepted(self):
        assert normalize_question_id("42") == 42
        assert normalize_question_id(" 42 ") == 42

    def test_non_int_string_is_rejected(self):
        with pytest.raises(ValueError):
            normalize_question_id("forty-two")

    def test_unsupported_type_is_rejected(self):
        with pytest.raises(TypeError):
            normalize_question_id(None)
        with pytest.raises(TypeError):
            normalize_question_id([1])


class TestResolvePredictionCoverage:
    """Coverage accounting for partial / duplicated / malformed predictions"""

    @staticmethod
    def _questions(*ids):
        return {
            qid: {"table_id": qid, "sql": "SELECT 1", "question": "q"} for qid in ids
        }

    @staticmethod
    def _item(qid):
        return {"question_id": qid, "completion": "SELECT 1"}

    def test_full_coverage(self):
        questions = self._questions(1, 2, 3)
        outputs = [self._item(1), self._item(2), self._item(3)]

        unique, coverage = resolve_prediction_coverage(outputs, questions)

        assert len(unique) == 3
        assert coverage == {
            "expected": 3,
            "answered": 3,
            "missing": 0,
            "duplicates": 0,
        }

    def test_partial_run_is_reported_as_missing(self):
        """Regression: 1 of 3 answered used to yield a full-looking accuracy."""
        questions = self._questions(1, 2, 3)

        unique, coverage = resolve_prediction_coverage([self._item(1)], questions)

        assert len(unique) == 1
        assert coverage["expected"] == 3
        assert coverage["answered"] == 1
        assert coverage["missing"] == 2

    def test_duplicates_keep_first_prediction(self):
        questions = self._questions(1, 2)
        outputs = [self._item(1), self._item(1), self._item(2), self._item(1)]

        unique, coverage = resolve_prediction_coverage(outputs, questions)

        assert [item["question_id"] for item in unique] == [1, 2]
        assert coverage["duplicates"] == 2

    def test_string_ids_are_matched_against_the_benchmark(self):
        questions = self._questions(1, 2)

        unique, coverage = resolve_prediction_coverage(
            [self._item("1"), self._item("2")], questions
        )

        assert [item["question_id"] for item in unique] == ["1", "2"]
        assert coverage["missing"] == 0

    def test_unknown_id_raises_with_the_id_in_the_message(self):
        questions = self._questions(1)

        with pytest.raises(ValueError, match="999"):
            resolve_prediction_coverage([self._item(999)], questions)

    def test_empty_predictions(self):
        questions = self._questions(1, 2)

        unique, coverage = resolve_prediction_coverage([], questions)

        assert unique == []
        assert coverage == {
            "expected": 2,
            "answered": 0,
            "missing": 2,
            "duplicates": 0,
        }
