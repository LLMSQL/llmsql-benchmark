"""Tests for llmsql.utils.evaluation_utils module."""

import sqlite3

import pytest

from llmsql.utils.evaluation_utils import (
    evaluate_sample,
    execute_sql,
    fix_table_name,
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
        assert 'FROM Table' not in result
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
        assert result == 'SELECT * FROM "my_table" WHERE table_name = \'other_table\''

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

    def test_invalid_question_id_type(self, eval_db, questions_dict) -> None:
        """Test assertion when question_id is not int."""
        item = {
            "question_id": "1",  # String instead of int
            "completion": "SELECT 1",
        }
        with pytest.raises(AssertionError):
            evaluate_sample(item, questions_dict, eval_db)

    def test_invalid_completion_type(self, eval_db, questions_dict) -> None:
        """Test assertion when completion is not string."""
        item = {
            "question_id": 1,
            "completion": ["SELECT 1"],  # List instead of string
        }
        with pytest.raises(AssertionError):
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
        assert isinstance(metrics["pred_none"], int)
        assert isinstance(metrics["gold_none"], int)
        assert isinstance(metrics["sql_error"], int)

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
