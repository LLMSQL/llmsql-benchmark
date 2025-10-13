"""Tests for llmsql.utils.regex_extractor module."""

from llmsql.utils.regex_extractor import find_sql


class TestFindSQL:
    """Test cases for find_sql function."""

    def test_single_select_query(self) -> None:
        """Test extraction of a single SELECT query."""
        output = "SELECT * FROM users"
        result = find_sql(output)
        assert len(result) == 1
        assert result[0] == "SELECT * FROM users"

    def test_multiple_queries_with_semicolons(self) -> None:
        """Test extraction of multiple queries separated by semicolons."""
        output = "SELECT id FROM users; SELECT name FROM products;"
        result = find_sql(output)
        assert len(result) == 2
        assert "SELECT id FROM users" in result
        assert "SELECT name FROM products" in result

    def test_query_ending_with_semicolon(self) -> None:
        """Test query terminated by semicolon."""
        output = "SELECT * FROM table;"
        result = find_sql(output)
        assert len(result) == 1
        assert result[0] == "SELECT * FROM table"

    def test_query_ending_with_newlines(self) -> None:
        """Test query terminated by 4+ newlines."""
        output = "SELECT col1, col2 FROM table\n\n\n\nSome other text"
        result = find_sql(output)
        assert len(result) == 1
        assert result[0] == "SELECT col1, col2 FROM table"

    def test_query_ending_with_markdown_fence(self) -> None:
        """Test query inside markdown code block."""
        output = "```\nSELECT * FROM data\n```"
        result = find_sql(output)
        assert len(result) == 1
        assert result[0] == "SELECT * FROM data"

    def test_query_ending_at_eof(self) -> None:
        """Test query ending at end of string."""
        output = "Here is the query: SELECT * FROM users"
        result = find_sql(output)
        assert len(result) == 1
        assert result[0] == "SELECT * FROM users"

    def test_case_insensitive_select(self) -> None:
        """Test that SELECT keyword matching is case-insensitive."""
        outputs = [
            "select * from users",
            "Select * from users",
            "SELECT * from users",
            "SeLeCt * from users",
        ]
        for output in outputs:
            result = find_sql(output)
            assert len(result) == 1
            assert "from users" in result[0].lower()

    def test_limit_parameter(self) -> None:
        """Test that limit parameter restricts number of results."""
        output = "SELECT 1; SELECT 2; SELECT 3; SELECT 4; SELECT 5;"
        result = find_sql(output, limit=3)
        assert len(result) == 3

    def test_duplicate_queries_deduplicated(self) -> None:
        """Test that duplicate queries are removed."""
        output = "SELECT * FROM users; SELECT * FROM users;"
        result = find_sql(output)
        assert len(result) == 1
        assert result[0] == "SELECT * FROM users"

    def test_empty_string(self) -> None:
        """Test extraction from empty string."""
        result = find_sql("")
        assert result == []

    def test_no_sql_found(self) -> None:
        """Test when no SQL queries are present."""
        output = "This is just some text without any queries."
        result = find_sql(output)
        assert result == []

    def test_nested_select(self) -> None:
        """Test query with nested SELECT statement."""
        output = "SELECT * FROM (SELECT id FROM users WHERE active = 1)"
        result = find_sql(output)
        # Note: The regex extracts both outer and inner SELECT as separate queries
        assert len(result) == 2
        assert any("SELECT * FROM" in r for r in result)
        assert any("SELECT id FROM users" in r for r in result)

    def test_select_in_natural_language(self) -> None:
        """Test SELECT appearing in middle of natural language text."""
        output = 'The answer is: SELECT "name" FROM "table"'
        result = find_sql(output)
        assert len(result) == 1
        assert 'SELECT "name" FROM "table"' in result[0]

    def test_multiple_extraction_methods_same_output(self) -> None:
        """Test multiple queries with different terminators."""
        output = "SELECT 1;\n\nSELECT 2\n\n\n\nSELECT 3```"
        result = find_sql(output)
        assert len(result) == 3

    def test_query_with_quotes(self) -> None:
        """Test query with quoted identifiers and strings."""
        output = """SELECT "Competition or tour" FROM "2-17637370-13" WHERE "Opponent" = 'Nordsjælland\'"""
        result = find_sql(output)
        assert len(result) == 1
        assert "Competition or tour" in result[0]

    def test_complex_example_from_docstring(self) -> None:
        """Test the example from the module's main block."""
        output = 'select "Competition or tour". There"s no aggregation required, just one value possibly multiple rows. Let"s useSELECT "Competition or tour" FROM "2-17637370-13" WHERE "Opponent" = "Nordsjælland" AND "Ground" = "HR"\n'
        result = find_sql(output)
        # Should extract both the lowercase 'select' and uppercase 'SELECT'
        assert len(result) == 2

    def test_whitespace_handling(self) -> None:
        """Test that leading/trailing whitespace is stripped."""
        output = "   SELECT * FROM users   ;"
        result = find_sql(output)
        assert len(result) == 1
        # Should be stripped
        assert result[0] == "SELECT * FROM users"

    def test_multiline_query(self) -> None:
        """Test query spanning multiple lines."""
        output = """SELECT id, name, email
FROM users
WHERE active = 1
ORDER BY name;"""
        result = find_sql(output)
        assert len(result) == 1
        assert "SELECT id, name, email" in result[0]
        assert "FROM users" in result[0]
        assert "WHERE active = 1" in result[0]

    def test_default_limit_value(self) -> None:
        """Test that default limit is 10."""
        # Generate 15 queries
        output = "; ".join([f"SELECT {i}" for i in range(15)]) + ";"
        result = find_sql(output)
        # Should only return 10 (default limit)
        assert len(result) == 10

    def test_limit_zero(self) -> None:
        """Test limit=0 returns empty list."""
        output = "SELECT * FROM users"
        result = find_sql(output, limit=0)
        assert result == []
