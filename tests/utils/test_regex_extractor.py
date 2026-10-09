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


class TestFindSQLQuotedTerminators:
    """A ';' only ends a query when it is outside quotes."""

    def test_semicolon_inside_string_literal(self) -> None:
        """Regression test: a ';' in a literal must not truncate the query."""
        output = """SELECT a FROM "Table" WHERE b = 'x;y';"""
        result = find_sql(output)
        assert result == ["""SELECT a FROM "Table" WHERE b = 'x;y'"""]

    def test_semicolon_inside_doubled_quote_escape(self) -> None:
        """'' is an escaped quote, so the ';' after it is still inside the literal."""
        output = "SELECT a FROM t WHERE b = 'it''s; here';"
        result = find_sql(output)
        assert result == ["SELECT a FROM t WHERE b = 'it''s; here'"]

    def test_semicolon_inside_quoted_identifier(self) -> None:
        output = 'SELECT "we;ird" FROM t;'
        result = find_sql(output)
        assert result == ['SELECT "we;ird" FROM t']

    def test_semicolon_inside_backtick_identifier(self) -> None:
        output = "SELECT `we;ird` FROM t;"
        result = find_sql(output)
        assert result == ["SELECT `we;ird` FROM t"]

    def test_semicolon_after_the_literal_still_terminates(self) -> None:
        """The terminating ';' must still work once the literal is closed."""
        output = "SELECT a FROM t WHERE b = 'x'; SELECT c FROM t;"
        result = find_sql(output)
        assert len(result) == 2
        assert result[0] == "SELECT a FROM t WHERE b = 'x'"


class TestFindSQLCandidateOrder:
    """The final answer is at the end of the output, not the start."""

    _QUERY = 'SELECT "a" FROM "t" WHERE "b" = 1'

    def _reasoning_then_query(self, sentences: int = 12) -> str:
        prose = "\n".join(
            f"I need to select the right column for question {i}."
            for i in range(sentences)
        )
        return f"{prose}\n{self._QUERY};"

    def test_query_after_reasoning_prose_is_kept(self) -> None:
        """
        Regression test: prose that starts with 'select' used to fill every
        slot, so the real query at the end was never evaluated.
        """
        result = find_sql(self._reasoning_then_query())
        assert self._QUERY in result

    def test_last_candidates_survive_the_limit(self) -> None:
        result = find_sql(self._reasoning_then_query(), limit=1)
        assert result == [self._QUERY]

    def test_first_candidates_are_the_prose(self) -> None:
        """The prose candidates are still extracted, just no longer preferred."""
        result = find_sql(self._reasoning_then_query())
        assert len(result) == 10
        assert result[-1] == self._QUERY


class TestFindSQLFencedBlocks:
    """A ```sql block is the clearest statement of intent."""

    def test_sql_fence_is_preferred_over_prose(self) -> None:
        output = (
            "I need to select the right column first.\n"
            "```sql\nSELECT a FROM t;\n```\n"
        )
        result = find_sql(output)
        assert result == ["SELECT a FROM t"]

    def test_last_sql_fence_wins(self) -> None:
        output = "```sql\nSELECT 1;\n```\nsome prose\n```sql\nSELECT 2;\n```\n"
        result = find_sql(output, limit=1)
        assert result == ["SELECT 2"]

    def test_untagged_fence_falls_back_to_scanning(self) -> None:
        """A bare ``` block is not a sql fence, so scanning still applies."""
        output = "```\nSELECT * FROM data\n```"
        assert find_sql(output) == ["SELECT * FROM data"]

    def test_empty_sql_fence_falls_back_to_scanning(self) -> None:
        output = "```sql\n\n```\nSELECT a FROM t;"
        assert find_sql(output) == ["SELECT a FROM t"]
