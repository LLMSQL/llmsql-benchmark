"""Tests for the LLMSQL 2.0 SQL extraction and lenient execution match."""

import pytest

from llmsql.utils.matching import extract_sql, norm, results_match

# ---------------------------------------------------------------------------
# extract_sql
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "text,expected",
    [
        ("```sql\nSELECT 1\n```", "SELECT 1;"),
        # the LAST block wins (models draft queries while reasoning)
        ("```sql\nSELECT 1;\n```\nbetter:\n```sql\nSELECT 2;\n```", "SELECT 2;"),
        ("```sqlite\nselect a from t;\n```", "select a from t;"),
        ("```\nSELECT a FROM t\n```", "SELECT a FROM t;"),
        ("```SQL\nSELECT a FROM t\n```", "SELECT a FROM t;"),
        # prose before the query inside the block is skipped
        ("```sql\n-- the answer\nSELECT a FROM t;\n```", "SELECT a FROM t;"),
        # CTEs are kept whole
        (
            "```sql\nWITH x AS (SELECT 1 AS v) SELECT v FROM x;\n```",
            "WITH x AS (SELECT 1 AS v) SELECT v FROM x;",
        ),
        # repeated trailing semicolons are normalised to one
        ("```sql\nSELECT 1;;\n```", "SELECT 1;"),
        # no fence: first WITH/SELECT to the end of the text
        ("The query is SELECT a FROM t", "SELECT a FROM t;"),
        ("no sql here", None),
        ("", None),
        (None, None),
        # a last block without SQL gives no query
        ("```sql\nSELECT 1\n```\n```text\nok\n```", None),
    ],
)
def test_extract_sql(text, expected):
    assert extract_sql(text) == expected


# ---------------------------------------------------------------------------
# norm / results_match
# ---------------------------------------------------------------------------


def test_norm_none():
    assert norm(None) is None


def test_norm_rounds_numbers_to_two_decimals():
    assert norm([(3.14159,), ("42",)]) == sorted([(3.14,), (42.0,)], key=str)


@pytest.mark.parametrize(
    "gold,pred",
    [
        # identical / order / numeric types
        ([["a"], ["b"]], [("b",), ("a",)]),
        ([[42]], [("42",)]),
        ([[42]], [(42.0,)]),
        ([[3.14159]], [(3.14,)]),
        # thousands separators
        ([[66714]], [("66,714",)]),
        ([[1234567]], [("1 234 567",)]),
        ([["66,714"]], [(66714,)]),
        # currency
        ([[265396]], [("$265,396",)]),
        ([[1.5]], [("£ 1.5",)]),
        ([[100]], [("€100",)]),
        # units / magnitude words
        ([[25.7]], [("25.7 million",)]),
        ([[0.4]], [("0.4%",)]),
        ([[120]], [("120 km",)]),
        # number in parentheses
        ([[0]], [("(0)",)]),
        # trailing count in parentheses
        ([["Al Horford"]], [("Al Horford (15)",)]),
        ([["Al Horford (15)"]], [("Al Horford",)]),
        # duplicate rows
        ([["27 April"]], [("27 April",), ("27 April",)]),
        # extra columns: one predicted column equals the single gold column
        ([["Smith"]], [("Smith", 12)]),
        ([[12]], [("Smith", 12)]),
        # two-part answer given in two columns
        ([["KeyArena 10,891"]], [("KeyArena", "10,891")]),
        # whitespace around strings
        ([["x"]], [("  x ",)]),
        # empty results
        ([], []),
    ],
)
def test_results_match_accepts(gold, pred):
    assert results_match(gold, pred)


@pytest.mark.parametrize(
    "gold,pred",
    [
        ([["a"]], None),  # failed query
        ([["a"]], [("b",)]),
        ([[3.14]], [(3.15,)]),
        ([["a"]], [("a",), ("b",)]),  # extra rows
        ([["a"], ["b"]], [("a",)]),  # missing rows
        ([["a", 1]], [("a",)]),  # missing column
        ([["Smith"]], [("Jones", 12)]),  # extra column, but wrong value
        ([["Smith"]], [("Smith", 1), ("Jones", 2)]),  # extra column and extra row
        ([["a"]], []),
        ([[1]], [(None,)]),
        ([["April"]], [("april",)]),  # strings are case-sensitive
    ],
)
def test_results_match_rejects(gold, pred):
    assert not results_match(gold, pred)


def test_results_match_none_gold():
    assert not results_match(None, [(1,)])
