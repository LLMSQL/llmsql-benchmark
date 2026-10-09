"""
LLMSQL 2.0 SQL extraction and execution matching
================================================

This module implements the official LLMSQL 2.0 evaluation protocol:

1. :func:`extract_sql` takes the SQL from the last ```sql block of a
   completion (falling back to the first ``WITH``/``SELECT`` statement).
2. The query is executed on ``sqlite_tables.db``.
3. :func:`results_match` compares the result rows with the reference
   ``answer`` of the question, tolerating differences that do not change the
   answer (row order, duplicates, number formatting, extra columns, ...).

The behaviour is a faithful port of the reference scorer used for the
published LLMSQL 2.0 results; changing it changes the reported numbers.
LLMSQL 1.0 does not use this module (it compares sorted result rows exactly).
"""

from __future__ import annotations

from collections.abc import Sequence
import math
import re
from typing import Any

Rows = Sequence[Sequence[Any]]

# A number followed by a unit / magnitude word: '25.7 million', '0.4%', '120 km'.
_NUMBER_WITH_UNIT = re.compile(
    r"(-?[\d,]*\d(?:\.\d+)?)\s*(?:million|billion|thousand|m|bn|k|%|km|kg|mph|km/h"
    r"|cm|mm|metres|meters|ft|lbs?|mhz|kw|s)",
    re.I,
)
# A number prefixed with a currency sign: '$265,396', '£ 1.5'.
_CURRENCY = re.compile(r"[$£€]\s?-?[\d,]*\d(?:\.\d+)?")
# A number in parentheses: '(0)'.
_PARENTHESISED_NUMBER = re.compile(r"\((-?\d+(?:\.\d+)?)\)")
# A number with thousands separators (comma or space): '66,714', '1 234 567'.
_THOUSANDS = re.compile(r"-?\d{1,3}([, ]\d{3})+(\.\d+)?")
# A plain number: '42', '-3.5'.
_PLAIN_NUMBER = re.compile(r"-?\d+(\.\d+)?")
# A trailing count in parentheses: 'Al Horford (15)'.
_TRAILING_COUNT = re.compile(r"\s*\(\d+\)$")


def extract_sql(text: str | None) -> str | None:
    """Extract the SQL query from a model completion.

    The query is taken from the **last** fenced code block (```sql,
    ```sqlite or a bare ```), so a model may draft queries while reasoning
    and give its final answer at the end. Inside that block (or in the whole
    text if there is no block) the query starts at the first ``WITH`` or
    ``SELECT`` keyword (case-insensitive) and runs to the end of the block.
    Trailing semicolons are normalised to exactly one.

    Args:
        text: Model completion.

    Returns:
        The SQL query ending with ``;``, or ``None`` if no ``WITH``/``SELECT``
        statement is found.
    """
    if not text:
        return None
    blocks = re.findall(r"```(?:sql|sqlite)?\s*(.*?)```", text, re.S | re.I)
    candidate = blocks[-1] if blocks else text
    m = re.search(r"(WITH\b.*|SELECT\b.*)", candidate, re.S | re.I)
    return m.group(1).strip().rstrip(";").strip() + ";" if m else None


def _norm_value(v: Any) -> Any:
    """Normalise one cell for comparison (see :func:`norm`)."""
    if isinstance(v, str):
        s = v.strip()
        m = _NUMBER_WITH_UNIT.fullmatch(s)
        if _CURRENCY.fullmatch(s):
            # Currency: '$265,396' == 265396.
            v = s[1:].strip().replace(",", "")
        m2 = isinstance(v, str) and _PARENTHESISED_NUMBER.fullmatch(v.strip())
        if m2:
            # Number in parentheses: '(0)' == 0.
            v = m2.group(1)
        elif m:
            # Number with a unit word: '25.7 million' == 25.7, '0.4%' == 0.4.
            v = m.group(1)
        if isinstance(v, str) and _THOUSANDS.fullmatch(v.strip()):
            # Thousands separators: '66,714' == 66714.
            v = v.strip().replace(",", "").replace(" ", "")
    if isinstance(v, int | float) or (
        isinstance(v, str) and _PLAIN_NUMBER.fullmatch(v.strip())
    ):
        f = float(v)
        # Numbers are compared as floats rounded to 2 decimals; NaN -> None.
        return round(f, 2) if not math.isnan(f) else None
    return str(v).strip()


def norm(res: Rows | None) -> list[tuple[Any, ...]] | None:
    """Turn a result set into an order-insensitive, type-tolerant key.

    Every cell is normalised as follows:

    * **numbers** (``int``/``float`` or numeric strings such as ``"42"``) become
      floats rounded to 2 decimals, so ``42 == "42" == 42.0 == 42.001``;
    * **thousands separators** are removed: ``"66,714" == 66714``
      (also ``"1 234 567"``);
    * **currency signs** are removed: ``"$265,396" == 265396``;
    * **units / magnitude words** after a number are dropped:
      ``"25.7 million" == 25.7``, ``"0.4%" == 0.4``, ``"120 km" == 120``;
    * **a number in parentheses** is unwrapped: ``"(0)" == 0``;
    * everything else becomes a whitespace-stripped string (``NULL`` becomes
      ``"None"``).

    Rows are then sorted, so row order does not matter.

    Args:
        res: Result rows (e.g. from ``cursor.fetchall()`` or the ``answer``
            field), or ``None`` for a failed query.

    Returns:
        The sorted list of normalised row tuples, or ``None``.
    """
    if res is None:
        return None
    out = [tuple(_norm_value(v) for v in row) for row in res]
    return sorted(out, key=str)


def _strip_trailing_count(rows: Rows) -> list[tuple[Any, ...]]:
    return [
        tuple(_TRAILING_COUNT.sub("", x) if isinstance(x, str) else x for x in r)
        for r in rows
    ]


def _alnum(v: Any) -> str:
    return re.sub(r"[^0-9A-Za-zÀ-ÿĀ-ž]", "", str(v)).lower()


def results_match(gold: Rows | None, pred: Rows | None) -> bool:
    """Lenient execution match between a reference answer and a prediction.

    The prediction is correct if any of the following holds (checked in order):

    1. **Normalised equality.** :func:`norm` of both sides is equal: row order,
       number types, rounding (2 decimals), thousands separators, currency
       signs, units and parenthesised numbers are ignored.
    2. **Trailing count.** Equal after removing a trailing count in
       parentheses from string cells: ``"Al Horford (15)" == "Al Horford"``
       (a "who" answer does not include the count shown next to the name).
    3. **Two-part answer.** The reference has one column and the prediction
       several, with the same number of rows, and each predicted row,
       concatenated, equals a reference cell up to case and non-alphanumeric
       characters: ``("KeyArena", "10,891") == "KeyArena 10,891"``.
    4. **Duplicates.** Equal as *sets* of normalised rows: duplicate rows do
       not change the answer (``"27 April"`` twice vs ``DISTINCT``).
    5. **Extra columns.** The reference has one column, the prediction several
       (same number of rows), and one of the predicted columns alone matches
       the reference under rule 1 (e.g. the model also returned the count it
       ranked by).

    Args:
        gold: Reference rows (the ``answer`` field).
        pred: Predicted rows, or ``None`` if the query failed.

    Returns:
        ``True`` if the prediction is accepted.
    """
    g, p = norm(gold), norm(pred)
    if p is None or g is None:
        return False
    assert gold is not None and pred is not None
    if g == p:
        return True
    if norm(_strip_trailing_count(gold)) == norm(_strip_trailing_count(pred)):
        return True
    if gold and pred and len(gold[0]) == 1 and len(pred[0]) > 1:
        if len(gold) == len(pred) and sorted(_alnum(r[0]) for r in gold) == sorted(
            _alnum("".join(str(x) for x in r)) for r in pred
        ):
            return True
    if sorted(set(g), key=str) == sorted(set(p), key=str):
        return True
    if gold and pred and len(gold[0]) == 1 and len(pred[0]) > 1:
        if len(gold) == len(pred):
            return any(
                norm([(row[j],) for row in pred]) == g for j in range(len(pred[0]))
            )
    return False
