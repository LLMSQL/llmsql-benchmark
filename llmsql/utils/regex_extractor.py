import re

_FENCE = "```"

# A fenced block tagged as SQL, e.g. ```sql\nSELECT 1\n```
_FENCED_SQL = re.compile(r"```[ \t]*sql[ \t]*\r?\n(.*?)```", re.IGNORECASE | re.DOTALL)

# Every occurrence of the SELECT keyword starts a candidate query
_SELECT = re.compile(r"(?i)(SELECT)")

# Characters that open a quoted run: a string literal, a quoted identifier
# or a backtick-quoted identifier.
_QUOTES = "'\"" + "`"


def _skip_quoted(text: str, index: int) -> int:
    """Return the index just past the quoted run that opens at ``index``.

    SQL escapes a quote inside a literal by doubling it, so ``'it''s'`` is a
    single literal rather than two adjacent ones.
    """
    quote = text[index]
    position = index + 1
    while position < len(text):
        if text[position] == quote:
            if text.startswith(quote * 2, position):
                position += 2
                continue
            return position + 1
        position += 1
    return len(text)


def _query_end(text: str, start: int) -> int:
    """Return the index just past the query that starts at ``start``.

    A query ends at a semicolon, at a run of at least four newlines, at a
    markdown fence, or at the end of the text. Semicolons inside a quoted
    literal or a quoted identifier do not end the query, so
    ``WHERE b = 'x;y';`` is no longer cut down to ``WHERE b = 'x``.
    """
    index = start
    newlines = 0
    while index < len(text):
        # A fence is checked before quotes: ``` is a markdown marker, not the
        # start of a backtick-quoted identifier.
        if text.startswith(_FENCE, index):
            return index
        char = text[index]
        if char in _QUOTES:
            index = _skip_quoted(text, index)
            newlines = 0
            continue
        if char == ";":
            return index
        if char == "\n":
            newlines += 1
            if newlines >= 4:
                return index - (newlines - 1)
        elif char != "\r":
            newlines = 0
        index += 1
    return len(text)


def _fenced_sql_blocks(model_output: str) -> list[str]:
    """Return the contents of every ````sql`` fenced block, in order."""
    blocks: list[str] = []
    for match in _FENCED_SQL.finditer(model_output):
        query = match.group(1).strip()
        # The scanner drops the terminating ";", so fenced queries match it.
        if query.endswith(";"):
            query = query[:-1].strip()
        if query and "select" in query.lower() and query not in blocks:
            blocks.append(query)
    return blocks


def _scan_selects(model_output: str) -> list[str]:
    """Return one candidate query per SELECT keyword occurrence."""
    results: list[str] = []
    for match in _SELECT.finditer(model_output):
        start_pos = match.start(1)  # Start of SELECT word
        query = model_output[start_pos : _query_end(model_output, start_pos)].strip()
        if query and query not in results:
            results.append(query)
    return results


def find_sql(model_output: str, limit: int = 10) -> list[str]:
    """Function to extract SQL queries from the model's response

    A fenced ````sql`` block is the clearest statement of intent, so its
    contents win over any SELECT found in the surrounding prose. Without a
    fence, every SELECT keyword starts a candidate.

    The last ``limit`` candidates are returned, not the first ones: a model
    that reasons out loud before answering starts its prose with words like
    "select", and those prose candidates used to fill every slot and push the
    real query out of the list.

    Args:
        model_output (str): Model's response as string
        limit (int, optional): The number of SQL queries to return. Defaults to 10.

    Returns:
        List[str]: SQL queries from input.
    """
    if limit <= 0:
        return []

    results = _fenced_sql_blocks(model_output) or _scan_selects(model_output)
    return results[-limit:]


if __name__ == "__main__":
    print(
        find_sql(
            'select "Competition or tour". There"s no aggregation required, just one value possibly multiple rows. Let"s useSELECT "Competition or tour" FROM "2-17637370-13" WHERE "Opponent" = "Nordsjælland" AND "Ground" = "HR"\n'
        )
    )
    # OUTPUT:
    # ['select "Competition or tour". There"s no aggregation required, just one value possibly multiple rows. Let"s useSELECT "Competition or tour" FROM "2-17637370-13" WHERE "Opponent" = "Nordsjælland" AND "Ground" = "HR"', 'SELECT "Competition or tour" FROM "2-17637370-13" WHERE "Opponent" = "Nordsjælland" AND "Ground" = "HR"']
