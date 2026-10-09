import json


def build_prompt_5shot(
    question: str,
    headers: list[str],
    types: list[str],
    sample_row: list[str | float | int],
) -> str:
    return f"""You are an expert SQLite SQL query generator.
Your task: Given a question and a table schema, output ONLY a valid SQL SELECT query.
⚠️ STRICT RULES:
 - Output ONLY SQL (no explanations, no markdown, no ``` fences)
 - Use table name "Table"
 - Allowed functions: ['MAX', 'MIN', 'COUNT', 'SUM', 'AVG']
 - Allowed condition operators: ['=', '>', '<', '!=']
 - Allowed SQL keywords: ['SELECT', 'WHERE', 'AND']
 - Always use "" with all column names and table name, even one word: "Price", "General column", "Something #"

### EXAMPLE 1:
Question: What is the price of the Samsung Galaxy S23?
Columns: ['Brand', 'Model', 'Price', 'Storage', 'Color']
Types: ['text', 'text', 'real', 'text', 'text']
Sample row: ['Apple', 'iPhone 14', 899.99, '128GB', 'White']
SQL: SELECT "Price" FROM "Table" WHERE "Brand" = "Samsung" AND "Model" = "Galaxy S23";

### EXAMPLE 2:
Question: How many books did Maya Chen publish?
Columns: ['Author', 'Books Published', 'Genre', 'Country', 'Years Active']
Types: ['text', 'real', 'text', 'text', 'text']
Sample row: ['John Smith', 3, 'Non-fiction', 'Canada', '2005–2015']
SQL: SELECT "Books Published" FROM "Table" WHERE "Author" = "Maya Chen";

### EXAMPLE 3:
Question: What is the total population of cities in California?
Columns: ['City', 'State', 'Population', 'Area', 'Founded']
Types: ['text', 'text', 'real', 'real', 'text']
Sample row: ['Houston', 'Texas', 2304580, 1651.1, '1837']
SQL: SELECT SUM("Population") FROM "Table" WHERE "State" = "California";

### EXAMPLE 4:
Question: How many restaurants serve Italian cuisine?
Columns: ['Restaurant', 'Cuisine', 'Rating', 'City', 'Price Range']
Types: ['text', 'text', 'real', 'text', 'text']
Sample row: ['Golden Dragon', 'Chinese', 4.2, 'Boston', '$$']
SQL: SELECT COUNT(*) FROM "Table" WHERE "Cuisine" = "Italian";

### EXAMPLE 5:
Question: What is the average salary for Software Engineers?
Columns: ['Job Title', 'Salary', 'Experience', 'Location', 'Company Size']
Types: ['text', 'real', 'text', 'text', 'text']
Sample row: ['Data Analyst', 70000, 'Junior', 'Chicago', '200–500']
SQL: SELECT AVG("Salary") FROM "Table" WHERE "Job Title" = "Software Engineer";

### NOW ANSWER:
Question: {question}
Columns: {headers}
Types: {types}
Sample row: {sample_row}
SQL:"""


def build_prompt_1shot(
    question: str,
    headers: list[str],
    types: list[str],
    sample_row: list[str | float | int],
) -> str:
    return f"""You are an expert SQLite SQL query generator.
Your task: Given a question and a table schema, output ONLY a valid SQL SELECT query.
⚠️ STRICT RULES:
 - Output ONLY SQL (no explanations, no markdown, no ``` fences)
 - Use table name "Table"
 - Allowed functions: ['MAX', 'MIN', 'COUNT', 'SUM', 'AVG']
 - Allowed condition operators: ['=', '>', '<', '!=']
 - Allowed SQL keywords: ['SELECT', 'WHERE', 'AND']
 - Always use "" with all column names and table name, even one word: "Price", "General column", "Something #"

### EXAMPLE 1:
Question: What is the price of the Samsung Galaxy S23?
Columns: ['Brand', 'Model', 'Price', 'Storage', 'Color']
Types: ['text', 'text', 'real', 'text', 'text']
Sample row: ['Apple', 'iPhone 14', 899.99, '128GB', 'White']
SQL: SELECT "Price" FROM "Table" WHERE "Brand" = "Samsung" AND "Model" = "Galaxy S23";

### NOW ANSWER:
Question: {question}
Columns: {headers}
Types: {types}
Sample row: {sample_row}
SQL:"""


def build_prompt_0shot(
    question: str,
    headers: list[str],
    types: list[str],
    sample_row: list[str | float | int],
) -> str:
    return f"""You are an expert SQLite SQL query generator.
Your task: Given a question and a table schema, output ONLY a valid SQL SELECT query.
⚠️ STRICT RULES:
 - Output ONLY SQL (no explanations, no markdown, no ``` fences)
 - Use table name "Table"
 - Allowed functions: ['MAX', 'MIN', 'COUNT', 'SUM', 'AVG']
 - Allowed condition operators: ['=', '>', '<', '!=']
 - Allowed SQL keywords: ['SELECT', 'WHERE', 'AND']
 - Always use "" with all column names and table name, even one word: "Price", "General column", "Something #"

### NOW ANSWER:
Question: {question}
Columns: {headers}
Types: {types}
Sample row: {sample_row}
SQL:"""


# ---------------------------------------------------------------------------
# LLMSQL 2.0
# ---------------------------------------------------------------------------

PROMPT_V2_TEMPLATE = """You are an expert SQLite query writer. Given the database schema with a few sample rows and a question,
write a single SQLite query that answers the question. Use the exact table names. Return only the SQL in a ```sql block.

{tables}

Question: {question}"""


def render_table_schema_v2(table: dict, n_rows: int | None = 3) -> str:
    """Render one table for the LLMSQL 2.0 prompt.

    The block consists of a ``CREATE TABLE`` statement with the real table id
    (columns typed ``REAL`` if their type is ``"real"`` and ``TEXT`` otherwise),
    the Wikipedia page / section the table comes from and the first ``n_rows``
    rows, one JSON array per line::

        CREATE TABLE "1-123-1" ("Week" REAL, "Result" TEXT);
        -- Wikipedia: 1990 Team season / Schedule
        -- first 3 of 16 rows:
        [1, "W 21–7"]
        ...

    Args:
        table: Table record with ``table_id``, ``page_title``,
            ``section_title``, ``header``, ``types`` and ``rows``.
        n_rows: Number of sample rows to show (``None`` shows all rows).

    Returns:
        The rendered table block.
    """
    cols = ", ".join(
        f'"{h}" {"REAL" if ty == "real" else "TEXT"}'
        for h, ty in zip(table["header"], table["types"], strict=False)
    )
    all_rows = table["rows"]
    rows = all_rows if n_rows is None else all_rows[:n_rows]
    body = "\n".join(json.dumps(r, ensure_ascii=False) for r in rows)
    shown = (
        "all rows"
        if n_rows is None or n_rows >= len(all_rows)
        else f"first {len(rows)} of {len(all_rows)} rows"
    )
    return (
        f'CREATE TABLE "{table["table_id"]}" ({cols});\n'
        f"-- Wikipedia: {table['page_title']} / {table['section_title']}\n"
        f"-- {shown}:\n{body}"
    )


def build_prompt_v2(question: str, tables: list[dict], n_rows: int = 3) -> str:
    """Build the official zero-shot LLMSQL 2.0 prompt.

    Every table in ``tables`` (the target table plus, for some questions,
    distractor tables from the same Wikipedia page, in the order given by the
    question's ``tables`` field) is rendered with
    :func:`render_table_schema_v2`; the blocks are separated by a blank line.
    The model is asked to answer with a single SQLite query in a fenced ``sql`` code block
    that uses the real table names.

    The result is byte-for-byte identical to the ``prompt`` field shipped
    with the dataset.

    Args:
        question: Natural-language question.
        tables: Table records shown to the model, in order.
        n_rows: Number of sample rows per table (3 in the official protocol).

    Returns:
        The prompt string.
    """
    rendered = "\n\n".join(render_table_schema_v2(t, n_rows) for t in tables)
    return PROMPT_V2_TEMPLATE.format(tables=rendered, question=question)
