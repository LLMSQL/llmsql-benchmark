import re


def find_sql(model_output: str, limit: int = 10) -> list[str]:
    """Function to extract SQL queries from the model's response

    Args:
        model_output (str): Model's response as string
        limit (int, optional): The number of SQL queries to return. Defaults to 10.

    Returns:
        List[str]: SQL queries from input.
    """
    results = []

    # Find all SELECT keywords that could start a query
    for match in re.finditer(r"(?i)(SELECT)", model_output, re.IGNORECASE):
        start_pos = match.start(1)  # Start of SELECT word

        # Look for end of query constraints from this position:
        # semicolumn, block of at least 4 newlines, markdown code fence or just end of string.
        remaining = model_output[start_pos:]
        query_match = re.search(
            r"(?s)SELECT\b.*?(?=(?:;|\n{4,}|```|$))",
            remaining,
            re.IGNORECASE,
        )

        if query_match:
            query = query_match.group(0).strip()
            if query and query not in results:
                results.append(query)

    return results[:limit]


if __name__ == "__main__":
    print(
        find_sql(
            'select "Competition or tour". There"s no aggregation required, just one value possibly multiple rows. Let"s useSELECT "Competition or tour" FROM "2-17637370-13" WHERE "Opponent" = "Nordsjælland" AND "Ground" = "HR"\n'
        )
    )
    # OUTPUT:
    # ['select "Competition or tour". There"s no aggregation required, just one value possibly multiple rows. Let"s useSELECT "Competition or tour" FROM "2-17637370-13" WHERE "Opponent" = "Nordsjælland" AND "Ground" = "HR"', 'SELECT "Competition or tour" FROM "2-17637370-13" WHERE "Opponent" = "Nordsjælland" AND "Ground" = "HR"']
