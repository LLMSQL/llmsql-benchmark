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
