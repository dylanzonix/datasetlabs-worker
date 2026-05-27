---
name: file-import
description: Import non-tabular uploaded files (JSON, JSONL, DOCX, TXT, XML) into a table via code_exec + dsl_tools.add_rows(). Load when the user uploads a file the `file` source can't parse (anything that isn't CSV/XLSX).
applies_to: [orchestrator]
---

## Importing non-tabular files

The `file` source only handles CSV/XLSX. For everything else (JSON, JSONL,
DOCX, TXT, XML, etc.), use `code_exec` to parse the file and bulk-insert
rows via `dsl_tools.add_rows()`.

### Step-by-step

1. **Create the table first** so the schema exists:
   ```
   table_create(source="llm", name="...", query_params={
     prompt: "Generate 1 example row matching the schema I'm about to import",
     columns_hint: ["col1", "col2", ...]
   })
   ```

2. **Parse and insert via code_exec**:
   ```
   code_exec(
     table_id: "<table_id from step 1>",
     files: ["<file_id UUID from project state>"],
     code: """
   import json
   import dsl_tools

   data = json.load(open('/workspace/<original_filename>'))
   # Transform into flat row dicts matching the table columns
   rows = [{"col1": item["field1"], "col2": item["field2"]} for item in data]
   dsl_tools.add_rows(rows)
   print(f"Queued {len(rows)} rows")
   """
   )
   ```

### Key rules

- **`files` is required** — pass the file_id UUID from the project state
  banner. Without it the file won't exist in the sandbox. The file lands
  at `/workspace/<original_filename>`.
- **`table_id` is required** — tells the worker which table to insert into
  after the code finishes.
- `dsl_tools.add_rows(items)` writes to a local file; the worker reads it
  after execution and does a single bulk INSERT. Up to 10,000 rows per
  call. No data round-trips through the LLM.
- For multiple `add_rows` calls (e.g. chunking >10k items), each call
  appends to the same ops file — all are committed in one pass.
- The sandbox has `json`, `csv`, `re`, `xml.etree`, `BeautifulSoup`,
  `pandas` available. Use whatever parses the format.

### DOCX files

```python
from docx import Document  # python-docx is available
doc = Document('/workspace/report.docx')
rows = []
for table in doc.tables:
    headers = [cell.text.strip() for cell in table.rows[0].cells]
    for row in table.rows[1:]:
        rows.append({h: cell.text.strip() for h, cell in zip(headers, row.cells)})
dsl_tools.add_rows(rows)
```

### Nested JSON

Flatten before inserting — every value must be a scalar (string, number,
bool, null). Lists/dicts won't render in the table.

```python
rows = []
for item in data:
    rows.append({
        "name": item["name"],
        "email": item.get("contact", {}).get("email", ""),
        "tags": ", ".join(item.get("tags", [])),
    })
dsl_tools.add_rows(rows)
```
