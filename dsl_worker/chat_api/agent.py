"""Chat-mode tool surface for the chat worker API.

The chat agent drives table state directly via flat tools (rows_*/columns_*).
Source tools (FullEnrich/Apollo/etc.) are appended at module load. Storage:

  - "rows" → samples table (one Sample per row, data in `row` JSONB)
  - "columns" → projects.columns (JSONB array)
  - Each chat-mode project has exactly one ProjectVersion (auto-created on
    first tool call) so the samples FK is satisfied. Version status is
    set to "chat" so neither the V13 worker nor the frontend treats it as
    running/paused/etc.
"""

from __future__ import annotations

import json
import logging
import uuid
from typing import Any, Dict, List, Optional, Tuple

from sqlalchemy import select, func
from sqlalchemy.orm import Session

from dsl_api.models import Project
from dsl_api.models.project_version import ProjectVersion
from dsl_api.models.sample import Sample
from dsl_api.schemas.chat import AppliedChange

from dsl_worker.chat_api import sources
from dsl_worker.chat_api import fill


log = logging.getLogger(__name__)


SYSTEM_PROMPT = """\
You are a chat agent that builds data tables collaboratively with the user.
The user describes what they want; you act in small batches and report back
before scaling up. The conversation is the control surface — there is no
separate setup screen, no "Start" button, no plan to approve upfront. Just
chat, do the next obvious thing, and ask before bulk operations.

# Project state

The system context message tells you the current state (column list, row
count, sample). Read it before every response — the table changes between
turns.

# Tools

You have flat function tools in three families:

## Table tools — manipulate the project's rows + columns

- **columns_add / columns_list / columns_modify / columns_delete** — define
  the schema. A column has `name`, optional `format` (e.g. "lowercase email
  or null", "range string like 10-15") and `description`.
- **rows_add(items, merge_key)** — insert rows. With `merge_key`, rows
  matching an existing row's value get merged (no overwrite).
- **rows_get / rows_count / rows_sample** — read.
- **rows_update / rows_delete** — modify (always count first before
  delete).

Filters are dicts: `{col: v}` for equality, `{col__lt: n}` / `__gt` / `__lte`
/ `__gte`, `{col__contains: s}`, `{col__in: [...]}`, `{col__isnull: true}`,
`{col: null}` for IS NULL. Multiple keys AND together.

## Source tools — fetch data from external providers

**FullEnrich** (LinkedIn-derived data, best for B2B/professional targets):
- `fullenrich_search_people(titles, industries, seniority, ...)` — find
  people. Industry values use LinkedIn's taxonomy ("Software Development",
  "Financial Services", "Marketing and Advertising") — NOT casual phrases
  like "SaaS" or "fintech". Broaden filters if a search returns 0.
- `fullenrich_search_companies(industries, headcount_min/max, ...)` — company
  equivalent.
- `fullenrich_enrich_contacts(contacts, fields)` — verified work emails /
  phones. Each contact needs `linkedin_url` OR (`first_name` + `last_name`
  + `domain`). ≤25 per call.

**Apollo** (alternative + complement to FE; people enrichment is paid):
- `apollo_search_companies(keywords, employee_ranges, ...)` — search
  companies. Useful when FE's filters don't match. Free per result.
- `apollo_enrich_person(linkedin_url, email, name, company, domain, ...)`
  — look up a single person's contact. ~$0.024 / match.
- `apollo_enrich_company(domain, name)` — look up a single company. Free.

**Apify** (catch-all for site-specific scrapers):
- `apify_search_actors(query)` — find a pre-built actor for a site. **Use
  this BEFORE any web_harvest** — Apify has 22,000+ actors, almost every
  named site (Reddit, Upwork, LinkedIn, Zillow, etc.) is covered.
- `apify_actor_details(actor_id)` — see an actor's input schema before
  invoking.
- `apify_call_actor(actor_id, input, max_results)` — run the actor and get
  results inline.

**Google Maps** (local businesses, anything with a physical address):
- `google_maps_search_places(query)` — text search. Bake the location into
  the query string ("dentists in Austin TX"). ~$0.032 per request.
- `google_maps_place_details(place_id)` — full details for a place.

## Source-selection rule of thumb

| You want | Reach for |
|----------|-----------|
| People at companies (B2B / tech / professional) | `fullenrich_search_people` |
| Verified emails for known people | `fullenrich_enrich_contacts` |
| Companies (any kind) | `fullenrich_search_companies`, fall back to `apollo_search_companies` |
| Local businesses / orgs / non-LI targets | `google_maps_search_places` |
| Anything from a specific named site (Reddit, Upwork, etc.) | `apify_search_actors` then `apify_call_actor` |
| One person/company lookup by URL or domain | `apollo_enrich_person` / `apollo_enrich_company` |

After a source call, **inspect the inline results, then `rows_add`** the
ones you want with the column shapes the user expects. Don't try to
commit the raw provider record — pick the fields that match the project's
column schema.

# Workflow

Default to interactive conversation, not autonomous big runs. The user
should never feel like work is happening behind their back.

- **Break up anything substantial.** A request like "find 100 founders"
  is not one action — it's "fetch a small batch → confirm fit → fetch
  more → enrich → confirm → enrich more." Do the next obvious step,
  then pause.
- **Clarify briefly** when the ask is ambiguous and the answer would
  meaningfully change what you do. One short question ("which region?",
  "verified emails or LinkedIn URLs?"). Not an interrogation. If the
  ambiguity is small, pick a reasonable default and call it out
  ("starting with US — say if you want global").
- **Pause before big or destructive ops.** Big = >10 rows touched, or
  estimated cost > ~$0.10. Destructive = `rows_delete`, `rows_update`
  on many rows, `columns_delete`. Always show a count + what'll happen,
  then wait.
- **One or two tool calls per turn**, then summarize and ask. Don't
  chain ten things and hope.
- **Sample first.** When you fetch from a source, default to ~10
  results, show the user what came back, ask before pulling the rest.
- **Trust the merge_key for rows.** Don't manually dedup; pass a
  merge_key (LinkedIn URL, domain, place_id) to `rows_add` and let it
  merge.

# Built-ins + last-resort tools

- **web_search** (OpenAI built-in) — quick factual lookups (recent news,
  public bios, "does this company still exist"). For context, NOT for
  list-fetching.
- **code_exec(code)** — Python sandbox, stateless per call. Useful for
  parsing nested data, mapping JSON to flat dicts, calculations. Stdlib
  + httpx + json + re. No DB access; print results, then `rows_add`
  separately.
- **web_harvest(query, candidate_description)** — last-resort research
  subagent that uses web_search + yields candidates. Use ONLY when
  there's no Apify actor for the target site and no structured source
  matches. Slower and pricier than direct APIs.
- **browser_use(task)** — last-resort cloud browser session. ONLY for ONE
  URL + ONE extraction task when Apify has no actor AND the page needs
  JS rendering / anti-bot / interactive flow. Slow (30-180s),
  $0.10-0.50/call.

# Strict rule of thumb for picking a source

1. People / companies on LinkedIn-like data → `fullenrich_*`
2. Specific named site (Reddit, Upwork, Zillow, etc.) → `apify_*`
3. Local biz / orgs / non-LI → `google_maps_*`
4. Job listings → `apollo_*` (or Apify)
5. No structured source fits → `web_harvest`
6. JS-heavy / anti-bot / Apify-doesn't-cover → `browser_use`

# Output style

Be concise. After a tool call, say what happened in one or two sentences
and (when relevant) suggest the next obvious move. No headers, no lists
unless they're genuinely shorter that way.

# Worked example (illustrative)

User: "Find me women's gym apparel founders on LinkedIn"

You: "Quick check — US only, or global? And do you want verified work
emails added too, or just the founder + company?"

User: "US, with emails"

You (one tool call): `fullenrich_search_people` with titles=["Founder",
"Co-Founder","CEO"], industries=["Apparel & Fashion","Sporting Goods"],
locations=["United States"], company_specialties=["women's apparel",
"activewear","gym wear"], limit=10.

You: "Got 10. Sample looks like founders of small (<50 person) athletic-
apparel brands. Want me to commit these and pull more, or refine
filters first?"

(...user approves; you `rows_add` the 10, then continue paginated
search; for each batch ask before the next; once they say "enough",
move to enrichment phase via `fullenrich_enrich_contacts` in batches
of ≤25.)

The shape: tiny clarification, one fetch, show, ask, proceed. Never
chain everything in one giant turn.
"""


# ---------------------------------------------------------------------------
# Tool schemas (OpenAI Responses API function-tool format)
# ---------------------------------------------------------------------------


_FILTER_DESC = (
    "Filter dict. Equality: {col: value}. Operators: {col__lt: n} (also __gt, "
    "__lte, __gte), {col__contains: s}, {col__in: [...]}, {col__isnull: true|false}. "
    "{col: null} → IS NULL. Multiple keys AND together. Empty/missing = all rows."
)


CHAT_TOOLS: List[Dict[str, Any]] = [
    {
        "type": "function",
        "name": "columns_add",
        "description": (
            "Define a new column on the project. Required: name. Optional: format "
            "(consistency hint shown to anything filling the column, e.g. 'lowercase "
            "email or null', 'range string like 10-15', '$X.XM', '(xxx) xxx-xxxx'), "
            "description (what the column means)."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "name": {"type": "string", "description": "Column name in Title Case."},
                "format": {"type": "string"},
                "description": {"type": "string"},
            },
            "required": ["name"],
        },
    },
    {
        "type": "function",
        "name": "columns_list",
        "description": "List all columns defined on the project, in display order.",
        "parameters": {"type": "object", "properties": {}},
    },
    {
        "type": "function",
        "name": "columns_modify",
        "description": "Update a column's metadata. Pass only the fields you want to change.",
        "parameters": {
            "type": "object",
            "properties": {
                "name": {"type": "string"},
                "new_name": {"type": "string", "description": "Rename the column."},
                "format": {"type": "string"},
                "description": {"type": "string"},
            },
            "required": ["name"],
        },
    },
    {
        "type": "function",
        "name": "columns_delete",
        "description": "Drop a column and remove its data from every row.",
        "parameters": {
            "type": "object",
            "properties": {"name": {"type": "string"}},
            "required": ["name"],
        },
    },
    {
        "type": "function",
        "name": "rows_add",
        "description": (
            "Insert (or merge by `merge_key`) a small batch of rows. Each item is a dict "
            "of column-name → value. Values for columns that don't exist yet are stored "
            "anyway; they become visible if you add the column later."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "items": {
                    "type": "array",
                    "items": {"type": "object"},
                    "description": "List of row dicts to insert/merge.",
                },
                "merge_key": {
                    "type": "string",
                    "description": (
                        "Column name. If a row with the same value exists, merge non-null "
                        "fields from the incoming row into the existing one (no overwrite "
                        "of non-null cells)."
                    ),
                },
            },
            "required": ["items"],
        },
    },
    {
        "type": "function",
        "name": "rows_count",
        "description": "Count rows matching `where`. " + _FILTER_DESC,
        "parameters": {
            "type": "object",
            "properties": {"where": {"type": "object"}},
        },
    },
    {
        "type": "function",
        "name": "rows_get",
        "description": (
            "Fetch rows matching `where`. Returns list of dicts. "
            + _FILTER_DESC
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "where": {"type": "object"},
                "limit": {"type": "integer", "minimum": 1},
                "columns": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Project specific columns. Default: all.",
                },
            },
        },
    },
    {
        "type": "function",
        "name": "rows_sample",
        "description": "Return up to N random rows for spot-checking the table.",
        "parameters": {
            "type": "object",
            "properties": {"n": {"type": "integer", "minimum": 1, "default": 3}},
        },
    },
    {
        "type": "function",
        "name": "rows_update",
        "description": "Set the given column values on every row matching `where`.",
        "parameters": {
            "type": "object",
            "properties": {
                "where": {"type": "object"},
                "values": {"type": "object"},
            },
            "required": ["where", "values"],
        },
    },
    {
        "type": "function",
        "name": "rows_delete",
        "description": (
            "Delete every row matching `where`. Always run rows_count with the same `where` "
            "first and tell the user how many will be deleted before calling this."
        ),
        "parameters": {
            "type": "object",
            "properties": {"where": {"type": "object"}},
            "required": ["where"],
        },
    },
    {
        "type": "function",
        "name": "rows_fill",
        "description": (
            "Per-cell research+fill. For each row matching `where`, spawn a "
            "small bounded subagent that uses the source tools (FE / Apollo "
            "/ Apify / Google Maps / web_harvest / browser_use) to research "
            "the listed columns and commit values. Each cell agent has a "
            "tight budget cap and turn limit; up to 5 cells run in parallel. "
            "Use this for 'fill emails for these rows', 'find LinkedIn URL "
            "for each', etc. Always start with a small `limit` (5-10 rows) "
            "to validate before running on the rest."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "columns": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "One or more column names to fill. The cell agent fills all listed columns for each row in one shot.",
                },
                "where": {
                    "type": "object",
                    "description": (
                        "Same filter syntax as rows_get. Typical pattern: "
                        "{<column>: null} to target unfilled cells only."
                    ),
                },
                "limit": {"type": "integer", "minimum": 1, "maximum": 200, "description": "Max rows to fill in this call (default 20)."},
                "max_cost": {"type": "number", "description": "Per-cell budget cap in USD (default 0.10)."},
                "max_turns": {"type": "integer", "minimum": 1, "maximum": 10, "description": "Per-cell turn cap (default 5)."},
            },
            "required": ["columns"],
        },
    },
    # Source tools (FullEnrich, Apollo, Apify, Google Maps, etc.) are
    # appended at module load so they're part of the same flat surface the
    # chat agent sees. Their handlers live in `sources`.
    *sources.SOURCE_TOOLS,
    # Built-in OpenAI web search. OpenAI handles invocation natively; the
    # tool-call loop just sees web_search_call items in the response (it
    # filters those out — only function_call items dispatch through our
    # execute_tool). Lets the agent do quick factual lookups without us
    # writing a wrapper.
    {
        "type": "web_search",
        "user_location": {"type": "approximate"},
        "search_context_size": "medium",
    },
]


# ---------------------------------------------------------------------------
# Helpers — version, where translation
# ---------------------------------------------------------------------------


def ensure_chat_version(db: Session, project: Project) -> ProjectVersion:
    """Make sure a chat-mode project has exactly one ProjectVersion to anchor
    samples on. Creates it lazily on first need.
    """
    if project.current_version_id is not None:
        version = db.query(ProjectVersion).filter(
            ProjectVersion.id == project.current_version_id
        ).first()
        if version is not None:
            return version

    version = ProjectVersion(
        project_id=project.id,
        version_number=1,
        num_samples=0,                    # not used in chat mode
        columns=project.columns or [],
        files_snapshot=[],
        examples_snapshot=[],
        status="chat",                    # neither worker nor frontend matches this
        generated_count=0,
    )
    db.add(version)
    db.flush()
    project.current_version_id = version.id
    return version


_OP_SUFFIXES = {
    "__ne": "!=",
    "__lt": "<",
    "__gt": ">",
    "__lte": "<=",
    "__gte": ">=",
}


def _row_value_expr(field: str) -> str:
    """SQL expression that selects a field from samples.row JSONB."""
    # Use the ->> operator which yields TEXT (or NULL).
    safe = field.replace("'", "''")
    return f"row ->> '{safe}'"


def _sql_param(idx: int) -> str:
    return f":p{idx}"


def _where_to_sql(where: Optional[Dict[str, Any]]) -> Tuple[str, Dict[str, Any]]:
    """Translate a dict-where into a Postgres SQL fragment + bound params.

    Returns ("TRUE", {}) for empty filters so callers can always inline it
    after WHERE.
    """
    if not where:
        return "TRUE", {}
    clauses: List[str] = []
    params: Dict[str, Any] = {}

    def add_param(value: Any) -> str:
        idx = len(params)
        key = f"p{idx}"
        params[key] = value
        return f":{key}"

    for raw_key, value in where.items():
        # Operator suffixes
        op = "="
        field = raw_key
        matched_op = False
        for suffix, sym in _OP_SUFFIXES.items():
            if raw_key.endswith(suffix):
                op = sym
                field = raw_key[: -len(suffix)]
                matched_op = True
                break

        if not matched_op:
            if raw_key.endswith("__contains"):
                field = raw_key[: -len("__contains")]
                clauses.append(f"{_row_value_expr(field)} LIKE {add_param(f'%{value}%')}")
                continue
            if raw_key.endswith("__in"):
                field = raw_key[: -len("__in")]
                if not isinstance(value, list) or not value:
                    clauses.append("FALSE")
                    continue
                placeholders = []
                for v in value:
                    placeholders.append(add_param(str(v)))
                clauses.append(f"{_row_value_expr(field)} IN ({', '.join(placeholders)})")
                continue
            if raw_key.endswith("__isnull"):
                field = raw_key[: -len("__isnull")]
                if value:
                    clauses.append(f"{_row_value_expr(field)} IS NULL")
                else:
                    clauses.append(f"{_row_value_expr(field)} IS NOT NULL")
                continue

        # Plain equality / operator with a value
        if value is None:
            if op == "=":
                clauses.append(f"{_row_value_expr(field)} IS NULL")
            elif op == "!=":
                clauses.append(f"{_row_value_expr(field)} IS NOT NULL")
            else:
                raise ValueError(f"Cannot compare {raw_key!r} to NULL with {op}")
            continue

        # JSONB ->> always returns text; coerce if comparing to a number.
        cast = "::numeric" if isinstance(value, (int, float)) and not isinstance(value, bool) else ""
        clauses.append(f"({_row_value_expr(field)}){cast} {op} {add_param(value)}")

    return " AND ".join(clauses), params


# ---------------------------------------------------------------------------
# Tool execution
# ---------------------------------------------------------------------------


def _column_index(project: Project) -> Dict[str, Dict[str, Any]]:
    return {c.get("name"): c for c in (project.columns or []) if isinstance(c, dict)}


def _next_seq(db: Session, version_id: uuid.UUID) -> int:
    last = db.query(func.max(Sample.seq)).filter(Sample.version_id == version_id).scalar()
    return (last or 0) + 1


def _row_to_dict(s: Sample) -> Dict[str, Any]:
    out = {"_id": str(s.id), "_seq": s.seq}
    if isinstance(s.row, dict):
        for k, v in s.row.items():
            out[k] = v
    return out


async def execute_tool(
    db: Session,
    project: Project,
    tool_name: str,
    args: Dict[str, Any],
    progress_cb: Optional[fill.ProgressCallback] = None,
) -> Tuple[Dict[str, Any], Dict[str, Any], float]:
    """Run one tool. Returns (applied, result, cost_usd).

    - `applied` is the change-summary the route persists onto the assistant
      chat message (drives the rendered "X changed" hints in the UI).
    - `result` is the structured payload we feed back to the LLM as the
      tool's output (serialized to JSON downstream).
    - `cost_usd` is the raw provider cost (FE/Apollo/Apify/etc.) for this
      tool call — the chat handler accumulates these and charges the
      user's credit balance at the end of the turn.
    """
    # Source tools (FullEnrich, etc.) — handled inline since they're async
    # and have their own cost reporting.
    if sources.is_source_tool(tool_name):
        result_text, cost_usd = await sources.execute_source_tool(
            tool_name, args
        )
        # Source tools don't make table changes themselves; the agent calls
        # rows_add / rows_update afterwards.
        try:
            result_dict = json.loads(result_text)
        except (TypeError, ValueError):
            result_dict = {"raw": result_text}
        return ({}, result_dict, cost_usd)

    version = ensure_chat_version(db, project)

    if tool_name == "columns_add":
        applied, result = _tool_columns_add(db, project, args)
        return applied, result, 0.0
    if tool_name == "columns_list":
        applied, result = _tool_columns_list(db, project, args)
        return applied, result, 0.0
    if tool_name == "columns_modify":
        applied, result = _tool_columns_modify(db, project, args)
        return applied, result, 0.0
    if tool_name == "columns_delete":
        applied, result = _tool_columns_delete(db, project, args)
        return applied, result, 0.0
    if tool_name == "rows_add":
        applied, result = _tool_rows_add(db, project, version, args)
        return applied, result, 0.0
    if tool_name == "rows_count":
        applied, result = _tool_rows_count(db, project, version, args)
        return applied, result, 0.0
    if tool_name == "rows_get":
        applied, result = _tool_rows_get(db, project, version, args)
        return applied, result, 0.0
    if tool_name == "rows_sample":
        applied, result = _tool_rows_sample(db, project, version, args)
        return applied, result, 0.0
    if tool_name == "rows_update":
        applied, result = _tool_rows_update(db, project, version, args)
        return applied, result, 0.0
    if tool_name == "rows_delete":
        applied, result = _tool_rows_delete(db, project, version, args)
        return applied, result, 0.0
    if tool_name == "rows_fill":
        applied, result, cost = await _tool_rows_fill(
            db, project, version, args, progress_cb=progress_cb
        )
        return applied, result, cost

    return ({}, {"error": f"unknown tool: {tool_name}"}, 0.0)


async def _tool_rows_fill(
    db: Session,
    project: Project,
    version: ProjectVersion,
    args: Dict[str, Any],
    progress_cb: Optional[fill.ProgressCallback] = None,
) -> Tuple[Dict[str, Any], Dict[str, Any], float]:
    columns = args.get("columns") or []
    if not isinstance(columns, list) or not columns:
        return {}, {"error": "columns must be a non-empty list of column names"}, 0.0

    where = args.get("where") or {}
    limit = args.get("limit", 20)
    if limit is not None:
        limit = min(int(limit), 200)
    max_cost = float(args.get("max_cost", 0.10))
    max_turns = int(args.get("max_turns", 5))

    where_sql, where_params = _where_to_sql(where)

    summary, total_cost = await fill.fill_rows(
        project=project,
        target_columns=columns,
        where_sql=where_sql,
        where_params=where_params,
        limit=limit,
        max_cost=max_cost,
        max_turns=max_turns,
        progress_cb=progress_cb,
    )
    applied = {"rows_filled": summary.get("cells_filled", 0)}
    return applied, summary, total_cost


# --- Column tools ---


def _tool_columns_add(db: Session, project: Project, args: Dict[str, Any]):
    name = (args.get("name") or "").strip()
    if not name:
        return {}, {"error": "name is required"}
    cols = list(project.columns or [])
    existing = _column_index(project)
    if name in existing:
        return {}, {"error": f"column {name!r} already exists"}
    new_col = {"name": name}
    if args.get("format"):
        new_col["format"] = args["format"]
    if args.get("description"):
        new_col["description"] = args["description"]
    cols.append(new_col)
    project.columns = cols
    if project.current_version is not None:
        project.current_version.columns = cols
    return {"columns": cols}, {"ok": True, "column": new_col, "total_columns": len(cols)}


def _tool_columns_list(db: Session, project: Project, args: Dict[str, Any]):
    return {}, {"columns": list(project.columns or [])}


def _tool_columns_modify(db: Session, project: Project, args: Dict[str, Any]):
    name = (args.get("name") or "").strip()
    if not name:
        return {}, {"error": "name is required"}
    cols = list(project.columns or [])
    target = None
    for c in cols:
        if isinstance(c, dict) and c.get("name") == name:
            target = c
            break
    if target is None:
        return {}, {"error": f"column {name!r} not found"}

    new_name = args.get("new_name")
    rename_field: Optional[Tuple[str, str]] = None
    if new_name and new_name != name:
        if any(isinstance(c, dict) and c.get("name") == new_name for c in cols):
            return {}, {"error": f"column {new_name!r} already exists"}
        rename_field = (name, new_name)
        target["name"] = new_name
    if "format" in args:
        target["format"] = args["format"]
    if "description" in args:
        target["description"] = args["description"]

    project.columns = cols
    if project.current_version is not None:
        project.current_version.columns = cols

    # If renamed, also rewrite samples.row keys
    if rename_field:
        old, new = rename_field
        version = project.current_version
        if version is not None:
            for sample in db.query(Sample).filter(Sample.version_id == version.id).all():
                data = dict(sample.row or {})
                if old in data:
                    data[new] = data.pop(old)
                    sample.row = data

    return {"columns": cols}, {"ok": True, "column": target}


def _tool_columns_delete(db: Session, project: Project, args: Dict[str, Any]):
    name = (args.get("name") or "").strip()
    if not name:
        return {}, {"error": "name is required"}
    cols = list(project.columns or [])
    new_cols = [c for c in cols if not (isinstance(c, dict) and c.get("name") == name)]
    if len(new_cols) == len(cols):
        return {}, {"error": f"column {name!r} not found"}
    project.columns = new_cols
    if project.current_version is not None:
        project.current_version.columns = new_cols
        # Strip the field from every row in this version
        affected = 0
        for sample in db.query(Sample).filter(Sample.version_id == project.current_version.id).all():
            data = dict(sample.row or {})
            if name in data:
                data.pop(name)
                sample.row = data
                affected += 1
    else:
        affected = 0
    return {"columns": new_cols}, {"ok": True, "rows_with_data_dropped": affected}


# --- Row tools ---


def _tool_rows_add(
    db: Session, project: Project, version: ProjectVersion, args: Dict[str, Any]
):
    items = args.get("items")
    if not isinstance(items, list):
        return {}, {"error": "items must be a list of objects"}
    merge_key = args.get("merge_key")

    inserted = 0
    merged = 0
    for item in items:
        if not isinstance(item, dict):
            continue
        merged_existing = False
        if merge_key:
            mv = item.get(merge_key)
            if mv is not None:
                from sqlalchemy import text
                stmt = text(
                    f"SELECT id, row FROM samples WHERE version_id = :vid "
                    f"AND ({_row_value_expr(merge_key)}) = :mv LIMIT 1"
                )
                existing = db.execute(
                    stmt, {"vid": version.id, "mv": str(mv)}
                ).first()
                if existing:
                    sample = db.query(Sample).filter(Sample.id == existing[0]).first()
                    if sample is not None:
                        data = dict(sample.row or {})
                        for k, v in item.items():
                            if v is not None and (data.get(k) is None or data.get(k) == ""):
                                data[k] = v
                        sample.row = data
                        merged += 1
                        merged_existing = True
        if merged_existing:
            continue

        seq = _next_seq(db, version.id)
        sample = Sample(
            project_id=project.id,
            version_id=version.id,
            seq=seq,
            row=dict(item),
            tags={},
        )
        db.add(sample)
        db.flush()
        version.generated_count = (version.generated_count or 0) + 1
        inserted += 1

    total = db.query(func.count(Sample.id)).filter(Sample.version_id == version.id).scalar() or 0
    return (
        {"rows": {"inserted": inserted, "merged": merged}},
        {"ok": True, "inserted": inserted, "merged": merged, "total": total},
    )


def _tool_rows_count(
    db: Session, project: Project, version: ProjectVersion, args: Dict[str, Any]
):
    where = args.get("where") or {}
    sql, params = _where_to_sql(where)
    from sqlalchemy import text
    stmt = text(
        f"SELECT COUNT(*) FROM samples WHERE version_id = :vid AND ({sql})"
    )
    n = db.execute(stmt, {"vid": version.id, **params}).scalar() or 0
    return {}, {"count": int(n)}


def _tool_rows_get(
    db: Session, project: Project, version: ProjectVersion, args: Dict[str, Any]
):
    where = args.get("where") or {}
    limit = args.get("limit")
    columns = args.get("columns")
    sql, params = _where_to_sql(where)
    from sqlalchemy import text
    base = (
        f"SELECT id, seq, row FROM samples WHERE version_id = :vid AND ({sql}) "
        f"ORDER BY seq"
    )
    if limit is not None:
        base += f" LIMIT {int(limit)}"
    rows = db.execute(text(base), {"vid": version.id, **params}).all()

    out: List[Dict[str, Any]] = []
    for r in rows:
        rid, seq, data = r
        d = {"_id": str(rid), "_seq": seq}
        if isinstance(data, dict):
            d.update(data)
        if columns is not None:
            d = {"_id": d["_id"], "_seq": d["_seq"], **{k: d.get(k) for k in columns}}
        out.append(d)
    return {}, {"rows": out, "count": len(out)}


def _tool_rows_sample(
    db: Session, project: Project, version: ProjectVersion, args: Dict[str, Any]
):
    n = int(args.get("n") or 3)
    from sqlalchemy import text
    rows = db.execute(
        text("SELECT id, seq, row FROM samples WHERE version_id = :vid ORDER BY RANDOM() LIMIT :n"),
        {"vid": version.id, "n": n},
    ).all()
    out: List[Dict[str, Any]] = []
    for r in rows:
        rid, seq, data = r
        d = {"_id": str(rid), "_seq": seq}
        if isinstance(data, dict):
            d.update(data)
        out.append(d)
    return {}, {"rows": out}


def _tool_rows_update(
    db: Session, project: Project, version: ProjectVersion, args: Dict[str, Any]
):
    where = args.get("where") or {}
    values = args.get("values") or {}
    if not isinstance(values, dict) or not values:
        return {}, {"error": "values must be a non-empty object"}
    sql, params = _where_to_sql(where)
    from sqlalchemy import text
    rows = db.execute(
        text(f"SELECT id FROM samples WHERE version_id = :vid AND ({sql})"),
        {"vid": version.id, **params},
    ).all()
    affected = 0
    for (rid,) in rows:
        sample = db.query(Sample).filter(Sample.id == rid).first()
        if sample is None:
            continue
        data = dict(sample.row or {})
        data.update(values)
        sample.row = data
        affected += 1
    return {"rows_updated": affected}, {"ok": True, "affected": affected}


def _tool_rows_delete(
    db: Session, project: Project, version: ProjectVersion, args: Dict[str, Any]
):
    where = args.get("where") or {}
    sql, params = _where_to_sql(where)
    from sqlalchemy import text
    rows = db.execute(
        text(f"SELECT id FROM samples WHERE version_id = :vid AND ({sql})"),
        {"vid": version.id, **params},
    ).all()
    deleted = 0
    for (rid,) in rows:
        sample = db.query(Sample).filter(Sample.id == rid).first()
        if sample is not None:
            db.delete(sample)
            deleted += 1
    if deleted and version.generated_count is not None:
        version.generated_count = max(0, version.generated_count - deleted)
    return {"rows_deleted": deleted}, {"ok": True, "deleted": deleted}


# ---------------------------------------------------------------------------
# Result formatting + applied-change description
# ---------------------------------------------------------------------------


def format_tool_result(tool_name: str, result: Dict[str, Any]) -> str:
    """Stringify the tool result for the LLM input feed."""
    return json.dumps(result, default=str)[:4000]


def describe_applied(applied: Dict[str, Any]) -> List[AppliedChange]:
    """Convert the merged 'applied' map across this turn's tool calls into
    AppliedChange records the frontend can render.
    """
    out: List[AppliedChange] = []
    if "columns" in applied:
        cols = applied["columns"]
        names = [c.get("name", "?") for c in cols if isinstance(c, dict)]
        preview = ", ".join(names[:5])
        if len(names) > 5:
            preview += f" (+{len(names) - 5} more)"
        out.append(AppliedChange(
            field="columns",
            description=f"Schema now {len(names)} column(s): {preview}",
        ))
    if "rows" in applied:
        ins = applied["rows"].get("inserted", 0)
        mer = applied["rows"].get("merged", 0)
        bits: List[str] = []
        if ins:
            bits.append(f"added {ins}")
        if mer:
            bits.append(f"merged {mer}")
        out.append(AppliedChange(
            field="rows",
            description="Rows: " + ", ".join(bits) if bits else "Rows updated",
        ))
    if "rows_updated" in applied:
        out.append(AppliedChange(
            field="rows",
            description=f"Updated {applied['rows_updated']} row(s)",
        ))
    if "rows_deleted" in applied:
        out.append(AppliedChange(
            field="rows",
            description=f"Deleted {applied['rows_deleted']} row(s)",
        ))
    if "rows_filled" in applied:
        out.append(AppliedChange(
            field="rows",
            description=f"Filled {applied['rows_filled']} cell(s)",
        ))
    return out


def build_context_message(db: Session, project: Project) -> str:
    """Compact state summary handed to the LLM as a system context message
    every turn."""
    parts = [f"Project: {project.name}"]

    # Columns
    cols = project.columns or []
    if cols:
        parts.append("Columns:")
        for c in cols:
            if not isinstance(c, dict):
                continue
            line = f"  - {c.get('name', '?')}"
            if c.get("format"):
                line += f" ({c['format']})"
            if c.get("description"):
                line += f" — {c['description']}"
            parts.append(line)
    else:
        parts.append("Columns: (none yet)")

    # Row count + tiny sample (only if we have a version)
    if project.current_version_id:
        version_id = project.current_version_id
        n = db.query(func.count(Sample.id)).filter(Sample.version_id == version_id).scalar() or 0
        parts.append(f"Rows: {n}")
        if n > 0:
            samples = (
                db.query(Sample)
                .filter(Sample.version_id == version_id)
                .order_by(Sample.seq)
                .limit(3)
                .all()
            )
            parts.append("Sample (first 3):")
            for s in samples:
                row_repr = json.dumps(s.row, default=str)
                if len(row_repr) > 200:
                    row_repr = row_repr[:200] + "..."
                parts.append(f"  - seq={s.seq}: {row_repr}")
    else:
        parts.append("Rows: 0 (no version yet)")

    return "\n".join(parts)
