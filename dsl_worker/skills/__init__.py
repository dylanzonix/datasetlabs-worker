"""Skills: per-topic markdown rules-of-thumb injected into agent prompts.

Each .md file in this directory is a small playbook. Frontmatter declares
who it applies to and what columns trigger it; the body is the rule
content (markdown bullets, examples, what to bail on). At fill time the
loader matches a skill's `triggers` against the target columns'
name/description/format and appends the matching skill bodies to the
cell-agent system prompt.

This is meant to be hand-curated. When we observe a recurring failure
pattern (founders' names differ between Speedrun and X, LinkedIn slug
collisions, etc.), we write it down here so future runs benefit. Auto
patterns can be added later (post-fill summarizer); for now humans edit
these files directly.

Frontmatter shape:

    ---
    name: find_x_handles
    description: Finding X (Twitter) handles for individuals
    applies_to: [cell_agent]
    triggers: [x handle, twitter, twitter handle]
    ---

    body markdown here...
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import yaml


log = logging.getLogger(__name__)


SKILLS_DIR = Path(__file__).parent
_FRONTMATTER_RE = re.compile(r"^---\s*\n(.*?)\n---\s*\n(.*)$", re.DOTALL)


@dataclass
class Skill:
    name: str
    description: str
    body: str
    triggers: List[str] = field(default_factory=list)
    applies_to: List[str] = field(default_factory=lambda: ["cell_agent"])
    file: Optional[str] = None


_skills_cache: Optional[List[Skill]] = None


def _parse_skill_file(path: Path) -> Optional[Skill]:
    try:
        text = path.read_text(encoding="utf-8")
    except Exception:
        log.exception("skills: failed to read %s", path)
        return None
    m = _FRONTMATTER_RE.match(text)
    if not m:
        log.warning("skills: no frontmatter in %s", path.name)
        return None
    front_raw, body = m.group(1), m.group(2).strip()
    try:
        meta = yaml.safe_load(front_raw) or {}
    except yaml.YAMLError:
        log.exception("skills: bad YAML in %s", path.name)
        return None
    if not isinstance(meta, dict):
        log.warning("skills: frontmatter is not a mapping in %s", path.name)
        return None
    name = str(meta.get("name") or path.stem).strip()
    if not name:
        return None
    triggers = meta.get("triggers") or []
    applies_to = meta.get("applies_to") or ["cell_agent"]
    return Skill(
        name=name,
        description=str(meta.get("description") or "").strip(),
        body=body,
        triggers=[str(t).strip().lower() for t in triggers if str(t).strip()],
        applies_to=[str(a).strip() for a in applies_to if str(a).strip()],
        file=path.name,
    )


def load_skills(force: bool = False) -> List[Skill]:
    """Load all skill files. Cached after first call unless force=True.

    Force-reload is mainly useful in dev — in production the worker
    process is long-lived and skills don't change at runtime.
    """
    global _skills_cache
    if _skills_cache is not None and not force:
        return _skills_cache
    out: List[Skill] = []
    if not SKILLS_DIR.exists():
        _skills_cache = out
        return out
    for p in sorted(SKILLS_DIR.glob("*.md")):
        skill = _parse_skill_file(p)
        if skill is not None:
            out.append(skill)
    _skills_cache = out
    return out


def _column_haystack(col: Dict[str, Any]) -> str:
    parts = [
        str(col.get("name") or ""),
        str(col.get("description") or ""),
        str(col.get("format") or ""),
    ]
    return " ".join(parts).lower()


def match_skills(
    applies_to: str,
    columns: Sequence[Dict[str, Any]],
) -> List[Skill]:
    """Return skills whose triggers match any of the given column specs.

    Match is a simple case-insensitive substring check against
    `name + description + format` of each column. A skill matches if
    ANY of its triggers matches ANY of the columns.

    De-duplicated and order-stable (sorted by skill name).
    """
    skills = load_skills()
    matched: Dict[str, Skill] = {}
    haystacks = [_column_haystack(c) for c in columns]
    for s in skills:
        if applies_to not in s.applies_to:
            continue
        if not s.triggers:
            continue
        for trig in s.triggers:
            if any(trig in h for h in haystacks):
                matched[s.name] = s
                break
    return [matched[k] for k in sorted(matched.keys())]


def render_skills(skills: Sequence[Skill]) -> str:
    """Render matched skills as a system-prompt extension. Empty if none."""
    if not skills:
        return ""
    parts = [
        "# Hard-won patterns (skills)",
        "Targeted advice for the kind of column you're filling. Apply when "
        "it fits this row; skip if it doesn't.",
    ]
    for s in skills:
        parts.append("")
        parts.append(s.body.strip())
    return "\n".join(parts)
