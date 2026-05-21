"""Skills: per-task markdown playbooks loaded on demand.

Each `.md` file in this directory is a skill — a hand-written playbook for
a specific recurring task pattern. Frontmatter declares the skill's name,
one-line description, and where it applies; the body is the playbook itself.

The directory is a curated reference shelf, not an exhaustive capability
list. Most tasks won't match any skill — that's expected. Skills are how
we accumulate hard-won patterns over time without ballooning the system
prompt.

Two scopes:

  applies_to: [orchestrator]   — orchestrator-level task playbook
                                  (e.g. "how to find subreddits about a topic")
  applies_to: [cell_agent]     — per-cell enrichment playbook
                                  (e.g. "how to detect if a company runs ads")

Both surfaces list available skills (name + description) in their system
prompts. The model calls `load_skill(name)` to read the playbook body
when relevant. Bodies never enter context until called.

Frontmatter shape:

    ---
    name: find-subreddits
    description: Listing subreddits relevant to a topic or audience.
    applies_to: [orchestrator]
    ---

    body markdown...
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

import yaml


log = logging.getLogger(__name__)


SKILLS_DIR = Path(__file__).parent
_FRONTMATTER_RE = re.compile(r"^---\s*\n(.*?)\n---\s*\n(.*)$", re.DOTALL)


@dataclass
class Skill:
    name: str
    description: str
    body: str
    applies_to: List[str] = field(default_factory=lambda: ["orchestrator"])
    file: Optional[str] = None


_skills_cache: Optional[Dict[str, Skill]] = None
_skills_cache_dir_mtime: Optional[float] = None


def _dir_mtime() -> float:
    """Max mtime across the skills directory + all .md files in it.

    Cheap (a handful of stat calls), and changes when any skill file is
    edited / added / removed. Used to invalidate the in-process cache so
    `.md` edits land without restarting the worker — uvicorn --reload
    watches Python files only.
    """
    if not SKILLS_DIR.exists():
        return 0.0
    times = [SKILLS_DIR.stat().st_mtime]
    for p in SKILLS_DIR.glob("*.md"):
        try:
            times.append(p.stat().st_mtime)
        except OSError:
            continue
    return max(times) if times else 0.0


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
    applies_to = meta.get("applies_to") or ["orchestrator"]
    return Skill(
        name=name,
        description=str(meta.get("description") or "").strip(),
        body=body,
        applies_to=[str(a).strip() for a in applies_to if str(a).strip()],
        file=path.name,
    )


def _load_skills(force: bool = False) -> Dict[str, Skill]:
    """Load all skill files into a {name: Skill} map.

    Cached in-process, but the cache is invalidated when any `.md` file
    in the skills directory changes. The mtime check is cheap (a few
    stat calls per request) and lets `.md` edits land without restarting
    the worker.
    """
    global _skills_cache, _skills_cache_dir_mtime
    current_mtime = _dir_mtime()
    if (
        not force
        and _skills_cache is not None
        and _skills_cache_dir_mtime == current_mtime
    ):
        return _skills_cache
    out: Dict[str, Skill] = {}
    if SKILLS_DIR.exists():
        for p in sorted(SKILLS_DIR.glob("*.md")):
            skill = _parse_skill_file(p)
            if skill is not None:
                out[skill.name] = skill
    _skills_cache = out
    _skills_cache_dir_mtime = current_mtime
    return out


def list_all_skills() -> List[Dict[str, object]]:
    """Return every skill's name + description + applies_to.

    Used by the orchestrator system prompt to show the full directory
    (with `(orchestrator)` / `(enrichment)` markers). Result is stable
    across runs because skills change rarely.
    """
    return [
        {
            "name": s.name,
            "description": s.description,
            "applies_to": list(s.applies_to),
        }
        for s in _load_skills().values()
    ]


def list_enrichment_skills() -> List[Dict[str, str]]:
    """Return skills applicable to the research-tier cell_agent.

    Used to render the cell_agent's `# Skills` section so it knows which
    enrichment playbooks are available before deciding whether to load
    one via load_skill.
    """
    return [
        {"name": s.name, "description": s.description}
        for s in _load_skills().values()
        if "cell_agent" in s.applies_to
    ]


def get_skill_body(name: str) -> Optional[str]:
    """Return the body of a named skill, or None if not found.

    Called by the load_skill tool handler. Strips frontmatter; returns
    just the playbook text the agent should follow.
    """
    skill = _load_skills().get(name)
    return skill.body if skill is not None else None
