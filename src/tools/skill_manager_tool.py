"""skill_manager_tool — list, read, add, and search skills.

The AI sees skills as a directory tree. Skills are stored as markdown
files under ~/.rika/skills/, optionally organized in category subdirectories.
"""
from __future__ import annotations

import pathlib
import re
from typing import Optional

_SKILLS_DIR = pathlib.Path.home() / ".rika" / "skills"

_REQUIRED_HEADER = re.compile(
    r"^#\s*skill:\s*\S+.*\ndescription:\s*\S+",
    re.MULTILINE,
)


def _iter_skills() -> list[dict]:
    """Return all skills with metadata, sorted by category then name."""
    results = []
    for p in sorted(_SKILLS_DIR.rglob("*.md")):
        rel   = p.relative_to(_SKILLS_DIR)
        parts = rel.parts
        cat   = parts[0] if len(parts) > 1 else ""
        name  = p.stem
        desc  = _extract_description(p)
        tools = _extract_tools(p)
        results.append({
            "name": name,
            "category": cat,
            "path": p,
            "description": desc,
            "tools": tools,
        })
    return results


def _extract_description(path: pathlib.Path) -> str:
    for line in path.read_text(errors="replace").splitlines()[:8]:
        if line.startswith("description:"):
            return line[len("description:"):].strip()
    return ""


def _extract_tools(path: pathlib.Path) -> list[str]:
    for line in path.read_text(errors="replace").splitlines()[:8]:
        if line.startswith("tools:"):
            raw = line[len("tools:"):].strip()
            return [t.strip() for t in raw.split(",") if t.strip()]
    return []


def _tree_view(skills: list[dict]) -> str:
    """Render skills as a directory tree."""
    by_cat: dict[str, list[dict]] = {}
    for s in skills:
        by_cat.setdefault(s["category"] or "_root", []).append(s)

    lines = [f"~/.rika/skills/  ({len(skills)} skills)"]
    cats = sorted(by_cat.keys())
    for i, cat in enumerate(cats):
        is_last_cat = i == len(cats) - 1
        cat_prefix  = "└── " if is_last_cat else "├── "
        child_prefix = "    " if is_last_cat else "│   "
        if cat == "_root":
            cat_prefix  = ""
            child_prefix = ""
        else:
            lines.append(f"{cat_prefix}{cat}/")
        entries = by_cat[cat]
        for j, s in enumerate(entries):
            is_last = j == len(entries) - 1
            sym  = "└── " if is_last else "├── "
            note = f"  [{', '.join(s['tools'])}]" if s["tools"] else ""
            lines.append(f"{child_prefix}{sym}{s['name']}{note}")
            if s["description"]:
                desc_prefix = child_prefix + ("    " if is_last else "│   ")
                short = s["description"][:80]
                lines.append(f"{desc_prefix}{short}")
    return "\n".join(lines)


async def skill_manager(
    action: str,
    name: str = "",
    content: str = "",
    query: str = "",
    category: str = "",
) -> str:
    """List, read, add, or search skills in ~/.rika/skills/.

    Args:
        action:   "list" | "read" | "add" | "search"
        name:     skill name (required for read/add)
        content:  full skill markdown (required for add)
        query:    keyword to search (required for search)
        category: optional subdirectory for add (e.g. "dev", "web")
    """
    _SKILLS_DIR.mkdir(parents=True, exist_ok=True)
    action = (action or "list").strip().lower()

    # ── list ─────────────────────────────────────────────────────────────────
    if action == "list":
        skills = _iter_skills()
        if not skills:
            return "No skills found in ~/.rika/skills/. Use action=\"add\" to create one."
        return _tree_view(skills)

    # ── read ─────────────────────────────────────────────────────────────────
    if action == "read":
        if not name:
            return "Error: 'name' is required for action=\"read\"."
        # Search across all categories
        for p in _SKILLS_DIR.rglob(f"{name}.md"):
            return f"# {p.relative_to(_SKILLS_DIR)}\n\n{p.read_text()}"
        return (
            f"Skill '{name}' not found.\n"
            f"Use action=\"list\" to see available skills."
        )

    # ── search ───────────────────────────────────────────────────────────────
    if action == "search":
        if not query:
            return "Error: 'query' is required for action=\"search\"."
        hits = []
        q_lower = query.lower()
        for s in _iter_skills():
            text = s["path"].read_text(errors="replace").lower()
            if q_lower in text or q_lower in s["name"].lower():
                cat_label = f"{s['category']}/" if s["category"] else ""
                hits.append(
                    f"  {cat_label}{s['name']}: {s['description'][:80]}"
                )
        if not hits:
            return f"No skills matched '{query}'."
        return f"Search results for '{query}':\n" + "\n".join(hits)

    # ── add ──────────────────────────────────────────────────────────────────
    if action == "add":
        if not name:
            return "Error: 'name' is required for action=\"add\"."
        if not content:
            return "Error: 'content' is required for action=\"add\"."
        # Validate format
        if not _REQUIRED_HEADER.search(content):
            return (
                "Error: skill content must begin with:\n"
                "# skill: <name>\n"
                "description: <one-line description>\n"
                "tools: <comma-separated tool names>\n\n"
                "<usage body>"
            )
        safe_name = re.sub(r"[^a-z0-9_-]", "_", name.lower())
        if category:
            safe_cat = re.sub(r"[^a-z0-9_-]", "_", category.lower())
            dest_dir = _SKILLS_DIR / safe_cat
        else:
            dest_dir = _SKILLS_DIR
        dest_dir.mkdir(parents=True, exist_ok=True)
        dest = dest_dir / f"{safe_name}.md"
        if dest.exists():
            return (
                f"Skill '{safe_name}' already exists at {dest.relative_to(_SKILLS_DIR)}.\n"
                f"Delete it first if you want to replace it."
            )
        dest.write_text(content, encoding="utf-8")
        # Reload the registry so the new skill is visible immediately
        try:
            from src.core.skill_registry import SkillRegistry
            SkillRegistry.get().reload()
        except Exception:
            pass
        rel = dest.relative_to(_SKILLS_DIR)
        return f"Skill '{safe_name}' saved to ~/.rika/skills/{rel}."

    return f"Unknown action '{action}'. Valid: list, read, add, search."
