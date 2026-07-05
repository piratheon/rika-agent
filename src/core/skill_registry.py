"""SkillRegistry — on-demand skill loading from ~/.rika/skills/.

Skills are Markdown files. On first run, bundled skills from src/skills/
are copied to ~/.rika/skills/. Users can add custom skills there.

Each skill file has:
  - A YAML-style header: `# skill: name` and `description: ...` and `tools: ...`
  - Body: usage instructions injected into the system prompt when activated

The LLM sees a compact skill list in every system prompt (name + description only).
Full instructions are injected only when use_skill(skill_name) is called.
"""
from __future__ import annotations

import re
import shutil
from pathlib import Path
from typing import Dict, List, Optional

from src.utils.logger import logger

_BUNDLED_SKILLS_DIR = Path(__file__).parent.parent / "skills"
_USER_SKILLS_DIR    = Path.home() / ".rika" / "skills"


def ensure_skills_bootstrapped() -> None:
    """Copy bundled skills to ~/.rika/skills/ preserving subdirectory structure.

    Never overwrites existing user-edited skills.
    """
    # patch_skills_manager: rglob_bootstrap
    _USER_SKILLS_DIR.mkdir(parents=True, exist_ok=True)
    if not _BUNDLED_SKILLS_DIR.exists():
        return
    for src in sorted(_BUNDLED_SKILLS_DIR.rglob("*.md")):
        rel  = src.relative_to(_BUNDLED_SKILLS_DIR)
        dest = _USER_SKILLS_DIR / rel
        dest.parent.mkdir(parents=True, exist_ok=True)
        if not dest.exists():
            shutil.copy2(src, dest)
            logger.info("skill_bootstrapped", skill=str(rel))


def _parse_skill_file(path: Path) -> Optional[Dict]:
    """Parse a skill markdown file and return its metadata dict."""
    try:
        text = path.read_text(encoding="utf-8")
    except OSError:
        return None

    lines = text.splitlines()
    name = description = tools_raw = None

    for line in lines[:10]:
        line = line.strip()
        if line.startswith("# skill:"):
            name = line[len("# skill:"):].strip()
        elif line.startswith("description:"):
            description = line[len("description:"):].strip()
        elif line.startswith("tools:"):
            tools_raw = line[len("tools:"):].strip()

    if not name or not description:
        return None

    tools: List[str] = [t.strip() for t in (tools_raw or "").split(",") if t.strip()]
    # Body is everything after the header block (first blank line after header)
    body_start = 0
    for i, line in enumerate(lines):
        if i > 2 and line.strip() == "":
            body_start = i + 1
            break
    body = "\n".join(lines[body_start:]).strip()

    return {
        "name": name,
        "description": description,
        "tools": tools,
        "body": body,
        "path": str(path),
    }


class SkillRegistry:
    """Loads and caches skill definitions from ~/.rika/skills/."""

    _instance: Optional["SkillRegistry"] = None

    def __init__(self) -> None:
        self._skills: Dict[str, Dict] = {}
        self._loaded = False

    @classmethod
    def get(cls) -> "SkillRegistry":
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    def load(self) -> None:
        """Scan ~/.rika/skills/ (including subdirectories) and load all skills."""
        # patch_skills_manager: rglob_load
        ensure_skills_bootstrapped()
        self._skills.clear()
        for path in sorted(_USER_SKILLS_DIR.rglob("*.md")):
            parsed = _parse_skill_file(path)
            if parsed:
                # Add category from parent directory name
                rel = path.relative_to(_USER_SKILLS_DIR)
                parsed["category"] = rel.parts[0] if len(rel.parts) > 1 else ""
                self._skills[parsed["name"]] = parsed
                logger.debug("skill_loaded", name=parsed["name"],
                             category=parsed.get("category", ""))
        self._loaded = True
        logger.info("skill_registry_loaded", count=len(self._skills))

    def reload(self) -> None:
        self._loaded = False
        self.load()

    def _ensure_loaded(self) -> None:
        if not self._loaded:
            self.load()

    def all_skills(self) -> List[Dict]:
        self._ensure_loaded()
        return list(self._skills.values())

    def get_skill(self, name: str) -> Optional[Dict]:
        self._ensure_loaded()
        return self._skills.get(name)

    def skill_list_prompt(self) -> str:
        """Compact skill list grouped by category for injection into every system prompt."""
        # patch_skills_manager: category_grouped_prompt
        self._ensure_loaded()
        if not self._skills:
            return ""
        by_cat: dict[str, list] = {}
        for s in self._skills.values():
            by_cat.setdefault(s.get("category", "") or "general", []).append(s)

        lines = [
            "\n\nAVAILABLE ON-DEMAND SKILLS — call use_skill(skill_name=\"name\") to activate:",
            "  Use skill_manager(action=\"list\") to browse the full skill tree.",
        ]
        for cat in sorted(by_cat.keys()):
            lines.append(f"  [{cat}]")
            for s in by_cat[cat]:
                tools_note = f" [{', '.join(s['tools'])}]" if s["tools"] else ""
                lines.append(f"    {s['name']}: {s['description'][:70]}{tools_note}")
        return "\n".join(lines)

    def activate_skill(self, name: str) -> str:
        """Return the skill body to inject as a tool result when use_skill is called."""
        self._ensure_loaded()
        skill = self._skills.get(name)
        if skill is None:
            available = ", ".join(self._skills.keys()) or "none"
            return f"Skill '{name}' not found. Available skills: {available}"
        return (
            f"SKILL ACTIVATED: {skill['name']}\n"
            f"Description: {skill['description']}\n\n"
            f"{skill['body']}"
        )
