"""Persistent task state — pause/resume across bot restarts.

State stored as JSON in ~/.rika/data/paused/<task_id>.json
"""
from __future__ import annotations
import json, uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional
from src.utils.logger import logger

_DIR = Path.home() / ".rika" / "data" / "paused"


@dataclass
class PausedTask:
    task_id: str; user_id: int; channel_id: str; platform: str
    original_message: str; thought_history: List[Dict]
    plan: Optional[List[Dict]]; current_step_idx: int
    agent_results: Dict; session_usage: Dict
    paused_at: str; pause_reason: str
    system_prompt: str = ""; summary: Optional[str] = None

    @classmethod
    def create(cls, *, user_id, channel_id, platform, original_message,
               thought_history, plan, current_step_idx, agent_results,
               session_usage, system_prompt="", summary=None,
               reason="quota_exhausted") -> "PausedTask":
        return cls(
            task_id=str(uuid.uuid4())[:8], user_id=user_id,
            channel_id=channel_id, platform=platform,
            original_message=original_message, thought_history=thought_history,
            plan=plan, current_step_idx=current_step_idx,
            agent_results=agent_results, session_usage=session_usage,
            paused_at=datetime.now(timezone.utc).isoformat(),
            pause_reason=reason, system_prompt=system_prompt, summary=summary,
        )

    def to_dict(self) -> Dict:
        return {k: getattr(self, k) for k in (
            "task_id","user_id","channel_id","platform","original_message",
            "thought_history","plan","current_step_idx","agent_results",
            "session_usage","paused_at","pause_reason","system_prompt","summary")}

    @classmethod
    def from_dict(cls, d: Dict) -> "PausedTask":
        return cls(**{k: d.get(k, None if k in ("plan","summary") else
                      ([] if k in ("thought_history",) else
                       ({} if k in ("agent_results","session_usage") else
                        (0 if k == "current_step_idx" else ""))))
                     for k in ("task_id","user_id","channel_id","platform",
                               "original_message","thought_history","plan",
                               "current_step_idx","agent_results","session_usage",
                               "paused_at","pause_reason","system_prompt","summary")})


class TaskStore:
    _instance: Optional["TaskStore"] = None

    @classmethod
    def get(cls) -> "TaskStore":
        if cls._instance is None: cls._instance = cls()
        return cls._instance

    def save(self, task: PausedTask) -> Path:
        _DIR.mkdir(parents=True, exist_ok=True)
        p = _DIR / f"{task.task_id}.json"
        p.write_text(json.dumps(task.to_dict(), indent=2))
        logger.info("task_paused", task_id=task.task_id, reason=task.pause_reason)
        return p

    def load(self, task_id: str) -> Optional[PausedTask]:
        p = _DIR / f"{task_id}.json"
        if not p.exists(): return None
        try: return PausedTask.from_dict(json.loads(p.read_text()))
        except Exception as e:
            logger.error("task_load_failed", task_id=task_id, error=str(e)); return None

    def delete(self, task_id: str) -> None:
        p = _DIR / f"{task_id}.json"
        if p.exists(): p.unlink(); logger.info("task_deleted", task_id=task_id)

    def load_pending(self, platform: Optional[str] = None) -> List[PausedTask]:
        if not _DIR.exists(): return []
        tasks = []
        for p in sorted(_DIR.glob("*.json")):
            try:
                t = PausedTask.from_dict(json.loads(p.read_text()))
                if platform is None or t.platform == platform: tasks.append(t)
            except Exception as e:
                logger.warning("task_load_skipped", path=str(p), error=str(e))
        return tasks

    def load_for_user(self, user_id: int) -> List[PausedTask]:
        return [t for t in self.load_pending() if t.user_id == user_id]
