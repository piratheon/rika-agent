import asyncio
from typing import Any


class BaseAgent:
    def __init__(self, spec: Any, bubble=None):
        self.spec = spec
        self.bubble = bubble
        self.status = "pending"

    async def run(self, context: dict) -> dict:
        """Run the agent. Must be overridden by subclasses."""
        raise NotImplementedError(f"{self.__class__.__name__} must implement run()")
