from __future__ import annotations

import asyncio
from typing import Any, Dict, List, Optional

from src.utils.logger import logger

_VECTOR_DISABLED = False  # True after first ONNX/fastembed failure

try:
    from qdrant_client import QdrantClient
    from qdrant_client.http import models as qdrant_models
    HAS_QDRANT = True
except ImportError:
    HAS_QDRANT = False


class VectorStore:
    """Semantic memory store backed by a local Qdrant instance."""

    _instance: Optional[VectorStore] = None

    def __new__(cls) -> VectorStore:
        if cls._instance is None:
            obj = super().__new__(cls)
            obj.client = None
            obj.collection_name = "collective_unconscious"
            if HAS_QDRANT:
                try:
                    obj.client = QdrantClient(path="./data/vector_db")
                    obj._ensure_collection()
                except Exception as exc:
                    logger.error("qdrant_init_failed", error=str(exc))
            else:
                logger.warning("qdrant_not_installed", detail="pip install 'qdrant-client[fastembed]'")
            cls._instance = obj
        return cls._instance

    def _ensure_collection(self) -> None:
        global _VECTOR_DISABLED
        if self.client is None or _VECTOR_DISABLED:
            return
        try:
            self.client.get_collection(self.collection_name)
        except Exception:
            try:
                self.client.create_collection(
                    collection_name=self.collection_name,
                    vectors_config=self.client.get_fastembed_vector_params(),
                )
                logger.info("vector_collection_created", name=self.collection_name)
            except Exception as exc:
                logger.error("vector_collection_creation_failed", error=str(exc))

    async def add_memory(
        self,
        user_id: int,
        text: str,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        if self.client is None:
            return
        loop = asyncio.get_running_loop()  # fixed: was get_event_loop()
        payload = {**(metadata or {}), "user_id": user_id}
        try:
            await loop.run_in_executor(
                None,
                lambda: self.client.add(
                    collection_name=self.collection_name,
                    documents=[text],
                    metadata=[payload],
                ),
            )
        except Exception as exc:
            if any(x in str(exc) for x in ("NO_SUCH", "onnx", "model_optimized", "fastembed")):
                global _VECTOR_DISABLED
                if not _VECTOR_DISABLED:
                    _VECTOR_DISABLED = True
                    logger.warning("vector_store_disabled", reason="ONNX model missing — pip install fastembed")
            else:
                logger.error("vector_add_failed", error=str(exc))

    async def search_memories(
        self, user_id: int, query: str, limit: int = 5
    ) -> List[Dict[str, Any]]:
        if self.client is None:
            return []
        loop = asyncio.get_running_loop()  # fixed: was get_event_loop()
        try:
            results = await loop.run_in_executor(
                None,
                lambda: self.client.query(
                    collection_name=self.collection_name,
                    query_text=query,
                    query_filter=qdrant_models.Filter(
                        must=[
                            qdrant_models.FieldCondition(
                                key="user_id",
                                match=qdrant_models.MatchValue(value=user_id),
                            )
                        ]
                    ),
                    limit=limit,
                ),
            )
            return [
                {"text": r.document, "score": r.score, "metadata": r.metadata}
                for r in results
            ]
        except Exception as exc:
            if any(x in str(exc) for x in ("NO_SUCH", "onnx", "model_optimized", "fastembed")):
                global _VECTOR_DISABLED
                if not _VECTOR_DISABLED:
                    _VECTOR_DISABLED = True
                    logger.warning("vector_store_disabled", reason="ONNX model missing")
            else:
                logger.error("vector_search_failed", error=str(exc))
            return []


vector_store = VectorStore()
