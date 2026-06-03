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
                logger.warning(
                    "qdrant_not_installed",
                    detail="pip install 'qdrant-client[fastembed]'",
                )
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

    def _embed_passage(self, text: str) -> list[float]:
        """Embed a document passage using the client's local fastembed model."""
        model = self.client._get_or_init_model(
            model_name=self.client.embedding_model_name,
            deprecated=True,
        )
        return list(model.passage_embed([text]))[0].tolist()

    def _embed_query(self, text: str) -> list[float]:
        """Embed a search query using the client's local fastembed model."""
        model = self.client._get_or_init_model(
            model_name=self.client.embedding_model_name,
            deprecated=True,
        )
        return list(model.query_embed(text))[0].tolist()

    async def add_memory(
        self,
        user_id: int,
        text: str,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        if self.client is None:
            return
        loop = asyncio.get_running_loop()
        payload = {**(metadata or {}), "user_id": user_id, "text": text}
        try:
            def _upsert() -> None:
                vector = self._embed_passage(text)
                self.client.upsert(
                    collection_name=self.collection_name,
                    points=[
                        qdrant_models.PointStruct(
                            id=abs(hash(text)) % (2 ** 63),
                            vector=vector,
                            payload=payload,
                        )
                    ],
                )
            await loop.run_in_executor(None, _upsert)
        except Exception as exc:
            global _VECTOR_DISABLED
            if any(x in str(exc) for x in ("NO_SUCH", "onnx", "model_optimized", "fastembed")):
                if not _VECTOR_DISABLED:
                    _VECTOR_DISABLED = True
                    logger.warning(
                        "vector_store_disabled",
                        reason="ONNX model missing — pip install fastembed",
                    )
            else:
                logger.error("vector_add_failed", error=str(exc))

    async def search_memories(
        self, user_id: int, query: str, limit: int = 5
    ) -> List[Dict[str, Any]]:
        if self.client is None:
            return []
        loop = asyncio.get_running_loop()
        try:
            def _query() -> list:
                vector = self._embed_query(query)
                result = self.client.query_points(
                    collection_name=self.collection_name,
                    query=vector,
                    using=self.client.get_vector_field_name(),
                    query_filter=qdrant_models.Filter(
                        must=[
                            qdrant_models.FieldCondition(
                                key="user_id",
                                match=qdrant_models.MatchValue(value=user_id),
                            )
                        ]
                    ),
                    limit=limit,
                    with_payload=True,
                )
                return result.points if hasattr(result, "points") else result

            points = await loop.run_in_executor(None, _query)
            return [
                {
                    "text": r.payload.get("text", "") if r.payload else "",
                    "score": r.score,
                    "metadata": {
                        k: v for k, v in (r.payload or {}).items() if k != "text"
                    },
                }
                for r in points
            ]
        except Exception as exc:
            global _VECTOR_DISABLED
            if any(x in str(exc) for x in ("NO_SUCH", "onnx", "model_optimized", "fastembed")):
                if not _VECTOR_DISABLED:
                    _VECTOR_DISABLED = True
                    logger.warning(
                        "vector_store_disabled",
                        reason="ONNX model missing",
                    )
            else:
                logger.error("vector_search_failed", error=str(exc))
            return []


vector_store = VectorStore()
