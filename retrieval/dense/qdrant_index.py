"""
Qdrant vector index manager.

Architecture-confirmed components:
- Vector store: Qdrant (Audit §5, §7)
- Embedding model: bge-m3, 1024-dim dense vectors
- Distance metric: COSINE (standard for normalized embeddings)

Production payload schema:
  - passage_id: str (canonical content-hash ID)
  - text: str (normalized passage text)
  - lang: str (ISO 639-1 language code)
  - representation_type: str ("passage", "sentence", etc.)
  - parent_id: str (canonical passage_id of parent, or self)
  - text_length: int (character count)

FORBIDDEN from production payload (Audit §5, freeze decision 3):
  - is_selected
  - Answer
  - Eng_Answer
  - source_query_ids
  - query_id (as passage metadata)

This is enforced structurally: the PayloadSchema pydantic model uses
extra='forbid', and an automated test verifies no forbidden fields
can be smuggled in.
"""
from __future__ import annotations

import structlog
from qdrant_client import QdrantClient  # type: ignore[import-untyped]
from qdrant_client.models import (
    FieldCondition,
    Filter,
    MatchValue,
    PointStruct,
    VectorParams,
)  # type: ignore[import-untyped]

from app.models.retrieval import FORBIDDEN_EVALUATION_FIELDS
from embeddings.bge_m3 import BGE_M3_DENSE_DIMENSION
from ingestion.representation.base import Representation

logger = structlog.get_logger(__name__)

# Production payload field names — the ONLY fields allowed in Qdrant payloads
PRODUCTION_PAYLOAD_FIELDS: frozenset[str] = frozenset({
    "passage_id",
    "text",
    "lang",
    "representation_type",
    "parent_id",
    "text_length",
})


def representation_payload(repr: Representation) -> dict:
    """
    Build the production payload dict from a Representation.

    Returns a dict with 'id' and 'payload' keys suitable for PointStruct construction.
    Raises ValueError if any forbidden evaluation field is accidentally
    present in the representation.
    """
    # Validate no forbidden fields leak through
    repr_dict = repr.__dict__
    leaked = set(repr_dict.keys()) & FORBIDDEN_EVALUATION_FIELDS
    if leaked:
        raise ValueError(
            f"Representation contains forbidden evaluation fields: {leaked}. "
            "These must never enter the production Qdrant index."
        )

    # Use first 16 chars of hex representation_id as Qdrant point UUID
    # (Qdrant accepts int or UUID-format strings)
    point_id = int(repr.representation_id[:16], 16)

    payload = {
        "passage_id": repr.passage_id,
        "text": repr.text,
        "lang": repr.lang,
        "representation_type": repr.representation_type,
        "parent_id": repr.parent_id,
        "text_length": repr.text_length,
    }

    return {"id": point_id, "payload": payload}


class QdrantIndexManager:
    """
    Manages a Qdrant collection for dense retrieval.

    Usage:
        manager = QdrantIndexManager()  # in-memory
        # or
        manager = QdrantIndexManager(path="data/qdrant_storage")

        manager.create_collection()
        manager.upsert_passages(representations, vectors)
        results = manager.search(query_vector, top_k=10, lang_filter="hi")
    """

    def __init__(
        self,
        collection_name: str = "rag_corpus",
        vector_dimension: int = BGE_M3_DENSE_DIMENSION,
        distance: str = "cosine",
        url: str | None = None,
        api_key: str | None = None,
        path: str | None = None,
    ) -> None:
        """
        Args:
            collection_name: Name of the Qdrant collection.
            vector_dimension: Embedding dimension (1024 for bge-m3).
            distance: Distance metric ("cosine", "dot", "euclid").
            url: Qdrant server URL. If None, uses in-memory client.
            api_key: Qdrant API key (for cloud instances).
            path: Local storage path for Qdrant.
        """
        self.collection_name = collection_name
        self.vector_dimension = vector_dimension
        self.distance = distance

        if url:
            self._client = QdrantClient(url=url, api_key=api_key)
        elif path:
            self._client = QdrantClient(path=path)
        else:
            self._client = QdrantClient(":memory:")

        logger.info(
            "qdrant_client_created",
            collection=collection_name,
            dimension=vector_dimension,
            distance=distance,
        )

    @property
    def client(self) -> QdrantClient:
        return self._client

    def create_collection(self, recreate: bool = False) -> None:
        """
        Create the Qdrant collection with the correct schema.

        Args:
            recreate: If True, delete and recreate the collection.
        """
        from qdrant_client.models import Distance as Dist

        dist_map = {"cosine": Dist.COSINE, "dot": Dist.DOT, "euclid": Dist.EUCLID}
        distance_value = dist_map.get(self.distance, Dist.COSINE)

        collections = self._client.get_collections().collections
        collection_names = [c.name for c in collections]

        if self.collection_name in collection_names:
            if recreate:
                logger.info("recreating_collection", collection=self.collection_name)
                self._client.delete_collection(self.collection_name)
            else:
                logger.info("collection_exists", collection=self.collection_name)
                return

        self._client.create_collection(
            collection_name=self.collection_name,
            vectors_config=VectorParams(
                size=self.vector_dimension,
                distance=distance_value,
            ),
        )

        # Create payload index on lang for efficient filtered search (Config 1/2)
        self._client.create_payload_index(
            collection_name=self.collection_name,
            field_name="lang",
            field_schema="keyword",
        )

        # Create payload index on passage_id for fast lookup
        self._client.create_payload_index(
            collection_name=self.collection_name,
            field_name="passage_id",
            field_schema="keyword",
        )

        # Create payload index on parent_id for parent/child lookups
        self._client.create_payload_index(
            collection_name=self.collection_name,
            field_name="parent_id",
            field_schema="keyword",
        )

        logger.info(
            "collection_created",
            collection=self.collection_name,
            dimension=self.vector_dimension,
            distance=self.distance,
        )

    def upsert_passages(
        self,
        representations: list[Representation],
        vectors: list[list[float]],
        batch_size: int = 100,
    ) -> int:
        """
        Upsert passages with their vectors into the collection.

        Args:
            representations: List of Representation objects.
            vectors: Corresponding embedding vectors.
            batch_size: Number of points per upsert batch.

        Returns:
            Number of points upserted.
        """
        assert len(representations) == len(vectors), (
            f"Representations ({len(representations)}) and vectors ({len(vectors)}) "
            "must have same length"
        )

        total = 0
        for start in range(0, len(representations), batch_size):
            batch_reprs = representations[start : start + batch_size]
            batch_vecs = vectors[start : start + batch_size]

            points = []
            for repr_obj, vec in zip(batch_reprs, batch_vecs, strict=False):
                point_id = int(repr_obj.representation_id[:16], 16)
                payload = {
                    "passage_id": repr_obj.passage_id,
                    "text": repr_obj.text,
                    "lang": repr_obj.lang,
                    "representation_type": repr_obj.representation_type,
                    "parent_id": repr_obj.parent_id,
                    "text_length": repr_obj.text_length,
                }
                points.append(PointStruct(id=point_id, vector=vec, payload=payload))

            self._client.upsert(
                collection_name=self.collection_name,
                points=points,
            )
            total += len(points)

        logger.info("upsert_complete", collection=self.collection_name, count=total)
        return total

    def search(
        self,
        query_vector: list[float],
        top_k: int = 10,
        lang_filter: str | None = None,
    ) -> list[dict]:
        """
        Search the index for nearest neighbors.

        Args:
            query_vector: Query embedding vector.
            top_k: Number of results to return.
            lang_filter: If set, filter to passages in this language (ISO 639-1).

        Returns:
            List of dicts with passage_id, score, text, lang, etc.
        """
        query_filter = None
        if lang_filter:
            query_filter = Filter(
                must=[FieldCondition(key="lang", match=MatchValue(value=lang_filter))]
            )

        results = self._client.query_points(
            collection_name=self.collection_name,
            query=query_vector,
            limit=top_k,
            query_filter=query_filter,
        )

        hits = []
        for point in results.points:
            payload = point.payload or {}
            hits.append({
                "passage_id": payload.get("passage_id", ""),
                "score": point.score,
                "text": payload.get("text", ""),
                "lang": payload.get("lang", ""),
                "representation_type": payload.get("representation_type", ""),
                "parent_id": payload.get("parent_id", ""),
                "point_id": point.id,
            })

        return hits

    def get_collection_info(self) -> dict:
        """Get collection metadata (vector count, config, etc.)."""
        try:
            info = self._client.get_collection(self.collection_name)
            result: dict = {"name": self.collection_name}
            # Safely extract attributes — local Qdrant may differ from server
            for attr in ("vectors_count", "points_count", "status", "optimizer_status"):
                if hasattr(info, attr):
                    val = getattr(info, attr)
                    result[attr] = str(val) if attr in ("status", "optimizer_status") else val
            return result
        except Exception:
            return {"name": self.collection_name, "error": "collection not found"}

    def delete_collection(self) -> None:
        """Delete the collection."""
        self._client.delete_collection(self.collection_name)
        logger.info("collection_deleted", collection=self.collection_name)
