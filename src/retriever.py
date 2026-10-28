"""FAISS Vector Index Store and semantic retrieval module."""

import json
import logging
from pathlib import Path
from threading import RLock
from typing import Any, Dict, List, Optional

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

from src.ingestion import chunk_text, load_documents

logger = logging.getLogger("rag.retriever")


class VectorIndexStore:
    """Manages document embeddings, persistent FAISS indexing, and similarity search."""

    def __init__(
        self,
        data_dir: str = "data",
        index_path: str = "vector.index",
        metadata_path: str = "chunks_metadata.json",
        embed_model: str = "all-MiniLM-L6-v2",
    ):
        self.data_dir = Path(data_dir)
        self.index_path = Path(index_path)
        self.metadata_path = Path(metadata_path)
        self.embed_model_name = embed_model

        self.index: Any = None
        self.chunks: List[str] = []
        self.metadata: List[Dict[str, Any]] = []

        self._embedder: Optional[SentenceTransformer] = None
        self._lock = RLock()

    @property
    def embedder(self) -> SentenceTransformer:
        """Thread-safe lazy initialization of the embedding model."""
        if self._embedder is None:
            with self._lock:
                if self._embedder is None:
                    self._embedder = SentenceTransformer(self.embed_model_name)
        return self._embedder

    def is_ready(self) -> bool:
        """Checks if the FAISS index and metadata exist and are loaded."""
        with self._lock:
            return (
                self.index_path.exists()
                and self.metadata_path.exists()
                and self.index is not None
                and bool(self.chunks)
            )

    def load_index(self) -> None:
        """Loads persistent FAISS index and JSON chunk metadata into memory."""
        with self._lock:
            if not self.index_path.exists() or not self.metadata_path.exists():
                raise FileNotFoundError("FAISS index or metadata file not found.")
            self.index = faiss.read_index(str(self.index_path))
            with open(self.metadata_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            self.chunks = data.get("chunks", [])
            self.metadata = data.get("metadata", [])

    def build_index(self) -> bool:
        """Ingests documents from data/, builds FAISS index, and persists safely as JSON."""
        with self._lock:
            docs = load_documents(str(self.data_dir))
            if not docs:
                return False

            new_chunks: List[str] = []
            new_metadata: List[Dict[str, Any]] = []

            for doc in docs:
                chunks = chunk_text(doc["text"])
                for c in chunks:
                    new_chunks.append(c)
                    new_metadata.append({"source": doc["source"]})

            if not new_chunks:
                return False

            embeddings = self.embedder.encode(new_chunks, convert_to_numpy=True)
            if embeddings.dtype != np.float32:
                embeddings = embeddings.astype(np.float32)

            dimension = embeddings.shape[1]
            new_index = faiss.IndexFlatL2(dimension)
            new_index.add(embeddings)

            # Persist FAISS and metadata JSON safely
            self.data_dir.mkdir(parents=True, exist_ok=True)
            faiss.write_index(new_index, str(self.index_path))

            temp_meta = self.metadata_path.with_suffix(".tmp")
            with open(temp_meta, "w", encoding="utf-8") as f:
                json.dump({"chunks": new_chunks, "metadata": new_metadata}, f, ensure_ascii=False)
            temp_meta.replace(self.metadata_path)

            self.index = new_index
            self.chunks = new_chunks
            self.metadata = new_metadata
            return True

    def search(self, query: str, top_k: int = 3) -> List[Dict[str, Any]]:
        """Encodes query and retrieves top-k most relevant chunks using FAISS."""
        if self.index is None:
            self.load_index()

        query_vec = self.embedder.encode([query], convert_to_numpy=True)
        if query_vec.dtype != np.float32:
            query_vec = query_vec.astype(np.float32)

        distances, indices = self.index.search(query_vec, top_k)
        results: List[Dict[str, Any]] = []

        with self._lock:
            for dist, idx in zip(distances[0], indices[0]):
                if idx == -1 or idx >= len(self.chunks):
                    continue
                results.append(
                    {
                        "source": self.metadata[idx]["source"],
                        "text": self.chunks[idx],
                        "distance": float(dist),
                    }
                )
        return results
