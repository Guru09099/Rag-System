import os
import pickle
from pathlib import Path
from typing import Any, Dict, List

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

from src.utils import read_documents


class VectorIndexStore:
    def __init__(
        self,
        data_dir: str = "data",
        index_path: str = "vector.index",
        metadata_path: str = "chunks_metadata.pkl",
        model_name: str = "all-MiniLM-L6-v2",
    ):
        self.data_dir = Path(data_dir)
        self.index_path = Path(index_path)
        self.metadata_path = Path(metadata_path)
        self.model_name = model_name
        self.index: Any = None
        self.chunks: List[str] = []
        self.metadata: List[Dict[str, Any]] = []
        self.embedder: SentenceTransformer | None = None

    def _ensure_embedder(self):
        if self.embedder is None:
            self.embedder = SentenceTransformer(self.model_name)

    def _save_index(self) -> None:
        faiss.write_index(self.index, str(self.index_path))
        with open(self.metadata_path, "wb") as f:
            pickle.dump({"chunks": self.chunks, "metadata": self.metadata}, f)

    def _load_index_data(self) -> None:
        with open(self.metadata_path, "rb") as f:
            data = pickle.load(f)
        self.chunks = data.get("chunks", [])
        self.metadata = data.get("metadata", [])

    def is_ready(self) -> bool:
        return self.index_path.exists() and self.metadata_path.exists() and bool(self.chunks)

    def document_count(self) -> int:
        return len({item["source"] for item in self.metadata})

    def load_index(self) -> None:
        if not self.index_path.exists() or not self.metadata_path.exists():
            raise FileNotFoundError("Vector index or metadata file is missing.")
        self.index = faiss.read_index(str(self.index_path))
        self._load_index_data()

    def build_index(self) -> bool:
        documents = read_documents(str(self.data_dir))
        if not documents:
            return False

        self._ensure_embedder()
        self.chunks = []
        self.metadata = []

        chunk_size = 800
        overlap = 160

        for document in documents:
            raw_text = document["text"].strip()
            if not raw_text:
                continue
            source = document["source"]
            for start in range(0, len(raw_text), chunk_size - overlap):
                chunk = raw_text[start : start + chunk_size]
                if len(chunk) < 50:
                    continue
                self.chunks.append(chunk)
                self.metadata.append(
                    {
                        "source": source,
                        "start": start,
                        "end": min(start + chunk_size, len(raw_text)),
                    }
                )

        if not self.chunks:
            return False

        embeddings = self.embedder.encode(self.chunks, convert_to_numpy=True)
        if embeddings.dtype != np.float32:
            embeddings = embeddings.astype(np.float32)

        dimension = embeddings.shape[1]
        self.index = faiss.IndexFlatL2(dimension)
        self.index.add(embeddings)

        os.makedirs(self.data_dir, exist_ok=True)
        self._save_index()
        return True

    def search(self, query: str, top_k: int = 3) -> List[Dict[str, Any]]:
        if self.index is None:
            self.load_index()

        self._ensure_embedder()
        query_vector = self.embedder.encode([query], convert_to_numpy=True)
        if query_vector.dtype != np.float32:
            query_vector = query_vector.astype(np.float32)

        distances, indices = self.index.search(query_vector, top_k)
        results: List[Dict[str, Any]] = []
        for score, idx in zip(distances[0], indices[0]):
            if idx == -1 or idx >= len(self.chunks):
                continue
            results.append(
                {
                    "source": self.metadata[idx]["source"],
                    "text": self.chunks[idx],
                    "score": float(score),
                }
            )
        return results
