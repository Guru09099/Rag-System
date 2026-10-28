"""Unified RAGPipeline integrating ingestion, retrieval, and LLM generation."""

from typing import Any, Dict, List, Optional

from src.generator import OllamaGenerator
from src.retriever import VectorIndexStore


class RAGPipeline:
    """Cohesive pipeline managing document search and LLM answer generation."""

    def __init__(
        self,
        data_dir: str = "data",
        index_path: str = "vector.index",
        metadata_path: str = "chunks_metadata.json",
        embed_model: str = "all-MiniLM-L6-v2",
        llm_model: Optional[str] = None,
        ollama_host: Optional[str] = None,
    ):
        self.store = VectorIndexStore(
            data_dir=data_dir,
            index_path=index_path,
            metadata_path=metadata_path,
            embed_model=embed_model,
        )
        self.generator = OllamaGenerator(host=ollama_host, model=llm_model)

    @property
    def chunks(self) -> List[str]:
        return self.store.chunks

    def is_ready(self) -> bool:
        return self.store.is_ready()

    def load_index(self) -> None:
        self.store.load_index()

    def build_index(self) -> bool:
        return self.store.build_index()

    def retrieve(self, query: str, top_k: int = 3) -> List[Dict[str, Any]]:
        return self.store.search(query, top_k=top_k)

    def generate(self, query: str, context: str) -> str:
        return self.generator.generate(query, context)

    def ask(self, query: str, top_k: int = 3) -> Dict[str, Any]:
        """Convenience method combining retrieval and generation."""
        sources = self.retrieve(query, top_k=top_k)
        context = "\n\n".join(s["text"] for s in sources)
        answer = self.generate(query, context)
        return {"query": query, "answer": answer, "sources": sources}
