"""LLM generation client interfacing with Ollama."""

import logging
import os
from typing import Optional

import httpx

logger = logging.getLogger("rag.generator")


class OllamaGenerator:
    """Manages prompt formatting and LLM answer generation using local Ollama."""

    def __init__(self, host: Optional[str] = None, model: Optional[str] = None):
        self.host = (host or os.getenv("OLLAMA_HOST", "http://127.0.0.1:11434")).rstrip("/")
        self.model = model or os.getenv("OLLAMA_MODEL", "llama3.2")

    def generate(self, query: str, context: str) -> str:
        """Synthesizes a response strictly grounded in retrieved context."""
        fallback = "I cannot find the answer in the provided documents."
        prompt = (
            "You are a helpful and accurate document assistant. "
            f"Use ONLY the following context to answer the question. If missing, say: '{fallback}'\n\n"
            f"<context>\n{context}\n</context>\n\n"
            f"Question: {query.strip()}\n\nAnswer:"
        )

        payload = {
            "model": self.model,
            "prompt": prompt,
            "stream": False,
            "options": {"temperature": 0.2, "num_predict": 512},
        }

        try:
            with httpx.Client(timeout=30.0) as client:
                res = client.post(f"{self.host}/api/generate", json=payload)
                if res.status_code == 200:
                    return res.json().get("response", "").strip()
                return f"Ollama error: HTTP {res.status_code}"
        except httpx.ConnectError:
            return f"Ollama is offline at {self.host}. Please run 'ollama serve'."
        except Exception as exc:
            return f"Generation error: {exc}"
