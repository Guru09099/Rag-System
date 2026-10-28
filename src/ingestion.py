"""Document ingestion and text chunking utilities.

Supports extracting clean text from PDF and TXT documents,
and splitting text preserving paragraph and sentence boundaries.
"""

import logging
import re
from pathlib import Path
from typing import Dict, List

from pypdf import PdfReader

logger = logging.getLogger("rag.ingestion")


def load_documents(data_dir: str = "data") -> List[Dict[str, str]]:
    """Loads text from all valid .pdf and .txt files in the target directory."""
    folder = Path(data_dir)
    if not folder.exists():
        folder.mkdir(parents=True, exist_ok=True)
        return []

    documents = []
    for file_path in sorted(folder.iterdir()):
        if file_path.name.startswith(".") or not file_path.is_file():
            continue

        ext = file_path.suffix.lower()
        if ext == ".pdf":
            try:
                reader = PdfReader(str(file_path))
                text_pages = [page.extract_text() or "" for page in reader.pages]
                full_text = "\n".join(t.strip() for t in text_pages if t.strip())
                if full_text:
                    documents.append({"source": file_path.name, "text": full_text})
            except Exception as exc:
                logger.error("Failed to read PDF %s: %s", file_path.name, exc)
        elif ext == ".txt":
            try:
                with file_path.open("r", encoding="utf-8", errors="replace") as f:
                    content = f.read().strip()
                if content:
                    documents.append({"source": file_path.name, "text": content})
            except Exception as exc:
                logger.error("Failed to read TXT %s: %s", file_path.name, exc)

    return documents


def chunk_text(text: str, chunk_size: int = 600, overlap: int = 100) -> List[str]:
    """Splits text into chunks respecting paragraph and sentence boundaries.

    Ensures words and sentences are not split abruptly across chunk boundaries.
    """
    cleaned = re.sub(r"\r\n|\r", "\n", text).strip()
    if not cleaned:
        return []

    paragraphs = [p.strip() for p in re.split(r"\n\s*\n", cleaned) if p.strip()]
    chunks: List[str] = []
    current_chunk: List[str] = []
    current_len = 0

    for para in paragraphs:
        if len(para) > chunk_size:
            sentences = [s.strip() for s in re.split(r"(?<=[.!?])\s+", para) if s.strip()]
            for s in sentences:
                if current_len + len(s) + 1 > chunk_size and current_chunk:
                    chunks.append(" ".join(current_chunk))
                    current_chunk = [current_chunk[-1]] if len(current_chunk[-1]) <= overlap else []
                    current_len = sum(len(c) for c in current_chunk)
                current_chunk.append(s)
                current_len += len(s) + 1
        else:
            if current_len + len(para) + 2 > chunk_size and current_chunk:
                chunks.append("\n\n".join(current_chunk))
                current_chunk = [current_chunk[-1]] if len(current_chunk[-1]) <= overlap else []
                current_len = sum(len(c) for c in current_chunk)
            current_chunk.append(para)
            current_len += len(para) + 2

    if current_chunk:
        chunks.append("\n\n".join(current_chunk))

    return [c.strip() for c in chunks if len(c.strip()) >= 30]
