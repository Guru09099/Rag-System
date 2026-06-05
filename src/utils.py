from pathlib import Path
from typing import Dict, List

from pypdf import PdfReader


def read_documents(folder_path: str) -> List[Dict[str, str]]:
    folder = Path(folder_path)
    if not folder.exists():
        folder.mkdir(parents=True, exist_ok=True)
        return []

    documents: List[Dict[str, str]] = []
    for file_path in sorted(folder.iterdir()):
        if file_path.suffix.lower() == ".pdf":
            try:
                reader = PdfReader(str(file_path))
                text_chunks = []
                for page in reader.pages:
                    content = page.extract_text()
                    if content:
                        text_chunks.append(content)
                documents.append({"source": file_path.name, "text": "\n".join(text_chunks)})
            except Exception as exc:
                print(f"Failed to read PDF {file_path.name}: {exc}")
        elif file_path.suffix.lower() == ".txt":
            try:
                with file_path.open("r", encoding="utf-8") as f:
                    documents.append({"source": file_path.name, "text": f.read()})
            except Exception as exc:
                print(f"Failed to read TXT {file_path.name}: {exc}")

    return documents
