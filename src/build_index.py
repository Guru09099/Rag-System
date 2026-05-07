import os
import pickle

import faiss
import numpy as np
from pypdf import PdfReader
from sentence_transformers import SentenceTransformer


def get_document_text(folder_path):
    text_corpus = []
    print(f"Scanning '{folder_path}' for PDF and TXT files...")

    if not os.path.exists(folder_path):
        os.makedirs(folder_path)
        print(f"   Created folder '{folder_path}'. Please put PDF or TXT files here.")
        return ""

    for filename in os.listdir(folder_path):
        file_path = os.path.join(folder_path, filename)
        try:
            if filename.lower().endswith(".pdf"):
                print(f"   Reading PDF: {filename}...")
                reader = PdfReader(file_path)
                page_text = ""
                for page in reader.pages:
                    content = page.extract_text()
                    if content:
                        page_text += content + "\n"
                text_corpus.append(page_text)
            elif filename.lower().endswith(".txt"):
                print(f"   Reading TXT: {filename}...")
                with open(file_path, "r", encoding="utf-8") as f:
                    text_corpus.append(f.read())
        except Exception as e:
            print(f"   Error reading {filename}: {e}")

    return "\n".join(text_corpus)


def create_vector_db():
    print("1. Processing documents...")
    full_text = get_document_text("./data")

    if not full_text:
        print("No text found. Add .pdf or .txt files to the 'data' folder.")
        return False

    print("2. Loading embedding model...")
    try:
        embedder = SentenceTransformer("all-MiniLM-L6-v2")
    except Exception as e:
        print(f"Could not load embedding model: {e}")
        print(
            "Check your internet connection/DNS, or run once while online "
            "so the model is cached."
        )
        return False

    # Chunking logic (Overlap helps preserve context across cuts)
    chunk_size = 500
    overlap = 50
    chunks = []

    for i in range(0, len(full_text), chunk_size - overlap):
        chunk = full_text[i : i + chunk_size]
        # Only keep chunks that have actual content
        if len(chunk) > 50:
            chunks.append(chunk)

    print(f"   Created {len(chunks)} chunks from your documents.")

    print("3. Creating embeddings...")
    embeddings = embedder.encode(chunks)
    embeddings = np.array(embeddings).astype("float32")

    print("4. Building FAISS index...")
    dimension = embeddings.shape[1]
    index = faiss.IndexFlatL2(dimension)
    index.add(embeddings)

    print("5. Saving index and metadata...")
    faiss.write_index(index, "vector.index")

    with open("chunks.pkl", "wb") as f:
        pickle.dump(chunks, f)

    print("Done. System is ready to serve.")
    return True


if __name__ == "__main__":
    create_vector_db()
