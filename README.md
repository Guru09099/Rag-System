# FastAPI RAG Document Assistant

A scalable retrieval-augmented generation (RAG) system for building a document knowledge assistant. This version uses FastAPI for a more modern API surface and supports PDF/TXT ingestion, FAISS retrieval, and Ollama generation.

## Application

This project is designed as a document search assistant for internal reports, manuals, and knowledge base files. Users upload documents, the system builds a vector index, and queries return natural-language answers with source citations.

## Repository Structure

- `app.py` - FastAPI application entrypoint
- `src/index_manager.py` - vector index lifecycle and search logic
- `src/llm_client.py` - Ollama generation client
- `src/schemas.py` - request and response models
- `src/utils.py` - document loading utilities
- `src/build_index.py` - legacy index builder
- `static/index.html` - browser UI
- `requirements.txt` - Python dependencies
- `Dockerfile` - container build file

## Requirements

- Python 3.11+ or compatible
- Ollama installed and running locally
- Ollama model downloaded, e.g. `llama3.2`

## Setup

1. Create and activate a Python environment:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

2. Install requirements:

```powershell
python -m pip install -r requirements.txt
```

3. Start Ollama and install the model if not already available:

```powershell
ollama pull llama3.2
ollama serve
```

4. Add `.pdf` or `.txt` documents to the `data/` folder.

## Running the App

```powershell
python app.py
```

Then open `http://localhost:8000` in your browser.

## API Endpoints

- `GET /health` — health check and index readiness
- `POST /upload` — upload a document and rebuild the index in background
- `POST /reindex` — rebuild the entire index from the `data/` directory
- `POST /query` — send a query and receive a RAG-generated answer with sources

## Environment Variables

- `OLLAMA_HOST` — Ollama API host (default: `http://127.0.0.1:11434`)
- `OLLAMA_MODEL` — Ollama model name (default: `llama3.2`)
- `RAG_DATA_DIR` — folder for source documents (default: `data`)
- `RAG_INDEX_PATH` — path for FAISS index file (default: `vector.index`)
- `RAG_METADATA_PATH` — path for chunk metadata file (default: `chunks_metadata.pkl`)

## Notes

- The FastAPI app uses asynchronous request handling with threadpool execution for CPU-bound search and generation.
- Document chunks include source metadata so returned answers can cite where text came from.
- Uploads are accepted immediately and index rebuilding occurs in the background for faster responsiveness.
