# RAG Web App

A simple retrieval-augmented generation (RAG) web application that ingests PDF and TXT documents, builds a vector index with FAISS, retrieves relevant document chunks for a user query, and generates answers using a local Ollama model.

## Features

- Upload `.pdf` and `.txt` documents from the browser
- Extract document text and split into overlapping chunks
- Embed text chunks with `sentence-transformers` (`all-MiniLM-L6-v2`)
- Store embeddings in a FAISS vector index
- Retrieve top matching chunks for a user query
- Send query + context to Ollama for answer generation
- Display generated answer and source context in the UI

## Repository Structure

- `src/build_index.py` - build the FAISS index from files in `data/`
- `src/retrieve.py` - search the FAISS index for relevant chunks
- `src/generate.py` - call Ollama HTTP API to generate answers
- `src/serve.py` - Sanic web server for upload and query endpoints
- `src/utils.py` - utilities for reading documents and creating the index
- `src/list_models.py` - helper script to list available Ollama models
- `static/index.html` - browser UI for upload and query
- `requirements.txt` - Python dependencies
- `Dockerfile` - optional container build file

## Requirements

- Python 3.10+ or compatible
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

### Option 1: Build index manually

```powershell
python src\build_index.py
```

### Option 2: Start the web server

```powershell
python src\serve.py
```

Then open `http://localhost:8000` in your browser.

## Usage

- Upload a `.pdf` or `.txt` file using the browser UI.
- The server saves the file to `data/`, rebuilds the index, and stores `vector.index` and `chunks.pkl`.
- Enter a question and click **Ask**.
- The app retrieves relevant text chunks from FAISS and calls Ollama to generate an answer.
- The generated answer and the source context are displayed.

## Environment Variables

- `OLLAMA_HOST` - URL for the Ollama API (default: `http://127.0.0.1:11434`)
- `OLLAMA_MODEL` - Ollama model to use (default: `llama3.2`)

## Notes

- The project uses synchronous retrieval and generation in `src/serve.py`, which is fine for a demo but may block under heavier load.
- Document chunking is done with fixed-size slices and 50-character overlap to preserve context.
- If the `data/` folder is empty, `src/build_index.py` will create it and prompt you to add files.

## Troubleshooting

- If the app cannot connect to Ollama, verify Ollama is running and the model is installed.
- If `vector.index` is missing, run `python src\build_index.py` or upload a document through the web UI.
- For local development, ensure `requirements.txt` is installed and the correct Python environment is active.
