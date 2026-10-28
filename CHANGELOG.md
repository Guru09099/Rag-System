# Project Change Log

## Date: 2026-10-28

### Summary of this change
Completed full codebase review, architectural consolidation, security hardening, and removal of duplicate legacy files to prepare the codebase for an automated evaluation (evals) suite.

### What changed

#### 1) Deleted duplicate and unwanted legacy prototype files
Permanently removed the old Sanic prototype and redundant scripts that conflicted with the active FastAPI app:
- `src/serve.py` (legacy Sanic server)
- `src/build_index.py` (redundant character-slicing index script)
- `src/retrieve.py` (redundant retrieval script relying on `chunks.pkl`)
- `src/generate.py` (synchronous urllib caller)
- `src/list_models.py` (one-off test script)
- `static/index.html` (legacy single-page UI replaced by React frontend)

#### 2) Eliminated Insecure Deserialization (Pickle -> JSON)
- Migrated metadata persistence in `src/index_manager.py` from `pickle` to safe, human-readable JSON (`chunks_metadata.json`).
- Completely eliminated Python arbitrary bytecode execution risks (RCE).

#### 3) Secured File Ingestion & API Routes
- Added a 25MB maximum upload file limit in `app.py`.
- Switched to streamed chunked file writes to prevent Out-Of-Memory (OOM) Denial-of-Service attacks.
- Added regex-based filename sanitization preventing directory traversal and Windows-reserved device names (`CON`, `PRN`, `NUL`, etc.).
- Fixed CORS misconfiguration (`allow_origins=["*"]` with `allow_credentials=True` replaced with explicit origins from `.env`).

#### 4) Concurrency Protection & Thread-Safety
- Added `threading.Lock()` across `VectorIndexStore` to serialize vector building and search calls, preventing FAISS index file corruption.
- Made embedding model initialization thread-safe.

#### 5) Upgraded Chunking & Semantic Integrity
- Replaced naive character slicing with boundary-aware paragraph and sentence chunking, preventing words or sentences from being cut in half.

#### 6) Prompt Injection Mitigation & Reusable HTTP Client
- Added structured `<context>...</context>` XML tagging and explicit system instructions in `src/llm_client.py`.
- Reused a pooled `httpx.AsyncClient` across requests with cleanup in FastAPI's modern `lifespan` handler.

#### 7) Frontend Build & Static Serving Fixes
- Mounted `/assets` directory in `app.py` so the React build from `frontend/dist` renders correctly when visiting `http://localhost:8000`.
- Resolved oxlint warnings in `frontend/src/App.jsx`.

#### 8) Modularized for Evals
- Ingestion (`src/utils.py`), retrieval (`src/index_manager.py`), and generation (`src/llm_client.py`) can now be imported and benchmarked independently for retrieval precision/recall and answer faithfulness.

---

## Date: 2026-10-24

### Summary of this change
Initial consolidation plan to focus on FastAPI.
