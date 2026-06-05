import os
from pathlib import Path

from fastapi import BackgroundTasks, FastAPI, File, HTTPException, UploadFile
from fastapi.concurrency import run_in_threadpool
from fastapi.responses import FileResponse
from fastapi.middleware.cors import CORSMiddleware

from src.index_manager import VectorIndexStore
from src.llm_client import OllamaClient
from src.schemas import QueryRequest, QueryResponse, UploadResponse

app = FastAPI(
    title="FastAPI RAG Document Assistant",
    description="A scalable retrieval-augmented generation system for document search and answer generation.",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

DATA_DIR = Path(os.getenv("RAG_DATA_DIR", "data"))
INDEX_PATH = Path(os.getenv("RAG_INDEX_PATH", "vector.index"))
METADATA_PATH = Path(os.getenv("RAG_METADATA_PATH", "chunks_metadata.pkl"))

index_store = VectorIndexStore(
    data_dir=str(DATA_DIR),
    index_path=str(INDEX_PATH),
    metadata_path=str(METADATA_PATH),
)
llm_client = OllamaClient()


@app.on_event("startup")
async def startup_event():
    try:
        index_store.load_index()
    except FileNotFoundError:
        # Index will be built on demand when documents arrive
        pass


@app.get("/", response_class=FileResponse)
async def root():
    return FileResponse("static/index.html")


@app.get("/health")
async def health():
    return {
        "status": "ok",
        "index_ready": index_store.is_ready(),
        "document_count": index_store.document_count(),
    }


@app.post("/upload", response_model=UploadResponse)
async def upload_document(background_tasks: BackgroundTasks, file: UploadFile = File(...)):
    if not file.filename.lower().endswith((".pdf", ".txt")):
        raise HTTPException(status_code=400, detail="Only .pdf and .txt files are allowed.")

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    file_path = DATA_DIR / Path(file.filename).name

    content = await file.read()
    file_path.write_bytes(content)

    background_tasks.add_task(index_store.build_index)
    return {"message": f"Uploaded {file.filename}. Rebuilding index in background."}


@app.post("/reindex", response_model=UploadResponse)
async def rebuild_index():
    success = await run_in_threadpool(index_store.build_index)
    if not success:
        raise HTTPException(status_code=500, detail="Reindexing failed. Check document files and logs.")
    return {"message": "Index built successfully."}


@app.post("/query", response_model=QueryResponse)
async def query(request: QueryRequest):
    if not request.query.strip():
        raise HTTPException(status_code=400, detail="Query text cannot be empty.")

    if not index_store.is_ready():
        raise HTTPException(status_code=503, detail="Index is not ready yet. Upload documents or rebuild the index.")

    results = await run_in_threadpool(index_store.search, request.query, request.top_k)
    context_text = "\n\n".join(item["text"] for item in results)
    answer = await llm_client.generate_answer(request.query, context_text)

    return {
        "query": request.query,
        "answer": answer,
        "sources": results,
    }


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=True)
