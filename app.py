"""FastAPI Server for RAG Document Assistant.

A lightweight, clean REST API serving the unified RAG pipeline and web UI.
"""

from pathlib import Path
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field

from src.pipeline import RAGPipeline

app = FastAPI(title="RAG Document Assistant", version="1.1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

pipeline = RAGPipeline()
STATIC_HTML = Path("static/index.html")
DATA_DIR = Path("data")


class QueryRequest(BaseModel):
    query: str = Field(..., min_length=1)
    top_k: int = Field(default=3, ge=1, le=10)


@app.on_event("startup")
def startup():
    try:
        pipeline.load_index()
    except Exception:
        pipeline.build_index()


@app.get("/")
def home():
    if STATIC_HTML.exists():
        return FileResponse(STATIC_HTML)
    return {"message": "RAG API is online. Query via POST /query"}


@app.get("/health")
def health():
    return {
        "status": "ok",
        "index_ready": pipeline.is_ready(),
        "chunks_count": len(pipeline.chunks),
    }


@app.post("/upload")
async def upload(file: UploadFile = File(...)):
    if not file.filename.lower().endswith((".pdf", ".txt")):
        raise HTTPException(status_code=400, detail="Only .pdf and .txt allowed.")
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    file_path = DATA_DIR / Path(file.filename).name
    content = await file.read()
    file_path.write_bytes(content)
    pipeline.build_index()
    return {"message": f"Uploaded and indexed '{file.filename}' successfully."}


@app.post("/reindex")
def reindex():
    success = pipeline.build_index()
    if not success:
        raise HTTPException(status_code=500, detail="Failed to build index. Check data files.")
    return {"message": "Index built successfully."}


@app.post("/query")
def query_docs(req: QueryRequest):
    if not pipeline.is_ready():
        pipeline.build_index()
    return pipeline.ask(req.query, top_k=req.top_k)


def start():
    """CLI launcher."""
    import uvicorn
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=True)


if __name__ == "__main__":
    start()
