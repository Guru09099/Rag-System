# Intelligent Document Query System (RAG + Evaluation)

A pure-Python, interview-ready Retrieval-Augmented Generation (RAG) system with an automated evaluation pipeline measuring **Precision@k**, **Recall@k**, **Mean Reciprocal Rank (MRR)**, and the **RAG Triad** (**Context Relevance**, **Faithfulness**, **Answer Relevance**).

Built using **FastAPI**, **FAISS**, **SentenceTransformers**, and **Ollama** (`llama3.2`).

---

## 🏗️ Architecture Overview

```
User Query (Web UI / API / Benchmark)
                  │
                  ▼
          FastAPI (app.py)
                  │
                  ▼
       RAGPipeline (src/pipeline.py)
   ┌──────────────┴──────────────┐
   ▼                             ▼
Vector Search (src/retriever.py) LLM Generator (src/generator.py)
(FAISS + SentenceTransformers)   (Ollama llama3.2 via httpx)
```

---

## 📁 Modular Directory Structure

```
d:\RAG\
├── src/
│   ├── __init__.py         # Package exports
│   ├── ingestion.py        # PDF/TXT extraction & boundary-aware chunking
│   ├── retriever.py        # VectorIndexStore (FAISS, SentenceTransformers, JSON metadata)
│   ├── generator.py        # OllamaGenerator (Ollama client, prompt formatting, injection guard)
│   └── pipeline.py         # RAGPipeline (High-level facade connecting retrieval & generation)
├── evals/
│   ├── __init__.py         # Package marker
│   ├── metrics.py          # Pure formulas for Precision@k, Recall@k, MRR & RAG Triad
│   ├── evaluate.py         # Benchmark runner and dashboard generator
│   └── eval_dataset.json   # Golden test dataset (Queries, Expected Sources, Ground Truth)
├── static/
│   └── index.html          # Clean, zero-build responsive web interface
├── data/                   # Document knowledge base (.txt, .pdf)
├── reports/
│   └── summary_statistics.png # Auto-generated 2-panel performance dashboard
├── app.py                  # Lean FastAPI server (REST API + Web UI)
├── evaluate.py             # CLI entrypoint for running evaluation benchmark
├── pyproject.toml          # Standard packaging & CLI scripts (rag-server, rag-eval)
├── requirements.txt        # Python dependency manifest
├── Dockerfile              # Container deployment recipe
└── .env.example            # Environment variables template
```

---

## 🚀 Quickstart

### 1. Installation

```powershell
# Create and activate virtual environment
python -m venv .venv
.\.venv\Scripts\Activate.ps1

# Install requirements
pip install -r requirements.txt
```

### 2. Start the Application

```powershell
python app.py
```
Open **`http://localhost:8000`** in your browser to interact with the web interface.

### 3. Run the Evaluation Benchmark

```powershell
python evaluate.py
```
This evaluates the retrieval and generation pipeline across the golden test dataset and automatically generates the visual dashboard in [`reports/summary_statistics.png`](reports/summary_statistics.png).

---

## 📊 Evaluation Metrics & Output

Running `python evaluate.py` outputs a clean benchmark report:

```
=================================================================
                      EVALUATION RESULTS
=================================================================
Total Benchmark Queries:     8
Mean Retrieval Latency:      15.4 ms
-----------------------------------------------------------------
STAGE 1: RETRIEVAL QUALITY
  * Precision@3:              62.50%
  * Recall@3:                 100.00%
  * Mean Reciprocal Rank (MRR):    1.0000
-----------------------------------------------------------------
STAGE 2: GENERATION QUALITY (THE RAG TRIAD)
  * Context Relevance:             79.03%
  * Faithfulness (Groundedness):   100.00%
  * Answer Relevance:              33.05%
=================================================================
[Dashboard Saved] -> reports/summary_statistics.png
```

---

## 🛠️ API Reference

| Method | Endpoint | Description |
| :--- | :--- | :--- |
| `GET` | `/` | Web user interface |
| `GET` | `/health` | Health check & index readiness |
| `POST` | `/upload` | Upload `.pdf` or `.txt` document and rebuild index |
| `POST` | `/reindex` | Rebuild FAISS vector index from `data/` directory |
| `POST` | `/query` | Retrieve context & generate grounded answer |
