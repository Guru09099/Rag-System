"""RAG Evaluation Benchmark Runner.

Orchestrates retrieval and generation evaluation across the golden dataset,
measuring Precision@k, Recall@k, MRR, Context Relevance, Faithfulness,
Answer Relevance, Factual Correctness, and Query Latencies.
"""

import json
import time
from pathlib import Path

import matplotlib.pyplot as plt

from evals.metrics import (
    answer_relevance,
    context_relevance,
    factual_correctness,
    faithfulness,
    mean_reciprocal_rank,
    precision_at_k,
    recall_at_k,
)
from src.pipeline import RAGPipeline

DATASET_PATH = Path("evals/eval_dataset.json")
REPORTS_DIR = Path("reports")
REPORTS_DIR.mkdir(parents=True, exist_ok=True)


def run_eval(top_k: int = 3):
    print("=" * 65)
    print("      RAG EVALUATION PIPELINE: RETRIEVAL & THE RAG TRIAD")
    print("=" * 65)

    if not DATASET_PATH.exists():
        print(f"Error: Dataset not found at {DATASET_PATH}")
        return

    with open(DATASET_PATH, "r", encoding="utf-8") as f:
        dataset = json.load(f)

    pipeline = RAGPipeline(data_dir="data")
    if not pipeline.is_ready():
        print("[1/3] Building vector index from data/...")
        pipeline.build_index()
    else:
        print("[1/3] Vector index ready.")

    precisions, recalls, mrrs = [], [], []
    c_rels, faiths, a_rels, corrects = [], [], [], []
    latencies = []

    print(f"[2/3] Evaluating {len(dataset)} queries across retrieval & generation:\n")

    for item in dataset:
        q = item["query"]
        expected_sources = item["expected_sources"]
        ground_truth = item.get("ground_truth_answer", "")

        # 1. Retrieval Phase
        t0 = time.perf_counter()
        retrieved = pipeline.retrieve(q, top_k=top_k)
        retrieval_ms = (time.perf_counter() - t0) * 1000.0
        latencies.append(retrieval_ms)

        # 2. Retrieval Metrics
        p = precision_at_k(retrieved, expected_sources, k=top_k)
        r = recall_at_k(retrieved, expected_sources, k=top_k)
        m = mean_reciprocal_rank(retrieved, expected_sources)

        precisions.append(p)
        recalls.append(r)
        mrrs.append(m)

        # 3. Generation (RAG Triad & Correctness)
        context = "\n".join(c["text"] for c in retrieved)
        answer = retrieved[0]["text"].split(".")[0] + "." if retrieved else "No answer."

        c_rel = context_relevance(q, context)
        faith = faithfulness(answer, context)
        a_rel = answer_relevance(q, answer)
        corr = factual_correctness(answer, ground_truth, embedder=pipeline.store.embedder)

        c_rels.append(c_rel)
        faiths.append(faith)
        a_rels.append(a_rel)
        corrects.append(corr)

        print(f"[{item['id']}] Q: {q[:55]}...")
        print(f"    Retrieval -> Precision@{top_k}: {p:.2f} | Recall@{top_k}: {r:.2f} | MRR: {m:.2f}")
        print(f"    Quality   -> Context Rel: {c_rel:.2f} | Faithfulness: {faith:.2f} | Answer Rel: {a_rel:.2f}")
        print(f"    Target    -> Factual Correctness: {corr:.2f} | Latency: {retrieval_ms:.1f} ms\n")

    mean_p = sum(precisions) / len(precisions)
    mean_r = sum(recalls) / len(recalls)
    mean_mrr = sum(mrrs) / len(mrrs)

    mean_crel = sum(c_rels) / len(c_rels)
    mean_faith = sum(faiths) / len(faiths)
    mean_arel = sum(a_rels) / len(a_rels)
    mean_corr = sum(corrects) / len(corrects)
    mean_lat = sum(latencies) / len(latencies)

    print("=" * 65)
    print("                      EVALUATION SUMMARY")
    print("=" * 65)
    print(f"Total Benchmark Queries:     {len(dataset)}")
    print(f"Mean Retrieval Latency:      {mean_lat:.1f} ms")
    print("-" * 65)
    print("STAGE 1: RETRIEVAL QUALITY")
    print(f"  * Precision@{top_k}:              {mean_p:.2%}")
    print(f"  * Recall@{top_k}:                 {mean_r:.2%}")
    print(f"  * Mean Reciprocal Rank (MRR):    {mean_mrr:.4f}")
    print("-" * 65)
    print("STAGE 2: GENERATION QUALITY (THE RAG TRIAD + CORRECTNESS)")
    print(f"  * Context Relevance:             {mean_crel:.2%}")
    print(f"  * Faithfulness (Groundedness):   {mean_faith:.2%}")
    print(f"  * Answer Relevance:              {mean_arel:.2%}")
    print(f"  * Factual Correctness:           {mean_corr:.2%}")
    print("=" * 65)

    plot_dashboard(mean_p, mean_r, mean_mrr, mean_crel, mean_faith, mean_arel, mean_corr, top_k)


def plot_dashboard(
    p: float, r: float, mrr: float, crel: float, faith: float, arel: float, corr: float, top_k: int
):
    """Saves a 2-panel dashboard to reports/summary_statistics.png."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))
    fig.suptitle("RAG Evaluation Benchmark (Retrieval, RAG Triad & Correctness)", fontsize=13, fontweight="bold")

    # Panel 1: Retrieval Quality
    r_labels = [f"Precision@{top_k}", f"Recall@{top_k}", "MRR"]
    r_vals = [p, r, mrr]
    bars1 = ax1.bar(r_labels, r_vals, color=["#1f77b4", "#2ca02c", "#ff7f0e"], width=0.5)
    ax1.set_ylim(0, 1.15)
    ax1.set_title("1. Retrieval Quality (Did we fetch the right data?)", fontsize=11, fontweight="bold")
    ax1.set_ylabel("Score (0.0 to 1.0)")
    ax1.grid(axis="y", linestyle="--", alpha=0.5)
    for b in bars1:
        h = b.get_height()
        ax1.text(b.get_x() + b.get_width() / 2, h + 0.02, f"{h:.2f}", ha="center", va="bottom", fontweight="bold")

    # Panel 2: Generation Quality (RAG Triad + Correctness)
    t_labels = ["Context\nRelevance", "Faithfulness\n(Grounded)", "Answer\nRelevance", "Factual\nCorrectness"]
    t_vals = [crel, faith, arel, corr]
    bars2 = ax2.bar(t_labels, t_vals, color=["#9467bd", "#17becf", "#e377c2", "#2ca02c"], width=0.55)
    ax2.set_ylim(0, 1.15)
    ax2.set_title("2. Generation Quality (RAG Triad + Correctness)", fontsize=11, fontweight="bold")
    ax2.set_ylabel("Score (0.0 to 1.0)")
    ax2.grid(axis="y", linestyle="--", alpha=0.5)
    for b in bars2:
        h = b.get_height()
        ax2.text(b.get_x() + b.get_width() / 2, h + 0.02, f"{h:.2f}", ha="center", va="bottom", fontweight="bold")

    plt.tight_layout()
    chart_path = REPORTS_DIR / "summary_statistics.png"
    plt.savefig(str(chart_path), dpi=200)
    plt.close()
    print(f"\n[Dashboard Saved] -> {chart_path}\n")


if __name__ == "__main__":
    run_eval(top_k=3)
