"""CLI entrypoint for running the RAG evaluation benchmark."""

from evals.evaluate import run_eval

if __name__ == "__main__":
    run_eval(top_k=3)
