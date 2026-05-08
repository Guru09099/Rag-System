# src/generate.py
import json
import os
from urllib import error
from urllib import request


OLLAMA_HOST = os.getenv("OLLAMA_HOST", "http://127.0.0.1:11434").rstrip("/")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "llama3.2")


def generate_answer(query, context):
    prompt = f"""
    You are a helpful assistant. Use the following context to answer the question.

    Context:
    {context}

    Question: {query}
    """

    try:
        payload = json.dumps({
            "model": OLLAMA_MODEL,
            "prompt": prompt,
            "stream": False,
        }).encode("utf-8")

        req = request.Request(
            f"{OLLAMA_HOST}/api/generate",
            data=payload,
            headers={"Content-Type": "application/json"},
            method="POST",
        )

        with request.urlopen(req, timeout=120) as response:
            data = json.loads(response.read().decode("utf-8"))
            return data.get("response", "No answer generated.")
    except error.URLError as e:
        return (
            f"Error connecting to Ollama at {OLLAMA_HOST}: {e}. "
            f"Start Ollama and install the model with: ollama pull {OLLAMA_MODEL}"
        )
    except Exception as e:
        return (
            f"Error generating answer with Ollama: {e}. "
            f"Make sure Ollama is running and {OLLAMA_MODEL} is installed."
        )
