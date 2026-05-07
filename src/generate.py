# src/generate.py
import json
from urllib import request


def generate_answer(query, context):
    prompt = f"""
    You are a helpful assistant. Use the following context to answer the question.

    Context:
    {context}

    Question: {query}
    """

    try:
        payload = json.dumps({
            "model": "llama3.2",
            "prompt": prompt,
            "stream": False,
        }).encode("utf-8")

        req = request.Request(
            "http://localhost:11434/api/generate",
            data=payload,
            headers={"Content-Type": "application/json"},
            method="POST",
        )

        with request.urlopen(req, timeout=120) as response:
            data = json.loads(response.read().decode("utf-8"))
            return data.get("response", "No answer generated.")
    except Exception as e:
        return (
            f"Error generating answer with Ollama: {e}. "
            "Make sure Ollama is running and llama3.2 is installed."
        )
