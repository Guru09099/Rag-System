import os

import httpx


class OllamaClient:
    def __init__(self):
        self.host = os.getenv("OLLAMA_HOST", "http://127.0.0.1:11434").rstrip("/")
        self.model = os.getenv("OLLAMA_MODEL", "llama3.2")
        self.temperature = float(os.getenv("OLLAMA_TEMPERATURE", "0.2"))
        self.max_tokens = int(os.getenv("OLLAMA_MAX_TOKENS", "512"))

    async def generate_answer(self, query: str, context: str) -> str:
        prompt = (
            "You are an expert assistant for internal business documents. "
            "Use the provided context to answer the question concisely and cite the source documents.\n\n"
            f"Context:\n{context}\n\n"
            f"Question: {query}\n\nAnswer:"
        )

        payload = {
            "model": self.model,
            "prompt": prompt,
            "stream": False,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
        }

        try:
            async with httpx.AsyncClient(timeout=120) as client:
                response = await client.post(
                    f"{self.host}/api/generate",
                    json=payload,
                    headers={"Content-Type": "application/json"},
                )
                response.raise_for_status()
                data = response.json()
                return data.get("response", "No answer generated.")
        except httpx.RequestError as exc:
            return (
                f"Error connecting to Ollama at {self.host}: {exc}. "
                f"Please make sure Ollama is running and the model {self.model} is installed."
            )
        except httpx.HTTPStatusError as exc:
            return f"Ollama returned an error: {exc.response.status_code} {exc.response.text}"
        except Exception as exc:
            return f"Error generating answer: {exc}"
