# src/list_models.py
import json
from urllib import request

print("--- CHECKING AVAILABLE MODELS ---")
try:
    with request.urlopen("http://localhost:11434/api/tags", timeout=10) as response:
        data = json.loads(response.read().decode("utf-8"))
        for model in data.get("models", []):
            print(model["name"])
except Exception as e:
    print(f"Error: {e}")
    print("Make sure Ollama is running.")
print("---------------------------------")
