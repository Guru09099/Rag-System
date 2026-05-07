# src/list_models.py
import os

from openai import OpenAI

from src.utils import load_env

load_env()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

print("--- CHECKING AVAILABLE MODELS ---")
try:
    for model in client.models.list():
        print(model.id)
except Exception as e:
    print(f"Error: {e}")
print("---------------------------------")
