# src/list_models.py
import google.generativeai as genai
from src.utils import load_env
import os

load_env()
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

print("--- CHECKING AVAILABLE MODELS ---")
try:
    for m in genai.list_models():
        if 'generateContent' in m.supported_generation_methods:
            print(m.name)
except Exception as e:
    print(f"Error: {e}")
print("---------------------------------")