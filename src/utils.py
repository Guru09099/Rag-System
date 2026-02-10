import os
from dotenv import load_dotenv

def load_env():
    load_dotenv()
    # Changed from OPENAI_API_KEY to GOOGLE_API_KEY
    if not os.getenv("GOOGLE_API_KEY"):
        print("Warning: GOOGLE_API_KEY not found in .env")