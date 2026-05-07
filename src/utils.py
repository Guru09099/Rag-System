import os
from dotenv import load_dotenv


def load_env():
    load_dotenv()
    if not os.getenv("OPENAI_API_KEY"):
        print("Warning: OPENAI_API_KEY not found in .env")
