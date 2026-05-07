# src/generate.py
import os

from openai import OpenAI

from src.utils import load_env

load_env()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))


def generate_answer(query, context):
    prompt = f"""
    You are a helpful assistant. Use the following context to answer the question.

    Context:
    {context}

    Question: {query}
    """

    try:
        response = client.responses.create(
            model="gpt-4.1-mini",
            input=prompt,
        )
        return response.output_text
    except Exception as e:
        return f"Error generating answer: {e}"
