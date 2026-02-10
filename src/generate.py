# src/generate.py
import google.generativeai as genai
from src.utils import load_env
import os

load_env()
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

def generate_answer(query, context):
    # We are using the model explicitly listed in your account
    model = genai.GenerativeModel('gemini-2.0-flash')

    prompt = f"""
    You are a helpful assistant. Use the following context to answer the question.
    
    Context:
    {context}
    
    Question: {query}
    """
    
    try:
        response = model.generate_content(prompt)
        return response.text
    except Exception as e:
        return f"Error generating answer: {e}"