# src/serve.py
from sanic import Sanic, response
from src.retrieve import search
from src.generate import generate_answer
import os

# Initialize Sanic App
app = Sanic("RAG_App")

# 1. Serve the HTML file at the root URL "/"
# We tell Sanic where the file is located
app.static('/', './static/index.html', name='home')

@app.post('/query')
async def query_endpoint(request):
    # Sanic access JSON data via request.json
    user_query = request.json.get('query')
    
    if not user_query:
        return response.json({"error": "No query provided"}, status=400)
    
    print(f"Received query: {user_query}")

    # Note: search() and generate_answer() are synchronous (blocking).
    # In a production Sanic app, you might run these in a separate thread,
    # but for this project, calling them directly is totally fine.
    
    # 1. Retrieve
    context = search(user_query)
    
    # 2. Generate (Gemini)
    answer = generate_answer(user_query, context)
    
    return response.json({
        "query": user_query,
        "answer": answer,
        "context_used": context
    })

if __name__ == '__main__':
    print("Starting Sanic RAG Server...")
    print("Go to: http://localhost:8000")
    # Sanic usually runs on port 8000 by default
    app.run(host="0.0.0.0", port=8000, debug=True, single_process=True)