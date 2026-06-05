# src/serve.py
import os

from sanic import Sanic, response
from src.build_index import create_vector_db
from src.generate import generate_answer
from src.retrieve import search

# Initialize Sanic App
app = Sanic("RAG_App")

# 1. Serve the HTML file at the root URL "/"
# We tell Sanic where the file is located
app.static('/', './static/index.html', name='home')


@app.post('/upload')
async def upload_endpoint(request):
    uploaded_file = request.files.get('file')

    if not uploaded_file:
        return response.json({"error": "No file uploaded"}, status=400)

    filename = os.path.basename(uploaded_file.name)
    if not filename.lower().endswith(('.pdf', '.txt')):
        return response.json({"error": "Only .pdf and .txt files are allowed"}, status=400)

    os.makedirs("data", exist_ok=True)
    file_path = os.path.join("data", filename)

    with open(file_path, "wb") as f:
        f.write(uploaded_file.body)

    indexed = create_vector_db()
    if not indexed:
        return response.json(
            {"error": f"Uploaded {filename}, but indexing failed"},
            status=500,
        )

    return response.json({"message": f"Uploaded and indexed {filename}"})


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

    # 2. Generate (Ollama)
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
