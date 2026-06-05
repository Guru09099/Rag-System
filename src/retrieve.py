# src/retrieve.py
import faiss
import pickle
import numpy as np
from sentence_transformers import SentenceTransformer
import os

# Define the model globally
embedder = SentenceTransformer('all-MiniLM-L6-v2')


def search(query, k=3):
    # Check if index exists before trying to load it
    if not os.path.exists("vector.index"):
        raise FileNotFoundError("Index file not found. Did you run build_index.py?")

    # Load the database
    index = faiss.read_index("vector.index")
    with open("chunks.pkl", "rb") as f:
        chunks = pickle.load(f)

    # 1. Turn query into vector
    query_vector = embedder.encode([query])
    query_vector = np.array(query_vector).astype('float32')
    # 2. Search FAISS
    distances, indices = index.search(query_vector, k)
    # 3. Retrieve text based on indices
    results = []
    for idx in indices[0]:
        # Handle cases where FAISS returns -1 (no match)
        if idx != -1 and idx < len(chunks):
            results.append(chunks[idx])
    return "\n\n".join(results)
