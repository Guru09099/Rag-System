import faiss
import numpy as np
import pickle
from sentence_transformers import SentenceTransformer
import os

def create_vector_db():
    print("1. Loading embedding model (this might take a moment)...")
    # This downloads a small, fast local model
    embedder = SentenceTransformer('all-MiniLM-L6-v2') 

    print("2. Processing text...")
    # Read file
    with open("./data/file1.txt", "r", encoding="utf-8") as f:
        text = f.read()

    # Simple splitting by newlines or crude fixed size (manual chunking)
    # For better results later, we can write a smarter splitter
    chunks = [text[i:i+500] for i in range(0, len(text), 450)] 
    
    print(f"   Created {len(chunks)} chunks.")

    print("3. Creating embeddings...")
    embeddings = embedder.encode(chunks)
    
    # FAISS expects float32
    embeddings = np.array(embeddings).astype('float32')

    print("4. Building FAISS index...")
    # Dimension of all-MiniLM-L6-v2 is 384
    dimension = embeddings.shape[1] 
    index = faiss.IndexFlatL2(dimension)
    index.add(embeddings)

    print("5. Saving index and metadata...")
    # Save the vector index
    faiss.write_index(index, "vector.index")
    
    # Save the actual text chunks so we can retrieve them later
    with open("chunks.pkl", "wb") as f:
        pickle.dump(chunks, f)
        
    print("Done! 'vector.index' and 'chunks.pkl' created.")

if __name__ == "__main__":
    create_vector_db()