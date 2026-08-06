import sys
sys.path.append('src')

import faiss
import numpy as np
from fastapi import FastAPI, Depends
from sqlalchemy.orm import Session
from sentence_transformers import SentenceTransformer
from database import get_db, engine, Base
from models import Document

# setup
app = FastAPI()
model = SentenceTransformer('all-MiniLM-L6-v2')
embeddings = np.load('embeddings/vectors.npy')
with open('embeddings/filenames.txt') as f:
    filenames = f.read().strip().split('\n')
index = faiss.read_index('embeddings/index.faiss')

Base.metadata.create_all(bind=engine)

# routes
@app.get("/")
def root():
    return {"status": "running"}

@app.get("/search")
def search(q: str, top_k: int = 5, db: Session = Depends(get_db)):
    query_vec = model.encode([q]).astype(np.float32)
    faiss.normalize_L2(query_vec)
    distances, indices = index.search(query_vec, top_k)

    results = []
    for dist, idx in zip(distances[0], indices[0]):
        results.append({
            "rank": len(results) + 1,
            "filename": filenames[idx],
            "score": round(float(dist), 4)
        })
    return {"query": q, "results": results}
