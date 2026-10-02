import sys
sys.path.append('src')

import faiss
import numpy as np
import os
import io

from fastapi import FastAPI, Depends, UploadFile, File
from sqlalchemy.orm import Session
from sentence_transformers import SentenceTransformer
from groq import Groq
from dotenv import load_dotenv
import pdfplumber

from database import get_db, engine, Base
from models import Document

# ── startup ──────────────────────────────────────────────
load_dotenv()
groq_client = Groq(api_key=os.getenv("GROQ_API_KEY"))

app = FastAPI()
model = SentenceTransformer('all-MiniLM-L6-v2')
index = faiss.read_index('embeddings/chunks_index.faiss')
chunk_ids = np.load('embeddings/chunk_ids.npy')
chunk_filenames = np.load('embeddings/chunk_filenames.npy')
Base.metadata.create_all(bind=engine)

# ── helpers ───────────────────────────────────────────────
def chunk_text(text, chunk_size=256, overlap=50):
    words = text.split()
    chunks = []
    start = 0
    while start < len(words):
        end = start + chunk_size
        chunk = ' '.join(words[start:end])
        chunks.append(chunk)
        start += chunk_size - overlap
    return chunks

# ── routes ────────────────────────────────────────────────
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
        chunk_id = int(chunk_ids[idx])
        filename = chunk_filenames[idx]
        chunk = db.query(Document).filter(Document.id == chunk_id).first()
        results.append({
            "rank": len(results) + 1,
            "filename": filename,
            "chunk_text": chunk.chunk_text if chunk else "",
            "score": round(float(dist), 4)
        })
    return {"query": q, "results": results}

@app.post("/ingest")
async def ingest(file: UploadFile = File(...), db: Session = Depends(get_db)):
    content = await file.read()
    text = content.decode('utf-8')
    chunks = chunk_text(text)
    for i, chunk in enumerate(chunks):
        db.add(Document(filename=file.filename, chunk_index=i, chunk_text=chunk))
    db.commit()
    return {"filename": file.filename, "chunks_stored": len(chunks)}

@app.post("/ingest-pdf")
async def ingest_pdf(file: UploadFile = File(...), db: Session = Depends(get_db)):
    content = await file.read()
    text = ""
    page_count = 0
    with pdfplumber.open(io.BytesIO(content)) as pdf:
        page_count = len(pdf.pages)
        for page in pdf.pages:
            page_text = page.extract_text()
            if page_text:
                text += page_text + "\n"
    if not text.strip():
        return {"error": "Could not extract text. PDF may be scanned or image-based."}
    chunks = chunk_text(text)
    for i, chunk in enumerate(chunks):
        db.add(Document(filename=file.filename, chunk_index=i, chunk_text=chunk))
    db.commit()
    return {"filename": file.filename, "pages": page_count, "chunks_stored": len(chunks)}

@app.get("/documents")
def list_documents(db: Session = Depends(get_db)):
    docs = db.query(Document.filename).distinct().all()
    return {"documents": [d[0] for d in docs]}

@app.get("/ask")
def ask(q: str, db: Session = Depends(get_db)):
    query_vec = model.encode([q]).astype(np.float32)
    faiss.normalize_L2(query_vec)
    distances, indices = index.search(query_vec, 5)

    chunks = []
    for dist, idx in zip(distances[0], indices[0]):
        chunk_id = int(chunk_ids[idx])
        filename = chunk_filenames[idx]
        chunk = db.query(Document).filter(Document.id == chunk_id).first()
        if chunk:
            chunks.append({
                "filename": filename,
                "text": chunk.chunk_text,
                "score": round(float(dist), 4)
            })

    context = ""
    for i, chunk in enumerate(chunks):
        context += f"[Source {i+1}] {chunk['filename']}\n{chunk['text']}\n\n"

    response = groq_client.chat.completions.create(
        model="qwen/qwen3.8-27b",
        messages=[
            {
                "role": "system",
                "content": "You are a document assistant. Answer using ONLY the context below. If the answer isn't there say 'I don't have enough information.' Cite sources as [Source N].\n\nContext:\n" + context
            },
            {"role": "user", "content": q}
        ]
    )

    return {
        "question": q,
        "answer": response.choices[0].message.content,
        "sources": [{"filename": c["filename"], "score": c["score"]} for c in chunks]
    }