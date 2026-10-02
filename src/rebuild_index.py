import sys
sys.path.append('src')

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
from database import SessionLocal
from models import Document

def rebuild_index():
    db = SessionLocal()
    model = SentenceTransformer('all-MiniLM-L6-v2')

    print("Loading all chunks from PostgreSQL...")
    all_chunks = db.query(Document).order_by(Document.id).all()
    print(f"Found {len(all_chunks)} chunks")

    texts = [chunk.chunk_text for chunk in all_chunks]
    chunk_ids = [chunk.id for chunk in all_chunks]
    filenames = [chunk.filename for chunk in all_chunks]

    print("Embedding all chunks...")
    embeddings = model.encode(texts, show_progress_bar=True, batch_size=64)
    embeddings = embeddings.astype(np.float32)
    faiss.normalize_L2(embeddings)

    print("Building FAISS index...")
    dim = embeddings.shape[1]
    index = faiss.IndexFlatL2(dim)
    index.add(embeddings)

    print("Saving index and metadata...")
    faiss.write_index(index, 'embeddings/chunks_index.faiss')
    np.save('embeddings/chunk_ids.npy', np.array(chunk_ids))
    np.save('embeddings/chunk_filenames.npy', np.array(filenames))

    print(f"Done — {index.ntotal} chunks indexed")
    db.close()

if __name__ == '__main__':
    rebuild_index()