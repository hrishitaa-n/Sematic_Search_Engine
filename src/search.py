import numpy as np
import faiss
from sentence_transformers import SentenceTransformer

# ── old functions (keep these) ──────────────────────────
def load_index(vec_path, names_path):
    embeddings = np.load(vec_path)
    with open(names_path) as f:
        filenames = f.read().strip().split('\n')
    return embeddings, filenames

def brute_force_search(query, embeddings, filenames, model, top_k=5):
    query_vec = model.encode([query])[0]
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    normed = embeddings / norms
    query_normed = query_vec / np.linalg.norm(query_vec)
    scores = normed @ query_normed
    top_indices = np.argsort(scores)[::-1][:top_k]
    results = []
    for idx in top_indices:
        results.append({
            'rank': len(results) + 1,
            'file': filenames[idx],
            'score': round(float(scores[idx]), 4)
        })
    return results

# ── new functions (add these) ───────────────────────────
def build_faiss_index(embeddings):
    dim = embeddings.shape[1]
    index = faiss.IndexFlatL2(dim)
    vecs = embeddings.astype(np.float32)
    faiss.normalize_L2(vecs)
    index.add(vecs)
    print(f"FAISS index built: {index.ntotal} vectors")
    return index

def faiss_search(query, index, filenames, model, top_k=5):
    query_vec = model.encode([query]).astype(np.float32)
    faiss.normalize_L2(query_vec)
    distances, indices = index.search(query_vec, top_k)
    results = []
    for rank, (dist, idx) in enumerate(zip(distances[0], indices[0])):
        results.append({
            'rank': rank + 1,
            'file': filenames[idx],
            'score': round(float(dist), 4)
        })
    return results

# ── main ────────────────────────────────────────────────
if __name__ == '__main__':
    model = SentenceTransformer('all-MiniLM-L6-v2')
    embeddings, filenames = load_index(
        'embeddings/vectors.npy',
        'embeddings/filenames.txt'
    )

    index = build_faiss_index(embeddings)
    faiss.write_index(index, 'embeddings/index.faiss')

    query = "What is a neural network?"

    print("\n--- Brute Force ---")
    bf = brute_force_search(query, embeddings, filenames, model)
    for r in bf[:3]:
        print(f"  #{r['rank']} [{r['score']}] {r['file']}")

    print("\n--- FAISS ---")
    fa = faiss_search(query, index, filenames, model)
    for r in fa[:3]:
        print(f"  #{r['rank']} [{r['score']}] {r['file']}")