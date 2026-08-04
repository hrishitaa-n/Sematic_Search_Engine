import json
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
import sys
sys.path.append('src')
from search import load_index, faiss_search

model = SentenceTransformer('all-MiniLM-L6-v2')
embeddings, filenames = load_index('embeddings/vectors.npy', 'embeddings/filenames.txt')
index = faiss.read_index('embeddings/index.faiss')

with open('eval/queries.json') as f:
    eval_set = json.load(f)

hits = 0
for item in eval_set:
    results = faiss_search(item['query'], index, filenames, model, top_k=5)
    top5_files = [r['file'] for r in results]

    is_hit = any(e in top5_files for e in item['expected_files'])
    hits += is_hit

    status = "HIT " if is_hit else "MISS"
    print(f"[{status}] {item['query']}")
    if not is_hit:
        print(f"       Expected: {item['expected_files']}")
        print(f"       Got:      {top5_files[:3]}")

print(f"\nRecall@5: {hits}/{len(eval_set)} = {hits/len(eval_set)*100:.0f}%")