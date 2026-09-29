import json
import os
from rank_bm25 import BM25Okapi

# load corpus
folder = 'data/corpus'
docs, filenames = [], []
for fname in sorted(os.listdir(folder)):
    if fname.endswith('.txt'):
        with open(os.path.join(folder, fname), encoding='utf-8') as f:
            docs.append(f.read().strip())
        filenames.append(fname)

# build BM25 index
tokenized = [doc.lower().split() for doc in docs]
bm25 = BM25Okapi(tokenized)

# load eval set
with open('eval/queries.json') as f:
    eval_set = json.load(f)

hits = 0
for item in eval_set:
    query_tokens = item['query'].lower().split()
    scores = bm25.get_scores(query_tokens)
    top5_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:5]
    top5_files = [filenames[i] for i in top5_indices]

    is_hit = any(e in top5_files for e in item['expected_files'])
    hits += is_hit

    status = "HIT " if is_hit else "MISS"
    print(f"[{status}] {item['query']}")
    if not is_hit:
        print(f"       Expected: {item['expected_files']}")
        print(f"       Got:      {top5_files[:3]}")

print(f"\nBM25 Recall@5: {hits}/{len(eval_set)} = {hits/len(eval_set)*100:.0f}%")