from sentence_transformers import SentenceTransformer
import numpy as np
import os

def load_corpus(folder):
    docs, filenames = [], []
    for fname in sorted(os.listdir(folder)):
        if fname.endswith('.txt'):
            with open(os.path.join(folder, fname), encoding='utf-8') as f:
                docs.append(f.read().strip())
            filenames.append(fname)
    return docs, filenames

def embed_corpus(docs, model_name='all-MiniLM-L6-v2'):
    model = SentenceTransformer(model_name)
    embeddings = model.encode(docs, show_progress_bar=True)
    return embeddings

if __name__ == '__main__':
    os.makedirs('embeddings', exist_ok=True)

    docs, filenames = load_corpus('data/corpus')
    print(f"Found {len(docs)} documents")

    embeddings = embed_corpus(docs)

    np.save('embeddings/vectors.npy', embeddings)
    with open('embeddings/filenames.txt', 'w') as f:
        f.write('\n'.join(filenames))

    print(f"Done — shape: {embeddings.shape}")