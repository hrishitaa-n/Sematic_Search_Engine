import requests
import os

folder = 'data/corpus'
files = [f for f in os.listdir(folder) if f.endswith('.txt')]

for fname in sorted(files):
    filepath = os.path.join(folder, fname)
    with open(filepath, 'rb') as f:
        response = requests.post(
            'http://127.0.0.1:8000/ingest',
            files={'file': (fname, f, 'text/plain')}
        )
    result = response.json()
    print(f"{fname} → {result.get('chunks_stored', 'ERROR')} chunks")

print("\nDone")