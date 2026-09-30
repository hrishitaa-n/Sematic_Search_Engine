# Semantic Search Engine

A full RAG (Retrieval Augmented Generation) system that answers questions 
from a document corpus using semantic search, not keyword matching.

## What it does

- Accepts natural language questions via REST API
- Finds relevant documents using vector similarity search (FAISS)
- Retrieves exact chunk text from PostgreSQL
- Generates answers using an LLM (Groq) with source citations
- Supports both .txt and PDF document ingestion

## Results

| Method | Recall@5 | Queries |
|--------|----------|---------|
| Semantic Search (FAISS + Sentence Transformers) | 92% | 25 |
| Keyword Search (BM25 baseline) | 64% | 25 |
| **Improvement** | **+28 points** | — |

Evaluation used 25 semantically challenging queries — questions that 
describe concepts without using the exact words from document titles.

## Architecture
User Query
↓
FastAPI (/ask endpoint)
↓
Sentence Transformers (embed query → 384-dim vector)
↓
FAISS (find top-5 nearest document vectors)
↓
PostgreSQL (fetch chunk text by filename)
↓
Groq LLM (generate answer from chunks, cite sources)
↓
JSON response with answer + citations





## Tech Stack

- **FastAPI** — REST API framework
- **Sentence Transformers** — all-MiniLM-L6-v2 embedding model
- **FAISS** — vector similarity search (Facebook AI Research)
- **PostgreSQL** — chunk text and metadata storage
- **SQLAlchemy** — ORM for database access
- **Groq** — LLM API for answer generation (Llama/Qwen)
- **pdfplumber** — PDF text extraction

## Project Structure

semantic-search/
├── data/corpus/ # 115 Wikipedia articles
├── embeddings/
│ ├── vectors.npy # (115, 384) embedding matrix
│ ├── filenames.txt # row → filename mapping
│ └── index.faiss # FAISS search index
├── src/
│ ├── embed.py # corpus ingestion pipeline
│ ├── search.py # brute force + FAISS search
│ ├── main.py # FastAPI application
│ ├── database.py # PostgreSQL connection
│ └── models.py # SQLAlchemy table definitions
└── eval/
├── queries.json # 25-query evaluation set
├── evaluate.py # Recall@5 measurement
└── bm25_eval.py # BM25 baseline comparison


## API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/` | GET | Health check |
| `/search?q=...` | GET | Semantic search, returns filenames + scores |
| `/ask?q=...` | GET | Full RAG — returns LLM answer + citations |
| `/ingest` | POST | Upload .txt file, chunk and store |
| `/ingest-pdf` | POST | Upload PDF, extract text, chunk and store |
| `/documents` | GET | List all ingested documents |

## Setup

```bash
# 1. Clone and create environment
git clone <your-repo>
cd semantic-search
python -m venv venv
venv\Scripts\activate  # Windows
source venv/bin/activate  # Mac/Linux

# 2. Install dependencies
pip install -r requirements.txt

# 3. Set up environment variables
cp .env.example .env
# Add your GROQ_API_KEY to .env

# 4. Start PostgreSQL and create database
python src/create_db.py
python src/init_db.py

# 5. Download corpus and build embeddings
python scripts/download_corpus.py
python src/embed.py

# 6. Start the API
uvicorn src.main:app --reload
```

## Evaluation Methodology

Recall@5 measures whether the correct document appears in the top 5 
search results. A query is a HIT if any expected document appears in 
the top 5, MISS otherwise.

The evaluation set uses semantically challenging queries — for example:
- "how do computers learn from examples" → machine_learning.txt
- "what technology powers chatgpt" → large_language_model.txt
- "how is data kept private on the internet" → encryption.txt

These queries deliberately avoid using words from document titles to 
test genuine semantic understanding rather than keyword overlap.

## Known Limitations

- PDF ingestion stores chunks in PostgreSQL but does not update the 
  FAISS index — PDFs are not searchable until embeddings are regenerated
- FAISS index must be rebuilt manually when new documents are added
- Short documents (under 300 words) score poorly in retrieval due to 
  sparse embeddings

## What I learned

- How vector embeddings represent semantic meaning
- Why normalisation is required before cosine similarity comparison
- The difference between exact search (IndexFlatL2) and approximate 
  search (IndexIVFFlat, IndexHNSW) in FAISS
- How chunking strategy affects retrieval quality
- How to measure retrieval quality with Recall@5 vs a BM25 baseline
- How RAG prevents LLM hallucination by restricting context