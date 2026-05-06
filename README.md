# Multi-PDF RAG Chatbot (Ollama + FAISS + FastAPI + Streamlit)

A fully local Retrieval-Augmented Generation (RAG) chatbot for PDF documents.
Upload one or more PDFs, index them locally, and ask natural language questions with
source citations for each answer. No API keys. No cloud dependencies.

## Highlights

- Local-first architecture: Ollama models run on your machine
- Multi-PDF ingestion: upload and query across multiple documents
- Persistent FAISS index: survives restarts and supports incremental indexing
- Source-aware answers: each response includes document and chunk citations
- Service architecture: FastAPI backend + Streamlit frontend, independently deployable

## Tech Stack

- Python 3.9+
- FastAPI + Uvicorn
- Streamlit
- FAISS (`faiss-cpu`)
- PyMuPDF (`fitz`) for PDF text extraction
- Ollama for embeddings and generation
- NumPy, requests, Pydantic

## Supported Models

- `nomic-embed-text` for embeddings
- `llama3.2:3b` for answer generation

## Project Structure

```text
rag_pdf_chatbot/
├── app/
│   ├── main.py          # FastAPI routes
│   ├── rag_engine.py    # PDF indexing, retrieval, and LLM query logic
│   └── models.py        # Pydantic request/response schemas
├── data/                # Optional sample PDFs
├── faiss_index/         # Persistent FAISS index and metadata
├── streamlit_app.py     # Streamlit frontend chat interface
├── requirements.txt
└── README.md
```

---

## Architecture & Design Decisions

### System overview

```
┌─────────────────────────────────────────────────────────┐
│                     Streamlit UI                        │
│         (upload PDFs, chat interface, citations)        │
└────────────────────────┬────────────────────────────────┘
                         │ HTTP (REST)
┌────────────────────────▼────────────────────────────────┐
│                   FastAPI Backend                       │
│   POST /upload   POST /ask   GET /documents   DELETE /reset │
└────────┬──────────────────────────┬─────────────────────┘
         │                          │
┌────────▼────────┐      ┌──────────▼──────────┐
│  Indexing path  │      │   Query path         │
│                 │      │                      │
│ PyMuPDF extract │      │ embed question       │
│ chunk + overlap │      │ FAISS top-k search   │
│ nomic-embed     │      │ build prompt         │
│ FAISS upsert    │      │ llama3.2:3b generate │
│ persist to disk │      │ return answer +      │
└─────────────────┘      │ source citations     │
                         └──────────────────────┘
```

### Service architecture: FastAPI backend + Streamlit frontend

The app is split into a FastAPI backend and a Streamlit frontend rather than built as a single Streamlit script. This is a deliberate production-alignment choice. A monolithic Streamlit app tightly couples UI state with business logic, making it hard to test, extend, or serve from a different client. Separating concerns means the RAG engine can be called by any client — the Streamlit UI, a REST client, a future CLI, or another service — without modification. It also means the backend can be containerised and deployed independently. The cost is a two-process local setup, which is a minor inconvenience worth the architectural clarity.

### FAISS over a hosted vector database

FAISS (`faiss-cpu`) was chosen over hosted vector databases (Pinecone, Weaviate, Qdrant) for two reasons: the system runs fully locally with no external dependencies, and FAISS is the industry-standard library that underpins many production vector search systems. Understanding FAISS directly — index construction, serialisation, similarity search — is more transferable than learning a managed API. The tradeoff is that FAISS has no built-in metadata filtering, distributed scaling, or access control. For a single-user local system these are non-issues; for a multi-tenant production system a managed store would be the right call.

### Persistent index with incremental upsert

The FAISS index and metadata are serialised to disk (`faiss_index/index.faiss` + `meta.pkl`) after every upload. This means the index survives restarts and new documents can be added without re-indexing existing ones. Duplicate filenames are detected at upload time and skipped — a simple guard that prevents index bloat from repeated uploads of the same file. The metadata store maps each FAISS vector ID back to its source filename and chunk text, enabling source citations in answers.

### Chunking strategy: fixed-size with overlap

Text is split into 600-character chunks with 120-character overlap (20%). The overlap ensures sentences that fall on a chunk boundary are represented in both adjacent chunks, reducing the chance that a retrieval miss occurs because a relevant sentence was split. The tradeoff vs semantic chunking (splitting on sentence or paragraph boundaries) is slight boundary imprecision, which is acceptable for this retrieval task. Chunk size of 600 characters sits at roughly 100-150 tokens — well within the embedding model's context window while maintaining retrieval granularity.

### Dual-model design: separate embedding and generation models

Embeddings use `nomic-embed-text` and generation uses `llama3.2:3b`. These are not interchangeable roles. Embedding models are optimised for semantic similarity in vector space — they produce dense representations suited for nearest-neighbour search but are not designed to generate coherent text. Generation models are optimised for language fluency and instruction following. Using a single model for both tasks would mean either weak embeddings or no generation capability. Keeping them separate is standard RAG architecture.

### Top-4 retrieval

The query path retrieves the top 4 chunks by cosine similarity. Too few chunks risks missing relevant content; too many dilutes the prompt and increases the chance the LLM ignores distant context. 4 chunks at ~600 characters each adds ~2400 characters of context to the prompt — a comfortable fit within `llama3.2:3b`'s context window while leaving room for the question and system instruction.

### nomic-embed-text over OpenAI embeddings

`nomic-embed-text` runs locally via Ollama, eliminating API cost and latency variance from network calls. It produces 768-dimensional embeddings and performs competitively with `text-embedding-ada-002` on retrieval benchmarks for English text. The tradeoff is higher per-call latency than a cached API response and a dependency on Ollama running locally. For a local-first system this is the correct choice.

### Known limitations

- **No re-ranking** — retrieved chunks are ranked by embedding similarity only. A cross-encoder re-ranker (e.g. `ms-marco-MiniLM`) would improve precision on multi-hop questions.
- **No query expansion** — questions are embedded as-is. HyDE (Hypothetical Document Embedding) or query rewriting would improve retrieval for vague or short questions.
- **Flat FAISS index** — uses `IndexFlatL2` (exact search). For large document libraries (10k+ chunks) an approximate index (`IndexIVFFlat`) would reduce search latency significantly.
- **No authentication** — the FastAPI backend is open on localhost. Not suitable for multi-user or networked deployment without adding auth middleware.

---

## Setup

### 1. Create and activate a virtual environment

Windows:

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
```

macOS / Linux:

```bash
python -m venv .venv
source .venv/bin/activate
```

### 2. Install Python dependencies

```bash
pip install -r requirements.txt
```

### 3. Install Ollama and pull models

```bash
ollama pull nomic-embed-text
ollama pull llama3.2:3b
```

### 4. Start the FastAPI backend

```bash
uvicorn app.main:app --reload
```

### 5. Start the Streamlit frontend

```bash
streamlit run streamlit_app.py
```

Then open:

- Streamlit UI: `http://localhost:8501`
- API docs: `http://127.0.0.1:8000/docs`

## Usage

### Streamlit interface

- Upload one or more PDF files in the sidebar
- View indexed document names
- Ask questions in the chat input
- Reset the index to clear all documents

### REST API

- `POST /upload` — upload and index a PDF file
- `POST /ask` — ask a question across all indexed PDFs
- `GET /documents` — list indexed document names
- `DELETE /reset` — delete the current index
- `POST /build-index` — index `data/sample.pdf` (if present)

## How It Works

1. PDF text is extracted with PyMuPDF.
2. Text is split into overlapping chunks (600 characters, 120-character overlap).
3. Chunks are embedded using `nomic-embed-text`.
4. Embeddings are stored in a persistent FAISS index with source filename metadata.
5. At query time, the question is embedded, the top 4 relevant chunks are retrieved,
   and `llama3.2:3b` generates the final answer with source citations.

## Notes

- Ollama must be running locally at `http://localhost:11434`.
- Duplicate filenames are skipped at upload time.
- The `faiss_index/` directory is created automatically on first upload.
- If the index does not exist, upload at least one PDF before querying.

## Troubleshooting

- If the frontend cannot connect to the API, verify the FastAPI server is running.
- If embeddings or generation fail, verify Ollama is running and models are pulled.
- If `data/sample.pdf` is missing, `POST /build-index` will fail until a valid PDF exists.
