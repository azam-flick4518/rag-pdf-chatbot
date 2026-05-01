# Multi-PDF RAG Chatbot (Ollama + FAISS + FastAPI + Streamlit)

A fully local Retrieval-Augmented Generation (RAG) chatbot for PDF documents.
Upload one or more PDFs, index them locally, and ask natural language questions with
source citations for each answer.

## Highlights

- Local-first architecture: Ollama models run on your machine
- Multi-PDF ingestion: upload and query across multiple documents
- Persistent FAISS index: survives restarts and supports incremental indexing
- Source-aware answers: each response includes document citations
- Streamlit chat UI + FastAPI backend

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
├── data/                # Optional sample data or local PDFs
├── faiss_index/         # Persistent FAISS index and metadata
├── streamlit_app.py     # Streamlit frontend chat interface
├── requirements.txt
└── README.md
```

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

### 3. Install Ollama and download models

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
- Reset the index if you want to clear all documents

### REST API

- `POST /upload` — upload and index a PDF file
- `POST /ask` — ask a question across all indexed PDFs
- `GET /documents` — list indexed document names
- `DELETE /reset` — delete the current index
- `POST /build-index` — index `data/sample.pdf` (if present)

## How It Works

1. PDF text is extracted with PyMuPDF.
2. Text is split into overlapping chunks (600 characters with 120-character overlap).
3. Chunks are embedded using `nomic-embed-text`.
4. Embeddings are stored in a FAISS index with source filename metadata.
5. At query time, the question is embedded, the top 4 relevant chunks are retrieved,
   and `llama3.2:3b` generates the final answer with source citations.

## Notes

- The app expects Ollama to be running locally at `http://localhost:11434`.
- Duplicate filenames are skipped in the Streamlit upload flow.
- If the index does not exist, you must upload at least one PDF before querying.
- A `faiss_index/` directory is created automatically to store the index and metadata.

## Requirements

```text
fastapi
uvicorn
faiss-cpu
pymupdf
numpy
requests
pydantic
streamlit
```

## Troubleshooting

- If the frontend cannot connect to the API, verify the FastAPI server is running.
- If embeddings or generation fail, verify Ollama is running and the required models are pulled.
- If `data/sample.pdf` is missing, `POST /build-index` will fail until a valid PDF exists.

## Summary

This repository implements a local RAG PDF chatbot with a clean separation between
PDF ingestion, vector retrieval, and LLM generation. It is optimized for local use
with Ollama and includes both a user-friendly Streamlit chat UI and a REST API.

