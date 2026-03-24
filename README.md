# Local RAG PDF Chatbot using Ollama, FAISS, FastAPI, and Python

## Overview
This project is a local Retrieval-Augmented Generation (RAG) PDF chatbot built to explore practical AI retrieval workflows and backend AI integration patterns.

It reads PDF documents, extracts text, splits the text into chunks, converts chunks into embeddings using Ollama, stores them in a FAISS vector index, and exposes question-answering through a FastAPI service.

## Features
- PDF text extraction using 'pypdf'
- Text chunking for retrieval
- Local embeddings using Ollama
- Vector search using FAISS
- Question-answering via local LLM
- FastAPI endpoints for index building and querying
- Swagger UI for API testing

## Tech Stack
- Python
- FastAPI
- Ollama
- FAISS
- NumPy
- Requests
- pypdf

## Project Structure
```text
rag-pdf-chatbot/
├── app/
│   ├── main.py
│   ├── rag_engine.py
│   └── models.py
├── .gitignore
├── README.md
├── requirements.txt

## How It Works

1. Read text from a PDF  
2. Split text into overlapping chunks  
3. Generate embeddings for each chunk using Ollama  
4. Store vectors in FAISS  
5. Embed the user question  
6. Retrieve the most relevant chunks  
7. Send retrieved context to the LLM  
8. Return a grounded answer  

---

## API Endpoints

### `GET /`
Health check endpoint  

---

### `POST /build-index`
Builds the FAISS index from the PDF  

---

### `POST /ask`
Accepts a question and returns:
- answer  
- question  
- retrieved chunks  

#### Example request:
```json
{
  "question": "What is this document about?"
}

Run Locally
1. Create and activate virtual environment
python -m venv .venv
.venv\Scripts\activate

2. Install dependencies
pip install -r requirements.txt

3. Start Ollama

Make sure Ollama is running locally and the model is available:

ollama list

4. Run the API

uvicorn app.main:app --reload

5. Open Swagger docs
http://127.0.0.1:8000/docs