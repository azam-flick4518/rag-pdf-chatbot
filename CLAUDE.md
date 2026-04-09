# CLAUDE.md - RAG PDF Chatbot Project

## Project Overview
A FastAPI-based RAG application using FAISS for vector storage and Ollama (Qwen 2.5 Coder) for generating answers from PDF content.

## Development Commands
- **Install Dependencies**: `pip install -r requirements.txt`
- **Run API Server**: `uvicorn app.main:app --reload`
- **Build Index**: `curl -X POST http://localhost:8000/build-index`
- **Ask Question**: `curl -X POST http://localhost:8000/ask -d '{"question": "your question here"}'`

## Code Style
- Use Python 3.9+ type hinting.
- Follow FastAPI best practices for error handling (HTTPException).
- Keep RAG logic inside `app/rag_engine.py` and models in `app/models.py`.