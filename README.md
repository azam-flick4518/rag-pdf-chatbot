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