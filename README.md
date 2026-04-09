# Local RAG PDF Chatbot (Ollama + FAISS + FastAPI)

A production-grade local Retrieval-Augmented Generation (RAG) system using dedicated embedding and chat models.

## 🚀 Features
- **Dual-Model Architecture**: Uses `nomic-embed-text` for retrieval and `llama3.2:3b` for generation.
- **Optimized Embeddings**: Local vector search powered by FAISS.
- **FastAPI Backend**: Professional API structure with Swagger documentation.
- **Zero Cloud Costs**: Runs entirely on local hardware via Ollama.

## 🛠️ Tech Stack
- **AI/LLM**: Ollama (Llama 3.2 & Nomic Embeddings)
- **Vector DB**: FAISS (Facebook AI Similarity Search)
- **Framework**: FastAPI / Uvicorn
- **Language**: Python 3.9+

## 📁 Project Structure
```text
rag_pdf_chatbot/
├── app/
│   ├── main.py          # FastAPI routes
│   ├── rag_engine.py    # Core RAG logic & indexing
│   └── models.py        # Pydantic request models
├── data/
│   └── sample.pdf       # Source document
├── faiss_index/         # Generated vector store (Auto-created)
├── requirements.txt
└── README.md

⚙️ Setup Instructions
1. Prepare Environment

python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt

2. Configure Ollama
Ensure Ollama is running and pull the necessary models:

ollama pull nomic-embed-text
ollama pull llama3.2:3b

3. Run the API

uvicorn app.main:app --reload

🔌 API Usage
Access the interactive documentation at: http://127.0.0.1:8000/docs

Key Endpoints:
POST /build-index: Processes the PDF and creates the FAISS vector store.

POST /ask: Queries the document.

Payload: {"question": "What are the key takeaways?"}

📝 Maintenance
To clear the database and re-index a new PDF:

Delete the faiss_index/ folder.

Place the new PDF in data/sample.pdf.

Run the Build Index command again.