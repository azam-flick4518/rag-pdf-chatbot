import os
import pickle
import faiss
import numpy as np
import requests
import fitz  # pymupdf - handles more PDF types than pypdf
from pathlib import Path

# Separate the roles
EMBED_MODEL = "nomic-embed-text"
CHAT_MODEL = "llama3.2:3b"
OLLAMA_URL = "http://localhost:11434"

INDEX_PATH = "faiss_index/index.faiss"
META_PATH = "faiss_index/meta.pkl"  # now stores chunks + sources together


def extract_text_from_bytes(contents: bytes) -> str:
    """Extract text from PDF bytes using pymupdf."""
    doc = fitz.open(stream=contents, filetype="pdf")
    text = "\n\n".join(page.get_text() for page in doc)
    return text


def extract_text_from_path(pdf_path: str) -> str:
    """Extract text from PDF file path using pymupdf."""
    doc = fitz.open(pdf_path)
    text = "\n\n".join(page.get_text() for page in doc)
    return text


def chunk_text(text, chunk_size=600, overlap=120):
    """Split text into overlapping chunks."""
    chunks = []
    start = 0
    while start < len(text):
        end = min(start + chunk_size, len(text))
        chunks.append(text[start:end])
        start += chunk_size - overlap
    return chunks


def get_embedding(text: str) -> list:
    """Get embedding vector from Ollama."""
    response = requests.post(
        f"{OLLAMA_URL}/api/embeddings",
        json={"model": EMBED_MODEL, "prompt": text},
        timeout=120
    )
    response.raise_for_status()
    return response.json()["embedding"]


def build_faiss_index(embeddings: np.ndarray):
    """Create a FAISS index from embeddings."""
    index = faiss.IndexFlatL2(embeddings.shape[1])
    index.add(embeddings)
    return index


def save_index(index, chunks: list, sources: list):
    """Save FAISS index and metadata to disk."""
    Path("faiss_index").mkdir(exist_ok=True)
    faiss.write_index(index, INDEX_PATH)
    with open(META_PATH, "wb") as f:
        pickle.dump({"chunks": chunks, "sources": sources}, f)


def load_index():
    """Load FAISS index and metadata from disk."""
    if os.path.exists(INDEX_PATH) and os.path.exists(META_PATH):
        index = faiss.read_index(INDEX_PATH)
        with open(META_PATH, "rb") as f:
            data = pickle.load(f)
        return index, data["chunks"], data["sources"]
    return None, [], []


def add_document(contents: bytes, filename: str) -> str:
    """
    Add a new PDF document to the existing index.
    Supports multiple PDFs — each chunk is tagged with its source filename.
    """
    text = extract_text_from_bytes(contents)
    if not text.strip():
        raise ValueError(f"No readable text found in {filename}.")

    new_chunks = chunk_text(text, chunk_size=600, overlap=120)
    new_sources = [filename] * len(new_chunks)

    # Embed new chunks
    print(f"Embedding {len(new_chunks)} chunks from {filename}...")
    embeddings = []
    for i, chunk in enumerate(new_chunks, 1):
        print(f"  Chunk {i}/{len(new_chunks)}")
        embeddings.append(get_embedding(chunk))

    new_vectors = np.array(embeddings, dtype="float32")

    # Load existing index if present, otherwise create fresh
    index, existing_chunks, existing_sources = load_index()

    if index is None:
        index = build_faiss_index(new_vectors)
    else:
        index.add(new_vectors)

    all_chunks = existing_chunks + new_chunks
    all_sources = existing_sources + new_sources

    save_index(index, all_chunks, all_sources)
    return f"Indexed {len(new_chunks)} chunks from {filename}"


def prepare_index(pdf_path: str = "data/sample.pdf") -> dict:
    """Build index from a local file path (kept for backward compatibility)."""
    with open(pdf_path, "rb") as f:
        contents = f.read()
    filename = Path(pdf_path).name
    message = add_document(contents, filename)
    return {"message": message}


def list_documents() -> list:
    """Return list of unique document names in the index."""
    _, _, sources = load_index()
    return list(set(sources))


def reset_index():
    """Delete the FAISS index and metadata from disk."""
    if os.path.exists(INDEX_PATH):
        os.remove(INDEX_PATH)
    if os.path.exists(META_PATH):
        os.remove(META_PATH)


def retrieve_chunks(query: str, index, chunks: list, sources: list, top_k: int = 4):
    """Retrieve top_k most relevant chunks for a query."""
    query_vector = np.array([get_embedding(query)], dtype="float32")
    distances, indices = index.search(query_vector, top_k)

    results, result_sources = [], []
    for idx in indices[0]:
        if 0 <= idx < len(chunks):
            results.append(chunks[idx])
            result_sources.append(sources[idx])
    return results, result_sources


def ask_llm(context: str, question: str, source_names: list) -> str:
    """Send context + question to local LLM and return answer."""
    sources_str = ", ".join(set(source_names))
    prompt = f"""You are a helpful assistant.
Answer the question using ONLY the context below.
If the answer is not found in the context, say: "I couldn't find that in the provided documents."
At the end of your answer, always cite: Sources: {sources_str}

Context:
{context}

Question:
{question}

Answer:"""

    response = requests.post(
        f"{OLLAMA_URL}/api/generate",
        json={"model": CHAT_MODEL, "prompt": prompt, "stream": False},
        timeout=180
    )
    response.raise_for_status()
    return response.json()["response"]


def answer_query(question: str) -> dict:
    """Main query function — retrieves chunks and generates answer."""
    index, chunks, sources = load_index()
    if index is None:
        raise ValueError("No index found. Please upload a PDF first.")

    relevant_chunks, relevant_sources = retrieve_chunks(question, index, chunks, sources)
    context = "\n\n".join(relevant_chunks)
    answer = ask_llm(context, question, relevant_sources)

    return {
        "question": question,
        "answer": answer,
        "sources": list(set(relevant_sources))
    }