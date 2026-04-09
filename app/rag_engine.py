import os
import pickle
import faiss
import numpy as np
import requests
from pypdf import PdfReader

PDF_PATH = "data/sample.pdf"

# Separate the roles
EMBED_MODEL = "nomic-embed-text"
CHAT_MODEL = "llama3.2:3b" # Or "qwen2.5-coder:1.5b"
OLLAMA_URL = "http://localhost:11434"

INDEX_PATH = "faiss_index/index.faiss"
CHUNKS_PATH = "faiss_index/chunks.pkl"


def extract_text_from_pdf(pdf_path):
    reader = PdfReader(pdf_path)
    text = []
    for page in reader.pages:
        page_text = page.extract_text()
        if page_text:
            text.append(page_text)
    return "\n".join(text)


def chunk_text(text, chunk_size=600, overlap=120, separator="\n\n"):
    chunks = []
    start = 0
    while start < len(text):
        end = min(start + chunk_size, len(text))
        if text[end-1] == separator or text[end-1] == "\n":  # Check for separator characters at the end of each chunk
            end -= 1
        chunks.append(text[start:end])
        start += chunk_size - overlap
    return chunks


def get_embedding(text):
    response = requests.post(
        f"{OLLAMA_URL}/api/embeddings",
        json={"model": EMBED_MODEL, "prompt": text}, # Use EMBED_MODEL
        timeout=120
    )
    response.raise_for_status()
    return response.json()["embedding"]


def build_faiss_index(chunks):
    embeddings = []
    for i, chunk in enumerate(chunks, start=1):
        print(f"Embedding chunk {i}/{len(chunks)}")
        embeddings.append(get_embedding(chunk))

    vectors = np.array(embeddings).astype("float32")
    index = faiss.IndexFlatL2(vectors.shape[1])
    index.add(vectors)
    return index


def save_index(index, chunks):
    os.makedirs("faiss_index", exist_ok=True)
    faiss.write_index(index, INDEX_PATH)
    with open(CHUNKS_PATH, "wb") as f:
        pickle.dump(chunks, f)


def load_index():
    if os.path.exists(INDEX_PATH) and os.path.exists(CHUNKS_PATH):
        index = faiss.read_index(INDEX_PATH)
        with open(CHUNKS_PATH, "rb") as f:
            chunks = pickle.load(f)
        return index, chunks
    return None, None


def prepare_index(pdf_path=PDF_PATH):
    text = extract_text_from_pdf(pdf_path)
    if not text.strip():
        raise ValueError("No readable text found in the PDF.")
    chunks = chunk_text(text, chunk_size=600, overlap=120, separator="\n\n")
    index = build_faiss_index(chunks)
    save_index(index, chunks)
    return {"message": "Index built successfully", "chunks": len(chunks)}


def retrieve_chunks(query, index, chunks, top_k=3):
    query_vector = np.array([get_embedding(query)]).astype("float32")
    distances, indices = index.search(query_vector, top_k)

    results = []
    for idx in indices[0]:
        if 0 <= idx < len(chunks):
            results.append(chunks[idx])
    return results


def ask_llm(context, question):
    prompt = f"""
You are a helpful assistant.
Answer the question using only the context below.
If the answer is not found in the context, say: "I couldn't find that in the PDF."

Context:
{context}

Question:
{question}
"""
    response = requests.post(
        f"{OLLAMA_URL}/api/generate",
        json={
            "model": CHAT_MODEL,
            "prompt": prompt,
            "stream": False
        },
        timeout=180
    )
    response.raise_for_status()
    return response.json()["response"]


def answer_query(question):
    index, chunks = load_index()
    if index is None or chunks is None:
        raise ValueError("Index not found. Please build the index first.")

    relevant_chunks = retrieve_chunks(question, index, chunks, top_k=3)
    context = "\n\n".join(relevant_chunks)
    answer = ask_llm(context, question)

    return {
        "question": question,
        "answer": answer,
        "retrieved_chunks": relevant_chunks
    }