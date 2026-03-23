import os
import pickle
import faiss
import numpy as np
import requests
from pypdf import PdfReader


PDF_PATH = "data/sample.pdf"
MODEL_NAME = "llama3.2:3b"
OLLAMA_URL = "http://localhost:11434"

INDEX_PATH = "faiss_index/index.faiss"
CHUNKS_PATH = "faiss_index/chunks.pkl"


def extract_text_from_pdf(pdf_path):
    reader = PdfReader(pdf_path)
    text = []
    for i, page in enumerate(reader.pages, start=1):
        page_text = page.extract_text()
        if page_text:
            text.append(page_text)
        else:
            print(f"Warning: no text extracted from page {i}")
    return "\n".join(text)


def chunk_text(text, chunk_size=600, overlap=120):
    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunks.append(text[start:end])
        start += chunk_size - overlap
    return chunks


def get_embedding(text):
    response = requests.post(
        f"{OLLAMA_URL}/api/embeddings",
        json={"model": MODEL_NAME, "prompt": text},
        timeout=120
    )
    response.raise_for_status()
    return response.json()["embedding"]


def build_faiss_index(chunks):
    embeddings = []
    total = len(chunks)

    for i, chunk in enumerate(chunks, start=1):
        print(f"Embedding chunk {i}/{total}")
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
            "model": MODEL_NAME,
            "prompt": prompt,
            "stream": False
        },
        timeout=180
    )
    response.raise_for_status()
    return response.json()["response"]


def main():
    if not os.path.exists(PDF_PATH):
        print(f"PDF not found at: {PDF_PATH}")
        return

    index, stored_chunks = load_index()

    if index is None or stored_chunks is None:
        print("Reading PDF...")
        text = extract_text_from_pdf(PDF_PATH)

        if not text.strip():
            print("No readable text found in the PDF.")
            return

        print("Chunking text...")
        chunks = chunk_text(text)

        print("Building embeddings and FAISS index...")
        index = build_faiss_index(chunks)
        stored_chunks = chunks

        print("Saving index...")
        save_index(index, stored_chunks)
    else:
        print("Loaded existing FAISS index.")

    print("\nRAG PDF Chatbot is ready. Type 'exit' to quit.\n")

    while True:
        query = input("Ask a question about the PDF: ").strip()
        if query.lower() in {"exit", "quit"}:
            print("Goodbye.")
            break

        try:
            relevant_chunks = retrieve_chunks(query, index, stored_chunks, top_k=3)

            print("\nRetrieved chunks:\n")
            for i, chunk in enumerate(relevant_chunks, start=1):
                print(f"[Chunk {i}] {chunk[:300]}")
                print("-" * 50)

            context = "\n\n".join(relevant_chunks)
            answer = ask_llm(context, query)

            print("\nAnswer:\n")
            print(answer)
            print("\n" + "=" * 70 + "\n")

        except Exception as e:
            print(f"\nError: {e}\n")


if __name__ == "__main__":
    main()