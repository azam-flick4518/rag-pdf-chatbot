import streamlit as st
import requests

API = "http://localhost:8000"

st.set_page_config(page_title="RAG PDF Chatbot", page_icon="📄", layout="wide")
st.title("📄 Multi-PDF RAG Chatbot")
st.caption("100% local — powered by Ollama + FAISS + FastAPI")

# ── Sidebar ────────────────────────────────────────────────────────────────
with st.sidebar:
    st.header("📁 Upload PDFs")
    uploaded_files = st.file_uploader(
        "Upload one or more PDFs",
        type="pdf",
        accept_multiple_files=True
    )

    if uploaded_files:
        # Get already-indexed docs first
        try:
            indexed = requests.get(f"{API}/documents", timeout=10).json().get("documents", [])
        except:
            indexed = []

        for f in uploaded_files:
            if f.name in indexed:
                continue  # skip — already indexed
            with st.spinner(f"Indexing {f.name}..."):
                try:
                    r = requests.post(
                        f"{API}/upload",
                        files={"file": (f.name, f.getvalue(), "application/pdf")},
                        timeout=300
                    )
                    if r.status_code == 200:
                        st.success(f"✅ {f.name}")
                    else:
                        st.error(f"❌ {f.name}: {r.json().get('detail', 'Unknown error')}")
                except requests.exceptions.ConnectionError:
                    st.error("Cannot connect to API. Is FastAPI running?")

    st.divider()

    st.header("📚 Indexed Documents")
    try:
        docs_resp = requests.get(f"{API}/documents", timeout=10)
        docs = docs_resp.json().get("documents", [])
        if docs:
            for doc in sorted(docs):
                st.markdown(f"- 📄 {doc}")
        else:
            st.caption("No documents indexed yet.")
    except requests.exceptions.ConnectionError:
        st.caption("API not reachable.")

    st.divider()

    if st.button("🗑️ Reset Index", use_container_width=True):
        try:
            requests.delete(f"{API}/reset", timeout=10)
            st.session_state.messages = []
            st.warning("Index cleared.")
            st.rerun()
        except requests.exceptions.ConnectionError:
            st.error("API not reachable.")

# ── Chat Interface ──────────────────────────────────────────────────────────
if "messages" not in st.session_state:
    st.session_state.messages = []

# Render chat history
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if msg.get("sources"):
            st.caption(f"📎 Sources: {', '.join(msg['sources'])}")

# Chat input
if prompt := st.chat_input("Ask a question about your documents..."):

    # Check documents exist before querying
    try:
        docs = requests.get(f"{API}/documents", timeout=10).json().get("documents", [])
    except:
        docs = []

    if not docs:
        st.warning("⚠️ Upload at least one PDF before asking questions.")
    else:
        # Show user message
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        # Get answer
        with st.chat_message("assistant"):
            with st.spinner("Searching documents and generating answer..."):
                try:
                    r = requests.post(
                        f"{API}/ask",
                        json={"question": prompt},
                        timeout=180
                    )
                    if r.status_code == 200:
                        data = r.json()
                        answer = data["answer"]
                        sources = data.get("sources", [])
                    else:
                        answer = f"Error: {r.json().get('detail', 'Unknown error')}"
                        sources = []
                except requests.exceptions.ConnectionError:
                    answer = "Cannot connect to API. Is FastAPI running?"
                    sources = []

            st.markdown(answer)
            if sources:
                st.caption(f"📎 Sources: {', '.join(sources)}")

        st.session_state.messages.append({
            "role": "assistant",
            "content": answer,
            "sources": sources
        })