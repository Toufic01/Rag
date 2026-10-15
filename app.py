"""
app.py  -  the web page (Streamlit). Run with:   streamlit run app.py
"""
from pathlib import Path
import streamlit as st
import rag_system as rag

st.set_page_config(page_title="Local Document Q&A", page_icon="📄")
UPLOAD_DIR = Path("uploads")
UPLOAD_DIR.mkdir(exist_ok=True)

# ---------------------------------------------------------------- sidebar
with st.sidebar:
    st.header("Settings")
    top_k = st.slider("Passages to retrieve (top-k)", 1, 8, rag.TOP_K)

    # Safe minimum set to 0.3 to prevent accidental zero-match lockouts
    max_distance = st.slider("Max distance (lower = stricter)", 0.3, 1.0, rag.MAX_DISTANCE, 0.05)
    if max_distance < 0.4:
        st.warning("⚠️ Threshold is very strict. You may see 'not found' responses.")

    st.header("Documents")
    uploaded = st.file_uploader("Upload PDF, DOCX or TXT", type=["pdf", "docx", "txt"], accept_multiple_files=True)

    if st.button("Index uploaded files", disabled=not uploaded):
        for f in uploaded:
            path = UPLOAD_DIR / f.name
            path.write_bytes(f.getbuffer())
            try:
                with st.spinner(f"Indexing {f.name} ..."):
                    n = rag.ingest(path)
                st.success(f"{f.name}: {n} chunks")
            except ValueError as e:
                st.error(str(e))

    if st.button("Index sample documents"):
        with st.spinner("Indexing ..."):
            for name, n in rag.ingest_folder("data").items():
                st.success(f"{name}: {n} chunks")

    st.write(f"**Chunks in database:** {rag.collection.count()}")
    if st.button("Clear database"):
        ids = rag.collection.get()["ids"]
        if ids:
            rag.collection.delete(ids=ids)
        st.rerun()

# ------------------------------------------------------------------- main
st.title("📄 Local Document Q&A")
st.caption("Everything runs on your computer: sentence-transformers + ChromaDB + Ollama")

def show(item):
    st.markdown(item["answer"])
    if item["hits"]:
        with st.expander(f"Sources ({len(item['hits'])})"):
            for i, h in enumerate(item["hits"], start=1):
                page = f" · page {h['page']}" if h["page"] else ""
                st.markdown(f"**[{i}] {h['source']}**{page} · distance `{h['distance']:.2f}`")
                st.caption(h["text"])
    st.caption(f"⏱ {item['secs']:.1f}s")

if "history" not in st.session_state:
    st.session_state.history = []

for item in st.session_state.history:
    with st.chat_message("user"):
        st.markdown(item["question"])
    with st.chat_message("assistant"):
        show(item)

question = st.chat_input("Ask a question about your documents")
if question:
    with st.chat_message("user"):
        st.markdown(question)
    with st.chat_message("assistant"):
        if rag.collection.count() == 0:
            st.warning("Index a document first (sidebar).")
        else:
            with st.spinner("Thinking ..."):
                answer, hits, secs = rag.ask(question, top_k, max_distance)
            item = {"question": question, "answer": answer, "hits": hits, "secs": secs}
            st.session_state.history.append(item)
            show(item)