from pathlib import Path
import streamlit as st
import rag_system as rag   # must match the name of your RAG file (without .py)

st.set_page_config(page_title="Local Document Q&A", page_icon="📄")
UPLOAD_DIR = Path("uploads")
UPLOAD_DIR.mkdir(exist_ok=True)

st.title("📄 Local Document Q&A")
st.caption("Everything runs on your computer: sentence-transformers + ChromaDB + Ollama")

# ---------- memory between reruns ----------
if "ready" not in st.session_state:
    st.session_state.ready = False      # True after Submit worked
    st.session_state.top_k = rag.TOP_K
    st.session_state.max_distance = rag.MAX_DISTANCE
    st.session_state.indexed = {}       # {file name: number of chunks}
    st.session_state.history = []       # chat messages

# ---------- Step 1: upload ----------
st.subheader("Step 1: Upload your documents")
uploaded = st.file_uploader(
    "PDF, DOCX or TXT", type=["pdf", "docx", "txt"], accept_multiple_files=True
)

# ---------- Step 2: settings ----------
st.subheader("Step 2: Choose settings")
col1, col2 = st.columns(2)
top_k = col1.slider("Passages to retrieve (top-k)", 1, 8, st.session_state.top_k)
max_distance = col2.slider(
    "Max distance (lower = stricter)", 0.3, 1.0, st.session_state.max_distance, 0.05
)
if max_distance < 0.4:
    st.warning("⚠️ Threshold is very strict. You may see 'not found' responses.")

# ---------- Step 3: submit ----------
st.subheader("Step 3: Submit")
if st.button("Submit", type="primary", disabled=not uploaded):
    indexed = {}
    for f in uploaded:
        path = UPLOAD_DIR / f.name
        path.write_bytes(f.getbuffer())
        try:
            with st.spinner(f"Indexing {f.name} ..."):
                indexed[f.name] = rag.ingest(path)
        except ValueError as e:
            st.error(f"{f.name}: {e}")

    if indexed:
        st.session_state.top_k = top_k
        st.session_state.max_distance = max_distance
        st.session_state.indexed = indexed
        st.session_state.history = []
        st.session_state.ready = True
    else:
        st.session_state.ready = False

if st.session_state.ready:
    files = ", ".join(f"{n} ({c} chunks)" for n, c in st.session_state.indexed.items())
    st.success(
        f"Ready. Indexed: {files}. "
        f"Using top-k = {st.session_state.top_k}, max distance = {st.session_state.max_distance}"
    )

# ---------- Step 4: ask ----------
st.subheader("Step 4: Ask a question")


def show(item):
    st.markdown(item["answer"])
    if item["hits"]:
        with st.expander(f"Sources ({len(item['hits'])})"):
            for i, h in enumerate(item["hits"], start=1):
                page = f" · page {h['page']}" if h["page"] else ""
                st.markdown(f"**[{i}] {h['source']}**{page} · distance `{h['distance']:.2f}`")
                st.caption(h["text"])
    st.caption(f"⏱ {item['secs']:.1f}s")


if not st.session_state.ready:
    st.info("Upload a file and click Submit to start asking questions.")
else:
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
            with st.spinner("Thinking ..."):
                answer, hits, secs = rag.ask(
                    question, st.session_state.top_k, st.session_state.max_distance
                )
            item = {"question": question, "answer": answer, "hits": hits, "secs": secs}
            st.session_state.history.append(item)
            show(item)