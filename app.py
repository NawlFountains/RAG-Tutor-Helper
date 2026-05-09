import streamlit as st
import tempfile
import os
import re
import unicodedata
import time

from langchain_community.document_loaders import PyMuPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_qdrant import QdrantVectorStore
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.output_parsers import StrOutputParser
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory
from operator import itemgetter

# ── Page config ────────────────────────────────────────────────────────────────
st.set_page_config(page_title="RAG Tutor", page_icon="🎓", layout="wide")
st.title("🎓 RAG Tutor")
st.caption("Upload your PDFs and ask questions about them.")

# ── Session state defaults ──────────────────────────────────────────────────────
for key, default in {
    "chain": None,
    "chat_history": [],
    "store": {},
    "ready": False,
}.items():
    if key not in st.session_state:
        st.session_state[key] = default


# ── Helpers ─────────────────────────────────────────────────────────────────────
def clean_text(text: str) -> str:
    """Fix broken encoding from LaTeX-generated PDFs."""
    text = unicodedata.normalize("NFKC", text)
    replacements = {
        "\u00b4a": "á", "\u00b4e": "é", "\u00b4i": "í",
        "\u00b4o": "ó", "\u00b4u": "ú", "\u00b4on": "ión",
        "\u00b4A": "Á", "\u00b4E": "É", "\u00b4I": "Í",
        "\u00b4O": "Ó", "\u00b4U": "Ú",
        "~n": "ñ",  "~N": "Ñ",
    }
    for broken, fixed in replacements.items():
        text = text.replace(broken, fixed)
    return text


@st.cache_resource(show_spinner="Loading embedding model (first time may take a while)...")
def load_embeddings():
    return HuggingFaceEmbeddings(
        model_name="intfloat/multilingual-e5-large",
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True, "batch_size": 8},
    )

def load_pdfs(uploaded_files):
    """Load, clean, chunk, embed and index uploaded PDFs."""
    pages = []

    # Save uploads to temp files (PyMuPDFLoader needs file paths)
    with tempfile.TemporaryDirectory() as tmp_dir:
        for uploaded_file in uploaded_files:
            path = os.path.join(tmp_dir, uploaded_file.name)
            with open(path, "wb") as f:
                f.write(uploaded_file.read())
            loader = PyMuPDFLoader(path)
            pages.extend(loader.load())

    # Clean encoding artifacts
    for doc in pages:
        doc.page_content = clean_text(doc.page_content)
    return pages

def chunk_documents(pages):
    # Chunk
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        separators=["\n\n", "\n", ".", "!", "¡", "¿", "?", ",", " "],
    )
    return splitter.split_documents(pages)

def build_vectorstore(splits):
    # Embed + store
    embeddings = load_embeddings()
    client = QdrantClient(":memory:")
    client.create_collection(
        collection_name="rag_docs",
        vectors_config=VectorParams(size=1024, distance=Distance.COSINE),
    )
    vectorstore = QdrantVectorStore(
        client=client, collection_name="rag_docs", embedding=embeddings
    )
    vectorstore.add_documents(splits)
    return vectorstore

def build_chain(uploaded_files, groq_api_key: str):
    # Retriever
    retriever = vectorstore.as_retriever(
        search_type="similarity", search_kwargs={"k": 5}
    )

    # LLM
    llm = ChatGroq(model="llama-3.3-70b-versatile", api_key=groq_api_key)

    # Prompt with memory
    prompt = ChatPromptTemplate.from_messages([
        ("system", """You are a helpful assistant specialized in mathematics and engineering.
The context may contain mathematical formulas with imperfect formatting — interpret them carefully.
Answer using ONLY the context below. If the answer is not in the context, say "I don't know / No lo sé".
Answer in the SAME LANGUAGE the question was asked in.

Context:
{context}"""),
        MessagesPlaceholder(variable_name="history"),
        ("human", "{question}"),
    ])

    def format_docs(docs):
        return "\n\n".join(doc.page_content for doc in docs)

    rag_chain = (
        {
            "context": itemgetter("question") | retriever | format_docs,
            "question": itemgetter("question"),
            "history": itemgetter("history"),
        }
        | prompt
        | llm
        | StrOutputParser()
    )

    store = {}

    def get_session_history(session_id: str):
        if session_id not in store:
            store[session_id] = ChatMessageHistory()
        return store[session_id]

    chain_with_history = RunnableWithMessageHistory(
        rag_chain,
        get_session_history,
        input_messages_key="question",
        history_messages_key="history",
    )

    return chain_with_history, store, len(splits)


# ── Sidebar ─────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.header("⚙️ Setup")

    groq_api_key = st.secrets["GROQ_API_KEY"] 

    uploaded_files = st.file_uploader(
        "Upload PDF documents",
        type="pdf",
        accept_multiple_files=True,
    )

    if uploaded_files:
        total_size = sum(f.size for f in uploaded_files)
        total_mb = total_size / (1024 * 1024)
        st.caption(f"📄 {len(uploaded_files)} file(s) — {total_mb:.1f} MB")
        if total_mb > 10:
            st.warning("⚠️ Large files detected — processing may take several minutes on first run.")

    process_btn = st.button(
        "🚀 Process Documents",
        disabled=not uploaded_files,
        use_container_width=True,
    )

    if process_btn:
        try:
            with st.status("Processing...", expanded=True) as status:
                st.write("📄 Loading and cleaning PDFs...")
                pages = load_pdfs(uploaded_files)
                st.write(f"✅ Loaded {len(pages)} pages")

                st.write("✂️ Chunking documents...")
                splits = chunk_documents(pages)
                st.write(f"✅ Created {len(splits)} chunks")

                st.write("🧠 Generating embeddings (this may take a few minutes)...")
                vectorstore = build_vectorstore(splits)
                st.write("✅ Embeddings done")

                st.write("🔗 Building RAG chain...")
                chain, store = build_chain(vectorstore, groq_api_key)
                st.write("✅ Chain ready")

                st.session_state.chain = chain
                st.session_state.store = store
                st.session_state.chat_history = []
                st.session_state.ready = True
                status.update(label="✅ Ready to chat!", state="complete")

        except MemoryError:
            st.error("❌ Out of memory — try uploading fewer or smaller PDFs.")
        except Exception as e:
            st.error(f"❌ Error: {e}")

    if st.session_state.ready:
        st.divider()
        if st.button("🗑️ Clear chat", use_container_width=True):
            st.session_state.chat_history = []
            st.session_state.store = {}
            st.rerun()


# ── Chat area ───────────────────────────────────────────────────────────────────
if not st.session_state.ready:
    st.info("👈 Upload PDFs, and click **Process Documents** to start.")
else:
    st.info("💡 Documents are processed per session — re-upload if you refresh the page.")
    # Render chat history
    for msg in st.session_state.chat_history:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    # Input
    if question := st.chat_input("Ask something about your documents..."):
        # Show user message
        st.session_state.chat_history.append({"role": "user", "content": question})
        with st.chat_message("user"):
            st.markdown(question)

        # Get answer
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                answer = st.session_state.chain.invoke(
                    {"question": question},
                    config={"configurable": {"session_id": "user_session"}},
                )
            st.markdown(answer)

        st.session_state.chat_history.append({"role": "assistant", "content": answer})
