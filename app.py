import streamlit as st
import tempfile
import os
import unicodedata
import uuid
import traceback

from langchain_community.document_loaders import PyMuPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_qdrant import QdrantVectorStore
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import HumanMessage, AIMessage
from langchain.chains import create_history_aware_retriever, create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain

# ── Page config ────────────────────────────────────────────────────────────────
st.set_page_config(page_title="RAG Tutor", page_icon="🎓", layout="wide")
st.title("🎓 RAG Tutor")
st.caption("Upload your PDFs and ask questions about them.")

# ── Session state defaults ──────────────────────────────────────────────────────
for key, default in {
    "chain": None,
    "chat_history": [],
    "lc_chat_history": [],
    "lc_chat_history": [],
    "store": {},
    "ready": False,
    "session_id": str(uuid.uuid4()),
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
def load_embeddings(model_name: str):
    return HuggingFaceEmbeddings(
        model_name=model_name,
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

def build_vectorstore(splits, model_option: str):
    config = MODEL_CONFIG[model_option]
    collection_name = f"session_{st.session_state.session_id}"

    # Embed + store
    embeddings = load_embeddings(config["name"])
    client = QdrantClient(
            url=st.secrets["QDRANT_URL"],
            api_key=st.secrets["QDRANT_API_KEY"],
    )

    # Referesh documents on restart but can keep adding on same session
    if client.collection_exists(collection_name):
        client.delete_collection(collection_name)

    client.create_collection(
        collection_name=collection_name,
        vectors_config=VectorParams(
            size=config["size"], distance=Distance.COSINE
        ),
    )
    vectorstore = QdrantVectorStore(
        client=client, collection_name=collection_name, embedding=embeddings
    )
    vectorstore.add_documents(splits)
    return vectorstore

SUMMARY_TRIGGERS = ['what do you know', 'summarize your knowledge', 'qué sabes', 'resumen tu conocimiento']

def summarize_knowledge(vectorstore, groq_api_key: str) -> str:
    from langchain.chains.summarize import load_summarize_chain
    llm = ChatGroq(model="llama-3.3-70b-versatile", api_key=groq_api_key)
    all_docs = vectorstore.similarity_search('',k=100)
    chain = load_summarize_chain(llm, chain_type='map_reduce')
    return chain.invoke(all_docs)['output_text']

def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

def build_chain(vectorstore, groq_api_key: str):
    retriever = vectorstore.as_retriever(
        search_type="similarity", search_kwargs={"k": 5}
    )
    llm = ChatGroq(model="llama-3.3-70b-versatile", api_key=groq_api_key)

    # Step 1: contextualize the question
    contextualize_prompt = ChatPromptTemplate.from_messages([
        ("system", "Given the chat history and the latest user question, reformulate it as a standalone question. Return it as-is if already standalone."),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}"),
    ])
    history_aware_retriever = create_history_aware_retriever(llm, retriever, contextualize_prompt)

    # Step 2: answer using retrieved context
    qa_prompt = ChatPromptTemplate.from_messages([
        ("system", """You are a helpful assistant specialized for helping students understand.
The context may contain mathematical formulas with imperfect formatting — interpret them carefully.
Answer using ONLY the context below. If the answer is not in the context, say "I don't know" or "No lo sé" depending on the language used in the conversation or recent prompt.
Answer in the SAME LANGUAGE the question was asked in.

{context}"""),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}"),
    ])
    question_answer_chain = create_stuff_documents_chain(llm, qa_prompt)

    # Step 3: combine
    rag_chain = create_retrieval_chain(history_aware_retriever, question_answer_chain)

    return rag_chain 

# ── Sidebar ─────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.header("⚙️ Setup")

    groq_api_key = st.secrets["GROQ_API_KEY"] 

    model_option = st.selectbox(
            "Embedding model",
            options=["small", "base", "large"],
            index=0,
            help="Small: faster, handles more docs, Large: slower, better quality."
    )

    MODEL_CONFIG = {
    "small": {"name": "intfloat/multilingual-e5-small", "size": 384, "label": "⚡ Fast — best for large documents"},
    "base":  {"name": "intfloat/multilingual-e5-base",  "size": 768, "label": "🟡 Balanced"},
    "large": {"name": "intfloat/multilingual-e5-large", "size": 1024, "label": "🎯 Best quality — small documents only"},
    }

    st.caption(MODEL_CONFIG[model_option]["label"])


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

    if model_option == "large":
        st.warning("⚠️ Large model may crash on Streamlit free tier with big documents.")
    
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
                vectorstore = build_vectorstore(splits, model_option)
                st.session_state.vectorstore = vectorstore
                st.write("✅ Embeddings done")

                st.write("🔗 Building RAG chain...")
                chain = build_chain(vectorstore, groq_api_key)
                st.write("✅ Chain ready")

                st.session_state.chain = chain
                st.session_state.chat_history = []
                st.session_state.ready = True
                status.update(label="✅ Ready to chat!", state="complete")

        except MemoryError:
            st.error("❌ Out of memory — try uploading fewer or smaller PDFs.")
        except Exception as e:
            st.error(f"❌ Error: {e}")
            st.code(traceback.format_exc())

    if st.session_state.ready:
        st.divider()
        if st.button("🗑️ Clear chat", use_container_width=True):
            st.session_state.chat_history = []
            st.session_state.lc_chat_history = {}
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
                if any(t in question.lower() for t in SUMMARY_TRIGGERS):
                    answer = summarize_knowledge(st.session_state.vectorstore, groq_api_key)
                else:
                    result = st.session_state.chain.invoke({
                        "input": question,
                        "chat_history": st.session_state.lc_chat_history,
                        })
                    answer = result["answer"]
                    st.session_state.lc_chat_history.append(HumanMessage(content=question))
                    st.session_state.lc_chat_history.append(AIMessage(content=answer))
            st.markdown(answer)

        st.session_state.chat_history.append({"role": "assistant", "content": answer})
