import streamlit as st
from pathlib import Path
import json
import os

# =========================
# IMPORT BACKEND COMPONENTS
# =========================
from rag_engine import RAGEngineOllama
from document_processor import DocumentProcessor

# =========================
# PAGE CONFIG
# =========================
st.set_page_config(
    page_title="VANTAGE - Intelligence OS",
    page_icon="📡",
    layout="wide",
    initial_sidebar_state="expanded"
)

# =========================
# PREMIUM STYLING (VANTAGE)
# =========================
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;600&family=JetBrains+Mono&display=swap');
    
    :root {
        --vantage-blue: #00f2ff;
        --vantage-glass: rgba(15, 23, 42, 0.8);
        --vantage-border: rgba(0, 242, 255, 0.2);
    }

    .stApp {
        background: radial-gradient(circle at top right, #0a192f, #020617);
        color: #e2e8f0;
        font-family: 'Outfit', sans-serif;
    }

    .main-header {
        font-size: 3rem;
        font-weight: 600;
        letter-spacing: -1px;
        background: linear-gradient(90deg, #fff, #00f2ff);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        text-align: center;
        margin-bottom: 0.5rem;
    }

    .sub-header {
        text-align: center;
        color: #94a3b8;
        font-size: 1.1rem;
        margin-bottom: 2rem;
        font-family: 'JetBrains Mono', monospace;
    }

    .stButton>button {
        background: linear-gradient(135deg, rgba(0, 242, 255, 0.1), rgba(0, 242, 255, 0.05));
        border: 1px solid var(--vantage-border);
        color: var(--vantage-blue);
        border-radius: 8px;
        transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
    }

    .stButton>button:hover {
        background: rgba(0, 242, 255, 0.15);
        border-color: var(--vantage-blue);
        box-shadow: 0 0 20px rgba(0, 242, 255, 0.2);
    }

    .chat-card {
        background: var(--vantage-glass);
        border: 1px solid var(--vantage-border);
        border-radius: 12px;
        padding: 1.5rem;
        margin-bottom: 1rem;
        backdrop-filter: blur(12px);
    }

    .source-pill {
        display: inline-block;
        padding: 2px 8px;
        background: rgba(0, 242, 255, 0.1);
        border: 1px solid var(--vantage-border);
        border-radius: 4px;
        font-size: 0.75rem;
        color: var(--vantage-blue);
        margin-right: 5px;
    }
</style>
""", unsafe_allow_html=True)

# =========================
# SESSION & PERSISTENCE
# =========================
HISTORY_FILE = "chat_history.json"

def save_history(history):
    with open(HISTORY_FILE, "w") as f:
        json.dump(history, f)

def load_history():
    if os.path.exists(HISTORY_FILE):
        try:
            with open(HISTORY_FILE, "r") as f:
                return json.load(f)
        except:
            return []
    return []

if "chat_history" not in st.session_state:
    st.session_state.chat_history = load_history()

if "rag_engine" not in st.session_state:
    st.session_state.rag_engine = None

if "documents_indexed" not in st.session_state:
    st.session_state.documents_indexed = []

DATA_DIR = Path("./data")
DATA_DIR.mkdir(exist_ok=True)

# =========================
# HEADER
# =========================
st.markdown('<div class="main-header">VANTAGE</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-header">PROXIMITY RADAR | RAG ENGINE | v2.0</div>', unsafe_allow_html=True)

# =========================
# SIDEBAR
# =========================
with st.sidebar:
    st.image("https://img.icons8.com/nolan/128/satellite.png", width=100)
    st.header("⚡ COMMAND CENTER")

    model_name = st.selectbox("Intelligence Model", ["llama3:latest", "mistral", "phi3"])
    use_rerank = st.toggle("Enable Reranking (FlashRank)", value=True)
    use_hyde = st.toggle("Enable HyDE (Neural Expansion)", value=True)
    chunk_method = st.radio("Chunking Strategy", ["fixed", "semantic"], horizontal=True)
    
    st.divider()
    
    st.subheader("📄 INGESTION")
    uploaded_file = st.file_uploader("Drop Intel Files", type=["pdf", "docx", "txt", "md"])

    if uploaded_file:
        if st.button("🔄 INDEX DOSSIER", use_container_width=True):
            file_path = DATA_DIR / uploaded_file.name
            with open(file_path, "wb") as f:
                f.write(uploaded_file.getbuffer())

            with st.spinner("Processing Signal..."):
                try:
                    if st.session_state.rag_engine is None:
                        st.session_state.rag_engine = RAGEngineOllama(
                            model=model_name,
                            use_reranker=use_rerank,
                            use_hyde=use_hyde
                        )
                    else:
                        st.session_state.rag_engine.use_hyde = use_hyde
                    
                    st.session_state.rag_engine.index_document(str(file_path), chunk_method=chunk_method)
                    
                    if uploaded_file.name not in st.session_state.documents_indexed:
                        st.session_state.documents_indexed.append(uploaded_file.name)
                    
                    st.success("✅ INTEL SECURED")
                    st.balloons()
                except Exception as e:
                    st.error(f"❌ SIGNAL LOSS: {e}")

    if st.session_state.documents_indexed:
        st.divider()
        st.subheader("📚 ARCHIVE")
        for doc in st.session_state.documents_indexed:
            st.caption(f"✓ {doc}")

    if st.button("🗑️ PURGE ALL", use_container_width=True):
        st.session_state.rag_engine = None
        st.session_state.chat_history = []
        st.session_state.documents_indexed = []
        if os.path.exists(HISTORY_FILE):
                os.remove(HISTORY_FILE)
        st.rerun()

# =========================
# CHAT INTERFACE
# =========================
for chat in st.session_state.chat_history:
    with st.chat_message("user"):
        st.markdown(chat["question"])
    with st.chat_message("assistant"):
        st.markdown(f'<div class="chat-card">{chat["answer"]}</div>', unsafe_allow_html=True)
        if chat.get("sources"):
             with st.expander("🔍 ANALYSIS SOURCES"):
                 for src in chat["sources"]:
                     st.markdown(f"""
                     <div style="background: rgba(255,255,255,0.05); padding: 10px; border-radius: 5px; margin-bottom: 5px;">
                        <span class="source-pill">{src['metadata'].get('source', 'Unknown')}</span>
                        <p style="font-size: 0.9rem; margin-top: 5px;">{src['content'][:250]}...</p>
                     </div>
                     """, unsafe_allow_html=True)

if question := st.chat_input("Input intelligence query..."):
    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"):
        if st.session_state.rag_engine:
            with st.spinner("Decoding..."):
                result = st.session_state.rag_engine.query(question)
                st.markdown(f'<div class="chat-card">{result["answer"]}</div>', unsafe_allow_html=True)
                
                new_chat = {
                    "question": question,
                    "answer": result["answer"],
                    "sources": result["sources"]
                }
                st.session_state.chat_history.append(new_chat)
                save_history(st.session_state.chat_history)
                st.rerun() # Refresh to show in history layout
        else:
            st.warning("⚠️ PROXIMITY SENSOR OFFLINE. Please index a document.")
