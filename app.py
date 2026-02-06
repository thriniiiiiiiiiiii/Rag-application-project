"""
RAG Application - Streamlit Interface
Upload documents and ask questions using AI-powered retrieval
"""

import streamlit as st
import os
from pathlib import Path
import sys

# Import your RAG components
from rag_engine import RAGEngine
from document_processor import DocumentProcessor

# ============================================================================
# PAGE CONFIGURATION (Must be the FIRST Streamlit command)
# ============================================================================
st.set_page_config(
    page_title="RAG Document Q&A",
    page_icon="📚",
    layout="wide",  # Use full width
    initial_sidebar_state="expanded"  # Sidebar open by default
)

# ============================================================================
# CUSTOM CSS FOR PROFESSIONAL STYLING
# ============================================================================
st.markdown("""
<style>
    /* Main header styling */
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f77b4;
        margin-bottom: 0.5rem;
        text-align: center;
    }
    
    /* Subheader styling */
    .sub-header {
        font-size: 1.2rem;
        color: #666;
        margin-bottom: 2rem;
        text-align: center;
    }
    
    /* Answer box styling */
    .answer-box {
        background: linear-gradient(135deg, #e8f4f8 0%, #f0f8ff 100%);
        padding: 1.5rem;
        border-radius: 0.75rem;
        border-left: 5px solid #1f77b4;
        margin: 1rem 0;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    
    /* Source box styling */
    .source-box {
        background-color: #f8f9fa;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 0.5rem 0;
        border-left: 3px solid #28a745;
    }
    
    /* Metric cards */
    .metric-card {
        background-color: #ffffff;
        padding: 1rem;
        border-radius: 0.5rem;
        box-shadow: 0 1px 3px rgba(0,0,0,0.1);
        margin: 0.5rem 0;
    }
    
    /* Success message */
    .success-message {
        background-color: #d4edda;
        color: #155724;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #28a745;
    }
    
    /* Error message */
    .error-message {
        background-color: #f8d7da;
        color: #721c24;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #dc3545;
    }
    
    /* Hide Streamlit branding */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    
    /* Sidebar styling */
    section[data-testid="stSidebar"] {
        background-color: #f8f9fa;
    }
</style>
""", unsafe_allow_html=True)

# ============================================================================
# SESSION STATE INITIALIZATION
# Session state persists data across Streamlit reruns
# ============================================================================

# Initialize RAG engine (None until first document is indexed)
if 'rag_engine' not in st.session_state:
    st.session_state.rag_engine = None

# Chat history storage
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []

# List of indexed document names
if 'documents_indexed' not in st.session_state:
    st.session_state.documents_indexed = []

# Total cost tracker
if 'total_cost' not in st.session_state:
    st.session_state.total_cost = 0.0

# Total tokens used
if 'total_tokens' not in st.session_state:
    st.session_state.total_tokens = 0

# ============================================================================
# CREATE DATA DIRECTORY FOR UPLOADS
# ============================================================================
DATA_DIR = Path("./data")
DATA_DIR.mkdir(exist_ok=True)

# ============================================================================
# HEADER SECTION
# ============================================================================
st.markdown('<p class="main-header">📚 RAG Document Q&A System</p>', unsafe_allow_html=True)
st.markdown('<p class="sub-header">Ask questions about your documents using AI-powered retrieval</p>', unsafe_allow_html=True)

# ============================================================================
# SIDEBAR: Configuration and Document Upload
# ============================================================================
with st.sidebar:
    st.header("⚙️ Configuration")
    
    # ========================================================================
    # MODEL SELECTION
    # ========================================================================
    st.subheader("🤖 LLM Settings")
    
    model_choice = st.selectbox(
        "Select LLM Model",
        ["gpt-3.5-turbo", "gpt-4", "gpt-4-turbo-preview"],
        index=0,  # Default to gpt-3.5-turbo
        help="💡 GPT-3.5: Fast & cheap | GPT-4: More accurate but slower & expensive"
    )
    
    # Show model info
    model_info = {
        "gpt-3.5-turbo": "⚡ Fast | 💰 $0.002/1K tokens",
        "gpt-4": "🎯 Accurate | 💰 $0.03/1K tokens",
        "gpt-4-turbo-preview": "🚀 Best | 💰 $0.01/1K tokens"
    }
    st.caption(model_info.get(model_choice, ""))
    
    st.divider()
    
    # ========================================================================
    # RETRIEVAL SETTINGS
    # ========================================================================
    st.subheader("🔍 Retrieval Settings")
    
    n_results = st.slider(
        "Number of chunks to retrieve",
        min_value=1,
        max_value=10,
        value=3,
        help="More chunks = more context but slower and more expensive"
    )
    
    chunk_size = st.slider(
        "Chunk size (characters)",
        min_value=200,
        max_value=1500,
        value=500,
        step=100,
        help="Larger chunks = more context per piece"
    )
    
    chunk_overlap = st.slider(
        "Chunk overlap (characters)",
        min_value=0,
        max_value=200,
        value=50,
        step=25,
        help="Overlap prevents cutting sentences"
    )
    
    st.divider()
    
    # ========================================================================
    # DOCUMENT UPLOAD
    # ========================================================================
    st.header("📄 Upload Documents")
    
    uploaded_file = st.file_uploader(
        "Choose a PDF file",
        type=['pdf'],
        help="Upload a PDF document to index and ask questions about"
    )
    
    # Show upload instructions
    if uploaded_file is None:
        st.info("👆 Upload a PDF to get started")
    else:
        # Display file info
        file_size_mb = len(uploaded_file.getvalue()) / (1024 * 1024)
        st.success(f"✅ File loaded: {uploaded_file.name}")
        st.caption(f"Size: {file_size_mb:.2f} MB")
        
        # ====================================================================
        # INDEX BUTTON
        # ====================================================================
        if st.button("🔄 Index Document", type="primary", use_container_width=True):
            # Save uploaded file to disk
            file_path = DATA_DIR / uploaded_file.name
            
            with open(file_path, "wb") as f:
                f.write(uploaded_file.getbuffer())
            
            # Show indexing progress
            with st.spinner(f"🔄 Indexing {uploaded_file.name}..."):
                try:
                    # Initialize RAG engine if this is the first document
                    if st.session_state.rag_engine is None:
                        st.session_state.rag_engine = RAGEngine(
                            model=model_choice,
                            n_retrieval_results=n_results
                        )
                    
                    # Index the document with custom chunk settings
                    st.session_state.rag_engine.index_document(
                        str(file_path),
                        chunk_size=chunk_size,
                        chunk_overlap=chunk_overlap
                    )
                    
                    # Add to indexed documents list (avoid duplicates)
                    if uploaded_file.name not in st.session_state.documents_indexed:
                        st.session_state.documents_indexed.append(uploaded_file.name)
                    
                    st.success(f"✅ Successfully indexed {uploaded_file.name}!")
                    st.balloons()  # Celebration animation!
                    
                except Exception as e:
                    st.error(f"❌ Error indexing document: {str(e)}")
                    # Show detailed error in expander
                    with st.expander("🔍 View Error Details"):
                        st.code(str(e))
    
    # ========================================================================
    # SHOW INDEXED DOCUMENTS
    # ========================================================================
    if st.session_state.documents_indexed:
        st.divider()
        st.subheader("📚 Indexed Documents")
        
        for i, doc in enumerate(st.session_state.documents_indexed, 1):
            st.text(f"{i}. ✓ {doc}")
    
    # ========================================================================
    # USAGE STATISTICS
    # ========================================================================
    if st.session_state.total_tokens > 0:
        st.divider()
        st.subheader("📊 Usage Statistics")
        
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Total Tokens", f"{st.session_state.total_tokens:,}")
        with col2:
            st.metric("Total Cost", f"${st.session_state.total_cost:.4f}")
    
    # ========================================================================
    # CLEAR ALL DATA BUTTON
    # ========================================================================
    st.divider()
    
    if st.button("🗑️ Clear All Data", type="secondary", use_container_width=True):
        # Confirm before clearing
        st.session_state.rag_engine = None
        st.session_state.chat_history = []
        st.session_state.documents_indexed = []
        st.session_state.total_cost = 0.0
        st.session_state.total_tokens = 0
        st.success("✅ All data cleared!")
        st.rerun()  # Refresh the app

# ============================================================================
# MAIN AREA: Chat Interface
# ============================================================================

st.header("💬 Ask Questions")

# ============================================================================
# DISPLAY CHAT HISTORY
# ============================================================================
for i, chat in enumerate(st.session_state.chat_history):
    # User question
    with st.chat_message("user", avatar="👤"):
        st.write(chat['question'])
    
    # Assistant answer
    with st.chat_message("assistant", avatar="🤖"):
        # Display answer in styled box
        st.markdown(
            f'<div class="answer-box">{chat["answer"]}</div>',
            unsafe_allow_html=True
        )
        
        # ====================================================================
        # SOURCES EXPANDER
        # ====================================================================
        with st.expander("📖 View Sources & Context"):
            for j, source in enumerate(chat['sources'], 1):
                relevance_pct = source['relevance_score'] * 100
                
                # Color code by relevance
                if relevance_pct >= 70:
                    relevance_color = "🟢"
                elif relevance_pct >= 50:
                    relevance_color = "🟡"
                else:
                    relevance_color = "🔴"
                
                st.markdown(f"""
                <div class="source-box">
                    <strong>{relevance_color} Source {j}</strong> 
                    (Relevance: {relevance_pct:.1f}%)<br>
                    <em>📄 {source['metadata']['source']} - Chunk {source['metadata']['chunk_id']}</em><br>
                    <hr style="margin: 0.5rem 0;">
                    <p style="margin-top: 0.5rem;">{source['content'][:400]}...</p>
                </div>
                """, unsafe_allow_html=True)
        
        # ====================================================================
        # METADATA EXPANDER
        # ====================================================================
        with st.expander("ℹ️ Response Details"):
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric("Model", chat['metadata']['model'])
            with col2:
                st.metric("Tokens", f"{chat['metadata']['tokens_used']:,}")
            with col3:
                st.metric("Cost", f"${chat['metadata']['cost']:.6f}")

# ============================================================================
# QUESTION INPUT
# ============================================================================
if st.session_state.rag_engine is not None:
    # Chat input at the bottom
    question = st.chat_input(
        "💭 Ask a question about your documents...",
        key="user_input"
    )
    
    if question:
        # ====================================================================
        # DISPLAY USER QUESTION IMMEDIATELY
        # ====================================================================
        with st.chat_message("user", avatar="👤"):
            st.write(question)
        
        # ====================================================================
        # GENERATE ANSWER
        # ====================================================================
        with st.chat_message("assistant", avatar="🤖"):
            with st.spinner("🤔 Thinking..."):
                try:
                    # Query the RAG engine
                    result = st.session_state.rag_engine.query(
                        question,
                        verbose=False  # Don't print to console
                    )
                    
                    # Display answer
                    st.markdown(
                        f'<div class="answer-box">{result["answer"]}</div>',
                        unsafe_allow_html=True
                    )
                    
                    # Update usage statistics
                    st.session_state.total_tokens += result['metadata']['tokens_used']
                    st.session_state.total_cost += result['metadata']['cost']
                    
                    # ========================================================
                    # SOURCES EXPANDER
                    # ========================================================
                    with st.expander("📖 View Sources & Context"):
                        for j, source in enumerate(result['sources'], 1):
                            relevance_pct = source['relevance_score'] * 100
                            
                            if relevance_pct >= 70:
                                relevance_color = "🟢"
                            elif relevance_pct >= 50:
                                relevance_color = "🟡"
                            else:
                                relevance_color = "🔴"
                            
                            st.markdown(f"""
                            <div class="source-box">
                                <strong>{relevance_color} Source {j}</strong> 
                                (Relevance: {relevance_pct:.1f}%)<br>
                                <em>📄 {source['metadata']['source']} - Chunk {source['metadata']['chunk_id']}</em><br>
                                <hr style="margin: 0.5rem 0;">
                                <p style="margin-top: 0.5rem;">{source['content'][:400]}...</p>
                            </div>
                            """, unsafe_allow_html=True)
                    
                    # ========================================================
                    # METADATA EXPANDER
                    # ========================================================
                    with st.expander("ℹ️ Response Details"):
                        col1, col2, col3 = st.columns(3)
                        
                        with col1:
                            st.metric("Model", result['metadata']['model'])
                        with col2:
                            st.metric("Tokens", f"{result['metadata']['tokens_used']:,}")
                        with col3:
                            st.metric("Cost", f"${result['metadata']['cost']:.6f}")
                    
                    # ========================================================
                    # ADD TO CHAT HISTORY
                    # ========================================================
                    st.session_state.chat_history.append({
                        'question': question,
                        'answer': result['answer'],
                        'sources': result['sources'],
                        'metadata': result['metadata']
                    })
                    
                except Exception as e:
                    st.error(f"❌ Error generating answer: {str(e)}")
                    
                    # Show detailed error
                    with st.expander("🔍 View Error Details"):
                        st.code(str(e))
                        import traceback
                        st.code(traceback.format_exc())

else:
    # ========================================================================
    # NO DOCUMENTS INDEXED YET
    # ========================================================================
    st.info("""
    ### 👋 Welcome to RAG Document Q&A!
    
    **To get started:**
    1. 👈 Upload a PDF document in the sidebar
    2. ⚙️ Adjust settings if needed (optional)
    3. 🔄 Click "Index Document"
    4. 💬 Start asking questions!
    
    **Example questions you can ask:**
    - "What is this document about?"
    - "Summarize the main points"
    - "What does section 3 discuss?"
    """)

# ============================================================================
# FOOTER
# ============================================================================
st.divider()

col1, col2, col3 = st.columns(3)

with col1:
    st.caption("🚀 Built with Streamlit")
with col2:
    st.caption("🤖 Powered by OpenAI")
with col3:
    st.caption("🔍 Vector Search by ChromaDB")