#  VANTAGE: Advanced Local Intelligence OS

VANTAGE is a high-performance, local-first RAG (Retrieval-Augmented Generation) platform designed for secure, private intelligence analysis. It leverages **Ollama** for local inference and **ChromaDB** for vector storage, enhanced with elite retrieval techniques like HyDE and GraphRAG.

![VANTAGE UI](https://img.icons8.com/nolan/256/satellite.png)
## Key Features

###  Elite RAG Pipeline
- **Neural Query Expansion (HyDE)**: Generates hypothetical answers to bridge the semantic gap between user queries and technical documents.
- **FlashRank Reranking**: Utilizes cross-encoders to re-score retrieval results, drastically reducing hallucinations.
- **Semantic Chunking**: Intelligent document splitting based on topic shifts rather than fixed character counts.
- **GraphRAG**: Extracts entities and relationships into a Knowledge Graph for complex relational reasoning.

###  Multi-Format Ingestion
Supports high-fidelity extraction from:
- **PDF** (via PyPDF2)
- **DOCX** (via python-docx)
- **Markdown & Text**

###  Premium Interface
- **Modern Glassmorphic UI**: A dark-themed, data-driven dashboard built with Streamlit.
- **Command Center**: Real-time toggles for HyDE, Reranking, and Chunking strategies.
- **Source Transparency**: Deep-dive into specific document chunks with relevance scoring.
- **Session Persistence**: Local history management for seamless research continuity.

## Tech Stack
- **AI Models**: Llama3, Mistral, Phi-3 (via Ollama)
- **VDB**: ChromaDB
- **Languages**: Python 3.10+
- **Frontend**: Streamlit
- **Embeddings**: Sentence-Transformers (`all-MiniLM-L6-v2`)
- **Reranker**: FlashRank

## Installation

1. **Install Ollama**: Download from [ollama.ai](https://ollama.ai) and run it.
2. **Clone the Repository**:
   ```bash
   git clone https://github.com/YOUR_USERNAME/VANTAGE.git
   cd VANTAGE
   ```
3. **Set Up Virtual Environment**:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```
4. **Install Dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

##  Usage

Launch the VANTAGE Intelligence OS:
```bash
streamlit run app.py
```

1. **Select Model**: Choose your preferred local LLM from the sidebar.
2. **Index Dossier**: Drop your PDF/DOCX files and click "Index".
3. **Analyze**: Use the chat interface to query your data with elite retrieval enabled.

.
