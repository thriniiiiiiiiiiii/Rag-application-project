"""
Vector Store: Manages ChromaDB operations
Production-grade vector database layer for RAG systems
"""

import os
from typing import List, Dict, Optional
import chromadb
from chromadb.config import Settings


class VectorStore:
    """
    Wrapper around ChromaDB for RAG operations

    Handles:
    - Vector storage
    - Persistence
    - Semantic search
    - Metadata filtering
    - Collection management
    """

    def __init__(self, persist_directory: str = "./chroma_db"):
        """
        Initialize ChromaDB client (modern API)

        Args:
            persist_directory: Directory to persist vector database
        """
        self.persist_directory = persist_directory
        os.makedirs(persist_directory, exist_ok=True)

        # Modern Chroma client initialization (migration-safe)
        settings = Settings(
            persist_directory=persist_directory,
            anonymized_telemetry=False
        )

        self.client = chromadb.Client(settings)
        self.collection = None

        print(f"✅ Vector store initialized at: {persist_directory}")

    # ------------------------------------------------------------------
    # Collection Management
    # ------------------------------------------------------------------

    def create_collection(
        self,
        collection_name: str,
        embedding_function: Optional[callable] = None
    ):
        """
        Create or load a collection

        Args:
            collection_name: Collection name
            embedding_function: Optional embedding function
        """
        try:
            self.collection = self.client.get_collection(
                name=collection_name,
                embedding_function=embedding_function
            )
            print(f"📂 Loaded existing collection: {collection_name}")
            print(f"   Documents: {self.collection.count()}")

        except Exception:
            self.collection = self.client.create_collection(
                name=collection_name,
                embedding_function=embedding_function,
                metadata={
                    "description": "RAG document collection",
                    "engine": "chromadb"
                }
            )
            print(f"📂 Created new collection: {collection_name}")

    # ------------------------------------------------------------------
    # Data Ingestion
    # ------------------------------------------------------------------

    def add_documents(
        self,
        chunks: List[Dict],
        embeddings: Optional[List[List[float]]] = None
    ):
        """
        Add document chunks to vector store

        Args:
            chunks: List of chunk dicts
            embeddings: Optional precomputed embeddings
        """
        if not self.collection:
            raise RuntimeError("❌ Collection not initialized. Call create_collection() first.")

        print(f"\n💾 Adding {len(chunks)} chunks to vector store...")

        documents = [c["content"] for c in chunks]
        metadatas = [c["metadata"] for c in chunks]
        ids = [f"chunk_{i}" for i in range(len(chunks))]

        if embeddings:
            self.collection.add(
                documents=documents,
                metadatas=metadatas,
                ids=ids,
                embeddings=embeddings
            )
        else:
            self.collection.add(
                documents=documents,
                metadatas=metadatas,
                ids=ids
            )

        print(f"✅ Added {len(chunks)} documents")
        print(f"📊 Total documents: {self.collection.count()}")

    # ------------------------------------------------------------------
    # Retrieval
    # ------------------------------------------------------------------

    def search(
        self,
        query: str,
        n_results: int = 3,
        where: Optional[Dict] = None
    ) -> Dict:
        """
        Semantic search

        Args:
            query: Query string
            n_results: Top-k results
            where: Optional metadata filters

        Returns:
            Chroma query result dict
        """
        if not self.collection:
            raise RuntimeError("❌ Collection not initialized. Call create_collection() first.")

        results = self.collection.query(
            query_texts=[query],
            n_results=n_results,
            where=where
        )

        return results

    # ------------------------------------------------------------------
    # Utilities
    # ------------------------------------------------------------------

    def delete_collection(self, collection_name: str):
        """Delete a collection"""
        self.client.delete_collection(name=collection_name)
        print(f"🗑️ Deleted collection: {collection_name}")

    def get_all_documents(self) -> Dict:
        """Return all stored documents"""
        if not self.collection:
            raise RuntimeError("❌ Collection not initialized.")
        return self.collection.get()


# ======================================================================
# TEST MODULE
# ======================================================================

if __name__ == "__main__":
    print("=" * 70)
    print("🧪 TESTING VECTOR STORE")
    print("=" * 70)

    # Initialize vector store
    store = VectorStore(persist_directory="./test_vector_db")
    store.create_collection("test_collection")

    # Sample chunks
    test_chunks = [
        {
            "content": "RAG combines retrieval and generation for better AI responses.",
            "metadata": {"source": "rag_intro.txt", "chunk_id": 0, "char_count": 65}
        },
        {
            "content": "Vector databases enable semantic search using embeddings.",
            "metadata": {"source": "rag_intro.txt", "chunk_id": 1, "char_count": 59}
        },
        {
            "content": "Python is widely used for machine learning applications.",
            "metadata": {"source": "python.txt", "chunk_id": 0, "char_count": 57}
        }
    ]

    # Add documents
    store.add_documents(test_chunks)

    # Search
    print("\n🔍 Testing search...")
    query = "How does RAG work?"
    results = store.search(query, n_results=2)

    print(f"\nQuery: '{query}'")
    print("=" * 70)

    for i, doc in enumerate(results["documents"][0]):
        print(f"\nResult {i+1}:")
        print(f"  Text: {doc}")
        print(f"  Source: {results['metadatas'][0][i]['source']}")
        print(f"  Distance: {results['distances'][0][i]:.4f}")

    # Cleanup
    store.delete_collection("test_collection")

    print("\n✅ Vector store test complete!")
