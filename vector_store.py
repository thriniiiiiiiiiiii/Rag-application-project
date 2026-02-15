"""
Vector Store: ChromaDB wrapper (Chroma 1.4+ compatible)
Handles storage, retrieval, and persistence of embeddings
"""

import os
from typing import List, Dict
import chromadb
from chromadb import Client
from chromadb.config import Settings


class VectorStore:
    """
    Production-ready ChromaDB wrapper for RAG
    (Compatible with ChromaDB 1.4+ new architecture)
    """

    def __init__(self, persist_directory: str = "./chroma_db"):
        """
        Initialize ChromaDB client (NEW API)

        Args:
            persist_directory: Directory to persist vector DB
        """
        self.persist_directory = persist_directory
        os.makedirs(persist_directory, exist_ok=True)

        # ✅ New Chroma client (non-legacy, non-deprecated)
        self.client = Client(
            Settings(
                persist_directory=persist_directory,
                anonymized_telemetry=False
            )
        )

        self.collection = None

        print(f"✅ VectorStore initialized at: {persist_directory}")

    def create_collection(self, collection_name: str, embedding_function=None):
        """
        Create or load a collection

        Args:
            collection_name: Name of collection
            embedding_function: Optional embedding function
        """
        try:
            self.collection = self.client.get_collection(
                name=collection_name,
                embedding_function=embedding_function
            )
            print(f"📂 Loaded existing collection: {collection_name}")
            print(f"📊 Documents: {self.collection.count()}")

        except Exception:
            self.collection = self.client.create_collection(
                name=collection_name,
                embedding_function=embedding_function,
                metadata={"description": "RAG document collection"}
            )
            print(f"📂 Created new collection: {collection_name}")

    def add_documents(
        self,
        chunks: List[Dict],
        embeddings: List[List[float]] = None
    ):
        """
        Add document chunks to vector DB

        Args:
            chunks: List of {"content": str, "metadata": dict}
            embeddings: Optional precomputed embeddings
        """

        if not chunks:
            raise ValueError("No chunks provided to add_documents")

        documents = [c["content"] for c in chunks]

        # ✅ Metadata must be non-empty dict
        metadatas = []
        for c in chunks:
            meta = c.get("metadata", {})
            if not isinstance(meta, dict) or len(meta) == 0:
                meta = {"source": "unknown"}
            metadatas.append(meta)

        # ✅ Unique stable IDs
        ids = [
            f"chunk_{i}_{abs(hash(c['content']))}"
            for i, c in enumerate(chunks)
        ]

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

        print(f"✅ Added {len(documents)} chunks")
        print(f"📊 Total in DB: {self.collection.count()}")

    def search(self, query: str, n_results: int = 3) -> Dict:
        """
        Semantic search

        Args:
            query: User query
            n_results: Number of results

        Returns:
            Chroma query result dict
        """
        if not self.collection:
            raise RuntimeError("Collection not initialized. Call create_collection() first.")

        return self.collection.query(
            query_texts=[query],
            n_results=n_results
        )

    def delete_collection(self, collection_name: str):
        """Delete collection"""
        self.client.delete_collection(collection_name)
        print(f"🗑️ Deleted collection: {collection_name}")

    def get_all_documents(self) -> Dict:
        """Return all documents"""
        return self.collection.get()


    def debug_retrieval(self, query: str, n_results: int = 5):
        """
        Print detailed retrieval information for debugging
        """
        results = self.search(query, n_results=n_results)

        print("\n================ RETRIEVAL DEBUG ================\n")
        print(f"Query: {query}\n")

        for i, doc in enumerate(results['documents'][0]):
            distance = results['distances'][0][i]
            relevance = 1 - distance
            print(f"Rank {i+1} | Relevance: {relevance:.3f} | Distance: {distance:.3f}")
            print(doc[:250])
            print("-" * 60)

        top_score = 1 - results['distances'][0][0]
        if top_score < 0.5:
            print("\n⚠️ WARNING: Poor retrieval quality detected!")

    print("\n✅ Vector store test complete!")

if __name__ == "__main__":
    print("=" * 70)
    print("🧪 TESTING VECTOR STORE")
    print("=" * 70)

    store = VectorStore(persist_directory="./test_vector_db")
    store.create_collection("test_collection")

    test_chunks = [
        {
            "content": "RAG combines retrieval and generation for better AI responses.",
            "metadata": {"source": "rag.txt", "chunk_id": 0}
        },
        {
            "content": "Vector databases enable semantic search using embeddings.",
            "metadata": {"source": "rag.txt", "chunk_id": 1}
        },
        {
            "content": "Python is widely used in machine learning.",
            "metadata": {"source": "python.txt", "chunk_id": 0}
        }
    ]

    store.add_documents(test_chunks)

    print("\n🔍 Testing search...")
    query = "What is RAG?"
    store.debug_retrieval(query)

    store.delete_collection("test_collection")
    print("\n✅ Vector store test complete!")
