"""
Embedding Engine: Convert text to embeddings
This is where semantic search magic happens!
"""

from sentence_transformers import SentenceTransformer
from typing import List
import numpy as np


class EmbeddingEngine:
    """
    Creates embeddings using Sentence Transformers

    Model: all-MiniLM-L6-v2
    - Size: ~420MB (downloads on first run)
    - Speed: Fast (~1000 sentences/sec on CPU)
    - Dimensions: 384
    - Quality: Excellent for RAG pipelines
    - FREE and runs locally
    """

    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        print(f"📥 Loading embedding model: {model_name}")
        print(f"   (First run downloads ~420MB)")
        
        self.model = SentenceTransformer(model_name)
        
        print(f"✅ Model loaded successfully!")
        print(f"📐 Embedding dimension: {self.model.get_sentence_embedding_dimension()}")

    # ---------------------------
    # Single text embedding
    # ---------------------------
    def embed_text(self, text: str) -> List[float]:
        """
        Convert a single text to an embedding vector

        Args:
            text: Input text

        Returns:
            Embedding vector (list of floats)
        """
        embedding = self.model.encode(text)
        return embedding.tolist()

    # ---------------------------
    # Batch embedding
    # ---------------------------
    def embed_batch(self, texts: List[str], show_progress: bool = True) -> List[List[float]]:
        """
        Convert multiple texts to embeddings (FAST)

        Args:
            texts: List of input texts
            show_progress: Show progress bar

        Returns:
            List of embedding vectors
        """
        print(f"🔢 Embedding {len(texts)} texts...")

        embeddings = self.model.encode(
            texts,
            show_progress_bar=show_progress,
            batch_size=32
        )

        print(f"✅ Created {len(embeddings)} embeddings")
        return embeddings.tolist()

    # ---------------------------
    # Similarity computation
    # ---------------------------
    def compute_similarity(self, embedding1: List[float], embedding2: List[float]) -> float:
        """
        Compute cosine similarity between two embeddings

        Returns:
            Similarity score between -1 and 1
            - 1.0 = identical meaning
            - 0.7+ = highly similar
            - 0.4–0.6 = moderately similar
            - <0.3 = weak similarity
        """
        vec1 = np.array(embedding1)
        vec2 = np.array(embedding2)

        similarity = np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))
        return float(similarity)


# ============================================================
# 🧪 TEST RUNNER
# ============================================================
if __name__ == "__main__":
    print("=" * 70)
    print("🧪 TESTING EMBEDDING ENGINE")
    print("=" * 70)

    engine = EmbeddingEngine()

    texts = [
        "Machine learning is a subset of artificial intelligence",
        "Deep learning uses neural networks",
        "Python is a programming language",
        "The weather is sunny today"
    ]

    # Create embeddings
    embeddings = engine.embed_batch(texts)

    print("\n🔍 Similarity Tests")
    print("=" * 70)

    query_embedding = embeddings[0]

    for i, text in enumerate(texts):
        similarity = engine.compute_similarity(query_embedding, embeddings[i])
        print(f"\nText 1 vs Text {i+1}:")
        print(f"  '{texts[0][:60]}...'")
        print(f"  '{text[:60]}...'")
        print(f"  Similarity Score: {similarity:.4f}")

    print("\n✅ Embedding engine test complete!")
