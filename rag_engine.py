"""
RAG Engine using Ollama (100% FREE!)
DEBUG + SAFE + ENGINEERING GRADE RAG
Model: llama3:latest
UPDATED for ChromaDB 1.4+
"""

import os
import requests
from typing import List, Dict

from document_processor import DocumentProcessor
from embedding_engine import EmbeddingEngine
from vector_store import VectorStore

# ==============================
# DEBUG + SAFETY UTILITIES
# ==============================

def validate_answer(answer: str, context_chunks: list):
    context_text = " ".join([c['content'] for c in context_chunks]).lower()
    answer_terms = set(answer.lower().split())
    context_terms = set(context_text.split())

    if not answer_terms:
        return False, "⚠️ Empty answer"

    overlap = len(answer_terms & context_terms) / len(answer_terms)

    if overlap < 0.30:
        return False, "⚠️ WARNING: Answer may not be grounded in context"
    return True, "✅ Answer grounded in context"


def estimate_tokens(text: str):
    return int(len(text.split()) * 1.3)


def check_context_size(chunks, limit=1800):
    total = 0
    safe_chunks = []

    for chunk in chunks:
        tokens = estimate_tokens(chunk['content'])
        if total + tokens <= limit:
            safe_chunks.append(chunk)
            total += tokens
        else:
            break

    return safe_chunks


def build_context(chunks):
    context = []
    for i, c in enumerate(chunks):
        context.append(f"[Source {i+1}]\n{c['content']}")
    return "\n\n".join(context)


def rag_guard(answer, chunks):
    if not chunks:
        return "I don't have that information in the provided documents."

    if len(answer.strip()) == 0:
        return "I don't have that information in the provided documents."

    return answer


# ==============================
# RAG ENGINE
# ==============================

class RAGEngineOllama:
    """Production-grade RAG system using FREE Ollama"""

    def __init__(
        self,
        collection_name: str = "rag_documents",
        model: str = "llama3:latest",
        ollama_url: str = "http://localhost:11434",
        n_retrieval_results: int = 3
    ):
        # Core components
        self.embedding_engine = EmbeddingEngine()
        self.vector_store = VectorStore()
        self.vector_store.create_collection(collection_name)

        # Ollama
        self.model = model
        self.ollama_url = ollama_url
        self.n_retrieval_results = n_retrieval_results

        # STRONG SYSTEM PROMPT (ANTI-HALLUCINATION)
        self.system_prompt = """
You are a retrieval-based AI assistant.

STRICT RULES:
- Answer ONLY using the provided context.
- Do NOT use external knowledge.
- Do NOT infer missing information.
- Do NOT guess.
- If the answer is not explicitly present, respond:
  "I don't have that information in the provided documents."

You must stay fully grounded in the context.
"""

        self._test_ollama()
        print("✅ RAG Engine (Ollama + Debug Mode) initialized!")

    def _test_ollama(self):
        try:
            response = requests.get(f"{self.ollama_url}/api/tags", timeout=5)
            if response.status_code != 200:
                raise Exception(f"Ollama returned status {response.status_code}")
            print("🟢 Ollama server connected successfully")
        except Exception as e:
            raise Exception(f"❌ Cannot connect to Ollama: {e}\nMake sure Ollama is running!")

    # ==============================
    # INDEXING
    # ==============================

    def index_document(self, pdf_path: str):
        print(f"\n{'='*60}")
        print(f"📚 INDEXING: {pdf_path}")
        print(f"{'='*60}")

        processor = DocumentProcessor(chunk_size=300, chunk_overlap=80)
        chunks = processor.process_document(pdf_path)

        chunk_texts = [chunk["content"] for chunk in chunks]
        embeddings = self.embedding_engine.embed_batch(chunk_texts)

        self.vector_store.add_documents(chunks=chunks, embeddings=embeddings)

        print(f"\n✅ Document indexed!")
        print(f"📊 Total chunks in DB: {self.vector_store.collection.count()}")

    # ==============================
    # RETRIEVAL
    # ==============================

    def retrieve_context(self, query: str) -> List[Dict]:
        results = self.vector_store.search(query, n_results=self.n_retrieval_results)

        retrieved_chunks = []
        for i in range(len(results['documents'][0])):
            chunk = {
                'content': results['documents'][0][i],
                'metadata': results['metadatas'][0][i],
                'distance': results['distances'][0][i],
                'relevance_score': 1 - results['distances'][0][i]
            }
            retrieved_chunks.append(chunk)

        return retrieved_chunks

    def debug_retrieval(self, query: str):
        results = self.vector_store.search(query, n_results=5)

        print("\n================ RETRIEVAL DEBUG ================\n")
        print(f"Query: {query}\n")

        for i, chunk in enumerate(results['documents'][0]):
            relevance = 1 - results['distances'][0][i]
            print(f"Rank {i+1} | Relevance: {relevance:.3f}")
            print(chunk[:250])
            print("-"*60)

        top_score = 1 - results['distances'][0][0]
        if top_score < 0.5:
            print("\n⚠️ WARNING: Poor retrieval quality detected!")

    # ==============================
    # GENERATION
    # ==============================

    def generate_answer(self, prompt: str) -> Dict:
        data = {
            "model": self.model,
            "prompt": prompt,
            "stream": False,
            "options": {
                "temperature": 0.0,   # 🔥 deterministic
                "num_predict": 500
            }
        }

        response = requests.post(
            f"{self.ollama_url}/api/generate",
            json=data,
            timeout=120
        )

        if response.status_code == 200:
            result = response.json()
            return {
                'answer': result.get('response', '').strip(),
                'model': self.model,
                'cost': 0.0
            }
        else:
            raise Exception(f"❌ Ollama error: {response.status_code} | {response.text}")

    # ==============================
    # FULL PIPELINE
    # ==============================

    def query(self, question: str, verbose: bool = True) -> Dict:
        if verbose:
            print(f"\n{'='*60}")
            print(f"❓ QUESTION: {question}")
            print(f"{'='*60}")

        # Retrieve
        if verbose:
            print("\n🔍 Retrieving context...")

        chunks = self.retrieve_context(question)

        if verbose:
            print(f"✅ Retrieved {len(chunks)} chunks")

        # DEBUG RETRIEVAL
        self.debug_retrieval(question)

        # CONTEXT CONTROL
        chunks = check_context_size(chunks, limit=1800)

        # BUILD CONTEXT
        context_str = build_context(chunks)

        full_prompt = f"""{self.system_prompt}

Context:
{context_str}

Question: {question}

Answer:"""

        # Generate
        if verbose:
            print("\n🤖 Generating answer...")

        result = self.generate_answer(full_prompt)

        # VALIDATION
        valid, msg = validate_answer(result['answer'], chunks)
        print(msg)

        # SAFETY GUARD
        final_answer = rag_guard(result['answer'], chunks)

        if verbose:
            print(f"\n💡 ANSWER:\n{final_answer}")
            print(f"\n💰 Cost: $0.00 (FREE!)")

        return {
            'answer': final_answer,
            'sources': chunks,
            'metadata': result
        }


# ============================
# TEST PIPELINE
# ============================
if __name__ == "__main__":
    print("🚀 Initializing DEBUG RAG with Ollama...")
    rag = RAGEngineOllama(collection_name="test_rag_debug")

    sample_text = """
    Company Refund Policy

    Full refunds within 30 days of purchase.
    Products must be unopened and in original packaging.

    To request refund:
    1. Email support@company.com
    2. Include order number
    3. State reason for refund

    Refunds processed in 5-7 business days.
    Shipping costs non-refundable unless item damaged.
    """

    processor = DocumentProcessor(chunk_size=200, chunk_overlap=40)
    chunks = processor.create_chunks(sample_text, "policy.txt")

    chunk_texts = [c["content"] for c in chunks]
    embeddings = rag.embedding_engine.embed_batch(chunk_texts)
    rag.vector_store.add_documents(chunks=chunks, embeddings=embeddings)

    questions = [
        "How long do I have to return a product?",
        "Explain refund process",
        "Is shipping refundable?",
        "What is the CEO name?"  # hallucination test
    ]

    for q in questions:
        rag.query(q)
        print("\n" + "="*60 + "\n")

    rag.vector_store.delete_collection("test_rag_debug")
