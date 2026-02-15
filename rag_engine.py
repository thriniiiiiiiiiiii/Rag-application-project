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
from graph_engine import GraphEngine

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
        n_retrieval_results: int = 15,
        use_reranker: bool = True,
        use_hyde: bool = True
    ):
        # Core components
        self.embedding_engine = EmbeddingEngine()
        self.vector_store = VectorStore()
        self.vector_store.create_collection(collection_name)

        # Ollama
        self.model = model
        self.ollama_url = ollama_url
        self.n_retrieval_results = n_retrieval_results
        self.use_hyde = use_hyde
        
        # Graph Engine
        self.graph_engine = GraphEngine(ollama_url=ollama_url)
        
        # Reranker
        self.use_reranker = use_reranker
        if self.use_reranker:
            try:
                from flashrank import Ranker
                self.ranker = Ranker(model_name="ms-marco-MiniLM-L-12-v2", cache_dir="/tmp/flashrank")
                print("✅ FlashRank Reranker initialized!")
            except Exception as e:
                print(f"⚠️ Could not init FlashRank: {e}. Falling back to standard retrieval.")
                self.use_reranker = False

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

        self._ensure_model_exists()
        print("✅ RAG Engine (VANTAGE Optimized) initialized!")

    def _ensure_model_exists(self):
        """Check if model exists in Ollama, pull if not"""
        try:
            response = requests.get(f"{self.ollama_url}/api/tags", timeout=5)
            tags = response.json().get('models', [])
            model_names = [m['name'] for m in tags]
            
            if self.model not in model_names and f"{self.model}:latest" not in model_names:
                print(f"📥 Pulling model {self.model} (this may take a few minutes)...")
                requests.post(f"{self.ollama_url}/api/pull", json={"name": self.model})
            else:
                print(f"🟢 Ollama model {self.model} is ready")
        except Exception as e:
            print(f"⚠️ Could not verify Ollama model: {e}")

    def index_document(self, file_path: str, chunk_method: str = "fixed"):
        print(f"\n{'='*60}")
        print(f"📚 INDEXING: {file_path} ({chunk_method})")
        print(f"{'='*60}")

        processor = DocumentProcessor(chunk_size=500, chunk_overlap=100)
        chunks = processor.process_document(file_path, chunk_method=chunk_method)

        chunk_texts = [chunk["content"] for chunk in chunks]
        embeddings = self.embedding_engine.embed_batch(chunk_texts)

        self.vector_store.add_documents(chunks=chunks, embeddings=embeddings)
        
        # Graph Extraction
        print("🔗 Extracting Relationships for GraphRAG...")
        for chunk in chunks[:10]: # Limit to first 10 for performance
            triplets = self.graph_engine.extract_triplets(chunk['content'], model=self.model)
            self.graph_engine.add_to_graph(triplets, source=file_path)
            
        print(f"\n✅ Document indexed (Vector + Graph)!")

    def retrieve_context(self, query: str) -> List[Dict]:
        results = self.vector_store.search(query, n_results=self.n_retrieval_results)

        retrieved_chunks = []
        for i in range(len(results['documents'][0])):
            chunk = {
                'id': i,
                'content': results['documents'][0][i],
                'metadata': results['metadatas'][0][i],
                'distance': results['distances'][0][i],
                'relevance_score': 1 - results['distances'][0][i]
            }
            retrieved_chunks.append(chunk)

        if self.use_reranker and len(retrieved_chunks) > 1:
            print(f"⚖️ Reranking {len(retrieved_chunks)} chunks...")
            formatted_passages = [
                {"id": i, "text": c["content"], "meta": c["metadata"]} 
                for i, c in enumerate(retrieved_chunks)
            ]
            
            from flashrank import RerankRequest
            rank_request = RerankRequest(query=query, passages=formatted_passages)
            reranked_results = self.ranker.rerank(rank_request)
            
            final_chunks = []
            for r in reranked_results[:5]:
                final_chunks.append({
                    'content': r['text'],
                    'metadata': r['meta'],
                    'relevance_score': r['score']
                })
            return final_chunks

        return retrieved_chunks[:5]

    def generate_answer(self, prompt: str) -> Dict:
        data = {
            "model": self.model,
            "prompt": prompt,
            "stream": False,
            "options": {"temperature": 0.1, "num_predict": 800}
        }
        try:
            response = requests.post(f"{self.ollama_url}/api/generate", json=data, timeout=180)
            if response.status_code == 200:
                result = response.json()
                return {'answer': result.get('response', '').strip(), 'model': self.model}
            else:
                raise Exception(f"Ollama error: {response.status_code}")
        except Exception as e:
            return {'answer': f"Error generating answer: {e}", 'model': self.model}

    def generate_hyde_passage(self, question: str) -> str:
        """
        Generate a hypothetical answer to the question for better retrieval.
        """
        print(f"🧠 Generating HyDE passage for: '{question}'")
        prompt = f"Write a paragraph that could serve as a direct, technical answer to the following question: {question}. Provide only the answer text, no introduction."
        
        result = self.generate_answer(prompt)
        return result['answer']

    def query(self, question: str, verbose: bool = True) -> Dict:
        if verbose:
            print(f"\n❓ QUESTION: {question}")

        # HyDE Query Expansion
        search_query = question
        if self.use_hyde:
            search_query = self.generate_hyde_passage(question)
            if verbose:
                print(f"🔍 Searching with HyDE Expansion: {search_query[:100]}...")

        chunks = self.retrieve_context(search_query)
        
        # Safety check: if no chunks found
        if not chunks:
            return {
                'answer': "I don't have any documents indexed yet. Please upload and index a document first.",
                'sources': [],
                'metadata': {}
            }

        # Graph Enhancement
        graph_context = []
        # Attempt to find related nodes for key terms
        for word in question.split():
            if len(word) > 4: # Simple heuristic for entities
                rels = self.graph_engine.search_relationships(word)
                graph_context.extend(rels)
        
        graph_str = "\n".join(graph_context[:5]) if graph_context else "None"

        chunks = check_context_size(chunks, limit=2500)
        context_str = build_context(chunks)

        full_prompt = f"""{self.system_prompt}

Document Context:
{context_str}

Knowledge Graph Context:
{graph_str}

Question: {question}

Answer:"""

        result = self.generate_answer(full_prompt)
        
        # VALIDATION
        valid, msg = validate_answer(result['answer'], chunks)
        final_answer = rag_guard(result['answer'], chunks)

        return {
            'answer': final_answer,
            'sources': chunks,
            'metadata': {**result, 'hyde_expanded': self.use_hyde},
            'grounded': valid
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
