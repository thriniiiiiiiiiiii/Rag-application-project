"""
RAG Quality Testing Framework
Tests retrieval accuracy, answer quality, and hallucination detection
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from rag_engine import RAGEngineOllama
from document_processor import DocumentProcessor
from typing import List, Dict
import json


class RAGQualityTester:
    """
    Comprehensive test suite for RAG system quality
    
    Tests:
    1. Retrieval Quality - Are correct chunks retrieved?
    2. Answer Accuracy - Is the answer correct?
    3. Hallucination Detection - Does it make things up?
    4. Context Grounding - Does answer use provided context?
    """
    
    def __init__(self, rag_engine: RAGEngineOllama):
        self.rag = rag_engine
        self.test_results = []
    
    def create_test_dataset(self) -> List[Dict]:
        """
        Create test questions with expected characteristics
        
        Customize these based on YOUR documents!
        """
        return [
            {
                "question": "How long do I have to return a product?",
                "expected_keywords": ["30 days", "refund", "return"],
                "should_find_answer": True,
                "category": "factual"
            },
            {
                "question": "What is the refund process?",
                "expected_keywords": ["email", "support", "order number"],
                "should_find_answer": True,
                "category": "process"
            },
            {
                "question": "What is the CEO's favorite color?",  # Not in doc
                "expected_keywords": [],
                "should_find_answer": False,
                "category": "hallucination_test"
            },
            {
                "question": "Are shipping costs refundable?",
                "expected_keywords": ["shipping", "non-refundable", "damaged"],
                "should_find_answer": True,
                "category": "specific_fact"
            }
        ]
    
    def test_retrieval_quality(self, question: str, expected_keywords: List[str]) -> Dict:
        """
        Test if retrieval finds relevant chunks
        
        Returns:
            Dictionary with retrieval metrics
        """
        chunks = self.rag.retrieve_context(question)
        
        if not chunks:
            return {
                "score": 0.0,
                "status": "FAIL",
                "reason": "No chunks retrieved"
            }
        
        # Combine all retrieved text
        all_text = " ".join([c['content'].lower() for c in chunks])
        
        # Check keyword presence
        keywords_found = sum(1 for kw in expected_keywords if kw.lower() in all_text)
        
        if expected_keywords:
            score = keywords_found / len(expected_keywords)
        else:
            score = 1.0  # No keywords to check
        
        # Check relevance scores
        avg_relevance = sum(c['relevance_score'] for c in chunks) / len(chunks)
        
        return {
            "score": score,
            "keywords_found": f"{keywords_found}/{len(expected_keywords)}",
            "avg_relevance": f"{avg_relevance:.3f}",
            "top_relevance": f"{chunks[0]['relevance_score']:.3f}",
            "status": "PASS" if score >= 0.5 else "FAIL"
        }
    
    def test_answer_grounding(self, question: str, should_find_answer: bool) -> Dict:
        """
        Test if answer is properly grounded in context
        
        Checks:
        - If should answer: gives substantive response
        - If shouldn't answer: says "I don't have that information"
        """
        result = self.rag.query(question, verbose=False)
        answer = result['answer'].lower()
        sources = result['sources']
        
        # Phrases indicating "I don't know"
        unknown_phrases = [
            "don't have",
            "not in the",
            "no information",
            "cannot answer",
            "not provided",
            "not mentioned",
            "i don't know"
        ]
        
        says_unknown = any(phrase in answer for phrase in unknown_phrases)
        
        # Determine pas