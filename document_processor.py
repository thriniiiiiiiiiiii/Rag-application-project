"""
Document Processor: Load PDFs and split into chunks
This is the first step in the indexing phase
"""

import os
from typing import List, Dict
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter


class DocumentProcessor:
    """
    Handles loading PDFs and splitting them into chunks
    
    Why chunking?
    - LLMs have token limits (can't process entire books)
    - Smaller chunks = more precise retrieval
    - Chunks need enough context to be meaningful
    
    Chunk size: 500 characters (~125 words)
    Overlap: 50 characters (prevents cutting sentences)
    """
    
    def __init__(
        self,
        chunk_size: int = 500,
        chunk_overlap: int = 50
    ):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        
        # RecursiveCharacterTextSplitter tries to split on:
        # 1. Double newlines (paragraphs)
        # 2. Single newlines
        # 3. Spaces
        # In that order - maintains document structure!
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            length_function=len,
            separators=["\n\n", "\n", " ", ""]
        )
    
    def load_pdf(self, pdf_path: str) -> str:
        """Extract text from PDF file"""
        print(f"📄 Loading PDF: {pdf_path}")
        
        if not os.path.exists(pdf_path):
            raise FileNotFoundError(f"PDF not found: {pdf_path}")
        
        try:
            reader = PdfReader(pdf_path)
            text = ""
            for page in reader.pages:
                text += (page.extract_text() or "") + "\n"
            return text
        except Exception as e:
            raise Exception(f"Error loading PDF: {str(e)}")

    def load_docx(self, docx_path: str) -> str:
        """Extract text from DOCX file"""
        from docx import Document
        print(f"📄 Loading DOCX: {docx_path}")
        
        try:
            doc = Document(docx_path)
            return "\n".join([para.text for para in doc.paragraphs])
        except Exception as e:
            raise Exception(f"Error loading DOCX: {str(e)}")

    def load_text(self, file_path: str) -> str:
        """Extract text from TXT or MD file"""
        print(f"📄 Loading Text/MD: {file_path}")
        try:
            with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
                return f.read()
        except Exception as e:
            raise Exception(f"Error loading text file: {str(e)}")
    
    def semantic_split(self, text: str, threshold: float = 0.6) -> List[str]:
        """
        Split text based on semantic topic shifts rather than fixed size.
        """
        import re
        from sklearn.metrics.pairwise import cosine_similarity
        from sentence_transformers import SentenceTransformer
        
        # 1. Split into sentences
        sentences = re.split(r'(?<=[.!?]) +', text)
        if len(sentences) < 2:
            return [text]

        # 2. Get embeddings for sentences
        model = SentenceTransformer("all-MiniLM-L6-v2")
        embeddings = model.encode(sentences)
        
        # 3. Calculate similarities between adjacent sentences
        chunks = []
        current_chunk = [sentences[0]]
        
        for i in range(len(sentences) - 1):
            sim = cosine_similarity([embeddings[i]], [embeddings[i+1]])[0][0]
            
            if sim < threshold:
                # Topic shift detected
                chunks.append(" ".join(current_chunk))
                current_chunk = [sentences[i+1]]
            else:
                current_chunk.append(sentences[i+1])
        
        chunks.append(" ".join(current_chunk))
        return chunks

    def create_chunks(self, text: str, source_name: str, method: str = "fixed") -> List[Dict]:
        """Split text into chunks with metadata (Fixed or Semantic)"""
        print(f"\n✂️  Creating chunks ({method}) for: {source_name}")
        
        if method == "semantic":
            raw_chunks = self.semantic_split(text)
        else:
            raw_chunks = self.text_splitter.split_text(text)
        
        chunks = []
        for i, chunk_text in enumerate(raw_chunks):
            chunk = {
                "content": chunk_text,
                "metadata": {
                    "source": source_name,
                    "chunk_id": i,
                    "char_count": len(chunk_text),
                    "method": method
                }
            }
            chunks.append(chunk)
        
        print(f"✅ Created {len(chunks)} chunks")
        return chunks
    
    def process_document(self, file_path: str, chunk_method: str = "fixed") -> List[Dict]:
        """Complete pipeline: Load File → Create Chunks"""
        filename = os.path.basename(file_path)
        ext = filename.lower().split('.')[-1]
        
        if ext == 'pdf':
            text = self.load_pdf(file_path)
        elif ext == 'docx':
            text = self.load_docx(file_path)
        elif ext in ['txt', 'md']:
            text = self.load_text(file_path)
        else:
            raise ValueError(f"Unsupported file format: {ext}")
        
        return self.create_chunks(text, filename, method=chunk_method)


# Test the processor
if __name__ == "__main__":
    """
    Test with sample text
    """
    
    processor = DocumentProcessor(chunk_size=300, chunk_overlap=50)
    
    # Sample text (simulating a PDF)
    sample_text = """
    Retrieval-Augmented Generation (RAG) is a technique that combines 
    information retrieval with text generation. It works by first retrieving 
    relevant documents from a knowledge base, then using those documents as 
    context for a language model to generate responses.
    
    The key components of RAG are:
    1. Document Processing: Breaking documents into chunks
    2. Embeddings: Converting text to numerical vectors
    3. Vector Database: Storing and searching embeddings efficiently
    4. Retrieval: Finding relevant chunks for a query
    5. Generation: Using an LLM to create responses based on retrieved context
    
    RAG solves the problem of LLMs not having access to recent or proprietary 
    information. By retrieving relevant context, the LLM can provide accurate 
    answers based on your specific documents rather than relying solely on 
    its training data.
    """ * 3  # Repeat to create enough text for multiple chunks
    
    chunks = processor.create_chunks(sample_text, "rag_guide.txt")
    
    print(f"\n📝 Sample Chunks:")
    print("=" * 60)
    for i, chunk in enumerate(chunks[:2]):  # Show first 2
        print(f"\nChunk {i + 1}:")
        print(f"Content: {chunk['content'][:150]}...")
        print(f"Characters: {chunk['metadata']['char_count']}")