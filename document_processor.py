"""
Document Processor: Load PDFs and split into chunks
This is the first step in the indexing phase
"""

import os
from typing import List, Dict
from PyPDF2 import PdfReader
from langchain_text_splitters import RecursiveCharacterTextSplitter


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
        """
        Extract text from PDF file
        
        Args:
            pdf_path: Path to PDF file
            
        Returns:
            Complete text from all pages
        """
        print(f"📄 Loading PDF: {pdf_path}")
        
        if not os.path.exists(pdf_path):
            raise FileNotFoundError(f"PDF not found: {pdf_path}")
        
        try:
            reader = PdfReader(pdf_path)
            text = ""
            
            # Extract text from each page
            for page_num, page in enumerate(reader.pages):
                page_text = page.extract_text()
                text += page_text + "\n"
                
                # Show progress for large PDFs
                if (page_num + 1) % 10 == 0:
                    print(f"   Processed {page_num + 1} pages...")
            
            print(f"✅ Loaded {len(reader.pages)} pages")
            print(f"📊 Total characters: {len(text):,}")
            
            return text
            
        except Exception as e:
            raise Exception(f"Error loading PDF: {str(e)}")
    
    def create_chunks(self, text: str, source_name: str) -> List[Dict]:
        """
        Split text into chunks with metadata
        
        Args:
            text: Full document text
            source_name: Name of source document
            
        Returns:
            List of chunks with metadata
        """
        print(f"\n✂️  Creating chunks...")
        print(f"   Chunk size: {self.chunk_size} characters")
        print(f"   Chunk overlap: {self.chunk_overlap} characters")
        
        # Split the text
        raw_chunks = self.text_splitter.split_text(text)
        
        print(f"✅ Created {len(raw_chunks)} chunks")
        
        # Add metadata to each chunk
        chunks = []
        for i, chunk_text in enumerate(raw_chunks):
            chunk = {
                "content": chunk_text,
                "metadata": {
                    "source": source_name,
                    "chunk_id": i,
                    "char_count": len(chunk_text)
                }
            }
            chunks.append(chunk)
        
        # Show statistics
        char_counts = [c["metadata"]["char_count"] for c in chunks]
        print(f"\n📊 Chunk Statistics:")
        print(f"   Average chars per chunk: {sum(char_counts) / len(char_counts):.0f}")
        print(f"   Min chars: {min(char_counts)}")
        print(f"   Max chars: {max(char_counts)}")
        
        return chunks
    
    def process_document(self, pdf_path: str) -> List[Dict]:
        """
        Complete pipeline: Load PDF → Create Chunks
        
        This is the main method you'll use
        """
        # Extract filename for metadata
        filename = os.path.basename(pdf_path)
        
        # Load PDF text
        text = self.load_pdf(pdf_path)
        
        # Create chunks
        chunks = self.create_chunks(text, filename)
        
        return chunks


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