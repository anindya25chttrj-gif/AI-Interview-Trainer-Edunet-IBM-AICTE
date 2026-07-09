import os
import numpy as np
from pypdf import PdfReader
from sentence_transformers import SentenceTransformer
from ibm_watsonx_ai import APIClient, Credentials
from ibm_watsonx_ai.foundation_models import ModelInference

class CoreRAGEngine:
    def __init__(self):
        print("Initializing Semantic Embedding Model (all-MiniLM-L6-v2)...")
        # Lightweight, high-performance embedding model that runs locally on your laptop CPU/GPU
        self.embedding_model = SentenceTransformer("all-MiniLM-L6-v2")
        self.chunks = []
        self.vector_database = None

    def load_and_chunk_pdf(self, pdf_path, chunk_size=500, overlap=100):
        """Parses a local PDF document, splits it into structured token chunks."""
        if not os.path.exists(pdf_path):
            print(f"Warning: Source document {pdf_path} not found.")
            return

        print(f"Ingesting knowledge source: {os.path.basename(pdf_path)}")
        reader = PdfReader(pdf_path)
        full_text = ""
        
        for page in reader.pages:
            text = page.extract_text()
            if text:
                full_text += text + "\n"

        # Sliding window text chunking strategy to preserve paragraph context boundaries
        words = full_text.split()
        for i in range(0, len(words), chunk_size - overlap):
            chunk = " ".join(words[i:i + chunk_size])
            self.chunks.append(f"[{os.path.basename(pdf_path)}] {chunk}")

    def build_vector_space(self):
        """Converts text chunks into a dense matrix representation (In-Memory Vector DB)."""
        if not self.chunks:
            print("No text data found to vectorize.")
            return
        print(f"Vectorizing {len(self.chunks)} text blocks into multi-dimensional matrix space...")
        self.vector_database = self.embedding_model.encode(self.chunks, convert_to_numpy=True)
        print("Vector database built successfully.")

    def retrieve_relevant_context(self, user_query, top_k=3):
        """Performs vector dot-product similarity comparison to extract top-k context blocks."""
        if self.vector_database is None or len(self.vector_database) == 0:
            return ""

        query_vector = self.embedding_model.encode([user_query], convert_to_numpy=True)
        
        # Mathematical Vector Dot-Product calculation for cosine similarity matching
        scores = np.dot(self.vector_database, query_vector.T).flatten()
        top_indices = np.argsort(scores)[::-1][:top_k]
        
        retrieved_blocks = [self.chunks[idx] for idx in top_indices]
        return "\n\n".join(retrieved_blocks)

# Global initialization wrapper to load your files directly
rag_instance = CoreRAGEngine()
# Grounding system with your specific resume writing guide and interview database
rag_instance.load_and_chunk_pdf("guide-to-resume-writing.pdf")
rag_instance.load_and_chunk_pdf("Top-50-Interview-Questions-and-Answers.pdf")
rag_instance.build_vector_space()