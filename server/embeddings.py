"""Embeddings utilities for wasmCloud RAG bot."""

import os
import logging
from typing import List, Dict, Any
import tiktoken
import openai
from openai import OpenAI
import numpy as np
from dotenv import load_dotenv

load_dotenv()

logger = logging.getLogger(__name__)

# Configuration
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "text-embedding-3-small")
CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "1000"))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "200"))

if not OPENAI_API_KEY:
    raise ValueError("OPENAI_API_KEY environment variable is required")

# Initialize OpenAI client
client = OpenAI(api_key=OPENAI_API_KEY)


class TextChunker:
    """Utility for chunking text into smaller pieces."""
    
    def __init__(self, chunk_size: int = CHUNK_SIZE, chunk_overlap: int = CHUNK_OVERLAP):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.encoding = tiktoken.get_encoding("cl100k_base")  # GPT-4 encoding
    
    def count_tokens(self, text: str) -> int:
        """Count tokens in text."""
        return len(self.encoding.encode(text))
    
    def chunk_text(self, text: str, metadata: Dict[str, Any] = None) -> List[Dict[str, Any]]:
        """
        Chunk text into smaller pieces with overlap.
        
        Args:
            text: Text to chunk
            metadata: Optional metadata to attach to each chunk
            
        Returns:
            List of chunk dictionaries with content, token_count, and metadata
        """
        if not text.strip():
            return []
        
        # Split text into sentences for better chunking
        sentences = self._split_into_sentences(text)
        
        chunks = []
        current_chunk = ""
        current_tokens = 0
        chunk_index = 0
        
        for sentence in sentences:
            sentence_tokens = self.count_tokens(sentence)
            
            # If adding this sentence would exceed chunk size, finalize current chunk
            if current_tokens + sentence_tokens > self.chunk_size and current_chunk:
                chunk_data = {
                    'content': current_chunk.strip(),
                    'token_count': current_tokens,
                    'chunk_index': chunk_index,
                }
                if metadata:
                    chunk_data.update(metadata)
                
                chunks.append(chunk_data)
                chunk_index += 1
                
                # Start new chunk with overlap
                overlap_text = self._get_overlap_text(current_chunk)
                current_chunk = overlap_text + " " + sentence
                current_tokens = self.count_tokens(current_chunk)
            else:
                current_chunk += " " + sentence if current_chunk else sentence
                current_tokens += sentence_tokens
        
        # Add the last chunk if it has content
        if current_chunk.strip():
            chunk_data = {
                'content': current_chunk.strip(),
                'token_count': current_tokens,
                'chunk_index': chunk_index,
            }
            if metadata:
                chunk_data.update(metadata)
            
            chunks.append(chunk_data)
        
        logger.info(f"Created {len(chunks)} chunks from text with {self.count_tokens(text)} tokens")
        return chunks
    
    def _split_into_sentences(self, text: str) -> List[str]:
        """Split text into sentences."""
        # Simple sentence splitting - could be improved with NLTK or spaCy
        import re
        sentences = re.split(r'(?<=[.!?])\s+', text)
        return [s.strip() for s in sentences if s.strip()]
    
    def _get_overlap_text(self, text: str) -> str:
        """Get overlap text from the end of current chunk."""
        tokens = self.encoding.encode(text)
        if len(tokens) <= self.chunk_overlap:
            return text
        
        overlap_tokens = tokens[-self.chunk_overlap:]
        return self.encoding.decode(overlap_tokens)


class EmbeddingGenerator:
    """Generate embeddings using OpenAI API."""
    
    def __init__(self, model: str = EMBEDDING_MODEL):
        self.model = model
        self.client = client
    
    def generate_embedding(self, text: str) -> List[float]:
        """Generate embedding for a single text."""
        try:
            response = self.client.embeddings.create(
                model=self.model,
                input=text
            )
            return response.data[0].embedding
        except Exception as e:
            logger.error(f"Error generating embedding: {e}")
            raise
    
    def generate_embeddings_batch(self, texts: List[str]) -> List[List[float]]:
        """Generate embeddings for multiple texts."""
        try:
            response = self.client.embeddings.create(
                model=self.model,
                input=texts
            )
            return [data.embedding for data in response.data]
        except Exception as e:
            logger.error(f"Error generating batch embeddings: {e}")
            raise
    
    def cosine_similarity(self, embedding1: List[float], embedding2: List[float]) -> float:
        """Calculate cosine similarity between two embeddings."""
        vec1 = np.array(embedding1)
        vec2 = np.array(embedding2)
        
        dot_product = np.dot(vec1, vec2)
        norm1 = np.linalg.norm(vec1)
        norm2 = np.linalg.norm(vec2)
        
        if norm1 == 0 or norm2 == 0:
            return 0.0
        
        return dot_product / (norm1 * norm2)


# Global instances
chunker = TextChunker()
embedding_generator = EmbeddingGenerator()


def chunk_document(content: str, title: str, url: str) -> List[Dict[str, Any]]:
    """Chunk a document and prepare for embedding."""
    metadata = {
        'title': title,
        'url': url
    }
    
    return chunker.chunk_text(content, metadata)


def generate_chunk_embedding(chunk_content: str) -> List[float]:
    """Generate embedding for a chunk."""
    return embedding_generator.generate_embedding(chunk_content)


def generate_query_embedding(query: str) -> List[float]:
    """Generate embedding for a search query."""
    return embedding_generator.generate_embedding(query) 