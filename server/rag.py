"""RAG (Retrieval-Augmented Generation) pipeline for wasmCloud bot."""

import os
import time
import logging
import json
import numpy as np
from typing import List, Dict, Any, Optional
from sqlalchemy import text
from sqlalchemy.orm import Session
from openai import OpenAI
from dotenv import load_dotenv

from .database import get_db
from .models import Chunk, Document, QueryLog, PGVECTOR_AVAILABLE
from .embeddings import generate_query_embedding

load_dotenv()

logger = logging.getLogger(__name__)

# Configuration
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
CHAT_MODEL = os.getenv("CHAT_MODEL", "gpt-4-1106-preview")

if not OPENAI_API_KEY:
    raise ValueError("OPENAI_API_KEY environment variable is required")

# Initialize OpenAI client
client = OpenAI(api_key=OPENAI_API_KEY)


def cosine_similarity(a: List[float], b: List[float]) -> float:
    """Calculate cosine similarity between two vectors."""
    a_np = np.array(a)
    b_np = np.array(b)
    return np.dot(a_np, b_np) / (np.linalg.norm(a_np) * np.linalg.norm(b_np))


class RAGPipeline:
    """RAG pipeline for answering questions about wasmCloud."""
    
    def __init__(self, top_k: int = 5, similarity_threshold: float = 0.7):
        self.top_k = top_k
        self.similarity_threshold = similarity_threshold
        self.client = client
    
    def retrieve_relevant_chunks(
        self, 
        query: str, 
        db: Session,
        top_k: Optional[int] = None
    ) -> List[Dict[str, Any]]:
        """
        Retrieve the most relevant chunks for a query using vector similarity.
        
        Args:
            query: User's question
            db: Database session
            top_k: Number of chunks to retrieve (defaults to self.top_k)
            
        Returns:
            List of relevant chunks with metadata
        """
        k = top_k or self.top_k
        
        try:
            # Generate embedding for the query
            query_embedding = generate_query_embedding(query)
            
            if PGVECTOR_AVAILABLE:
                # Use pgvector for similarity search
                embedding_str = "[" + ",".join(map(str, query_embedding)) + "]"
                
                sql_query = text("""
                    SELECT 
                        c.id,
                        c.content,
                        c.chunk_index,
                        c.token_count,
                        d.title,
                        d.url,
                        (c.embedding <=> :query_embedding) as distance,
                        (1 - (c.embedding <=> :query_embedding)) as similarity
                    FROM chunks c
                    JOIN documents d ON c.document_id = d.id
                    WHERE (1 - (c.embedding <=> :query_embedding)) > :threshold
                    ORDER BY c.embedding <=> :query_embedding
                    LIMIT :limit
                """)
                
                result = db.execute(
                    sql_query,
                    {
                        "query_embedding": embedding_str,
                        "threshold": self.similarity_threshold,
                        "limit": k
                    }
                )
                
                chunks = []
                for row in result:
                    chunks.append({
                        'id': row.id,
                        'content': row.content,
                        'chunk_index': row.chunk_index,
                        'token_count': row.token_count,
                        'title': row.title,
                        'url': row.url,
                        'similarity': float(row.similarity),
                        'distance': float(row.distance)
                    })
            else:
                # Fallback to manual similarity calculation with JSON embeddings
                sql_query = text("""
                    SELECT 
                        c.id,
                        c.content,
                        c.chunk_index,
                        c.token_count,
                        c.embedding,
                        d.title,
                        d.url
                    FROM chunks c
                    JOIN documents d ON c.document_id = d.id
                """)
                
                result = db.execute(sql_query)
                
                chunks_with_similarity = []
                for row in result:
                    try:
                        chunk_embedding = json.loads(row.embedding) if isinstance(row.embedding, str) else row.embedding
                        similarity = cosine_similarity(query_embedding, chunk_embedding)
                        
                        if similarity > self.similarity_threshold:
                            chunks_with_similarity.append({
                                'id': row.id,
                                'content': row.content,
                                'chunk_index': row.chunk_index,
                                'token_count': row.token_count,
                                'title': row.title,
                                'url': row.url,
                                'similarity': similarity,
                                'distance': 1.0 - similarity
                            })
                    except Exception as e:
                        logger.warning(f"Error processing chunk {row.id}: {e}")
                        continue
                
                # Sort by similarity and take top k
                chunks = sorted(chunks_with_similarity, key=lambda x: x['similarity'], reverse=True)[:k]
            
            logger.info(f"Retrieved {len(chunks)} relevant chunks for query")
            return chunks
            
        except Exception as e:
            logger.error(f"Error retrieving chunks: {e}")
            raise
    
    def _build_context(self, chunks: List[Dict[str, Any]]) -> str:
        """Build context string from retrieved chunks."""
        if not chunks:
            return "No relevant information found."
        
        context_parts = []
        for i, chunk in enumerate(chunks, 1):
            context_parts.append(
                f"[Source {i}: {chunk['title']} - {chunk['url']}]\n"
                f"{chunk['content']}\n"
            )
        
        return "\n---\n".join(context_parts)
    
    def _build_prompt(self, query: str, context: str) -> str:
        """Build the prompt for GPT-4."""
        return f"""You are a helpful assistant that answers questions about wasmCloud based on the official documentation. 

Use the following context from the wasmCloud documentation to answer the user's question. If the context doesn't contain enough information to answer the question, say so and suggest what additional information might be needed.

Always cite your sources by mentioning the relevant documentation sections when possible.

Context:
{context}

Question: {query}

Answer:"""
    
    def generate_response(
        self, 
        query: str, 
        context: str,
        model: Optional[str] = None
    ) -> str:
        """Generate response using GPT-4."""
        try:
            prompt = self._build_prompt(query, context)
            
            response = self.client.chat.completions.create(
                model=model or CHAT_MODEL,
                messages=[
                    {
                        "role": "system",
                        "content": "You are a helpful assistant that answers questions about wasmCloud, a CNCF project for building and running WebAssembly applications. Provide accurate, helpful responses based on the documentation context provided."
                    },
                    {
                        "role": "user",
                        "content": prompt
                    }
                ],
                max_tokens=1000,
                temperature=0.7
            )
            
            return response.choices[0].message.content.strip()
            
        except Exception as e:
            logger.error(f"Error generating response: {e}")
            raise
    
    def answer_question(
        self, 
        query: str,
        db: Session,
        include_sources: bool = True
    ) -> Dict[str, Any]:
        """
        Complete RAG pipeline: retrieve relevant chunks and generate answer.
        
        Args:
            query: User's question
            db: Database session
            include_sources: Whether to include source information in response
            
        Returns:
            Dictionary with answer, sources, and metadata
        """
        start_time = time.time()
        
        try:
            # Retrieve relevant chunks
            chunks = self.retrieve_relevant_chunks(query, db)
            
            if not chunks:
                return {
                    'answer': "I couldn't find relevant information in the wasmCloud documentation to answer your question. Please try rephrasing your question or asking about a different topic.",
                    'sources': [],
                    'chunks_used': 0,
                    'response_time': time.time() - start_time
                }
            
            # Build context and generate response
            context = self._build_context(chunks)
            answer = self.generate_response(query, context)
            
            # Prepare sources information
            sources = []
            if include_sources:
                seen_urls = set()
                for chunk in chunks:
                    if chunk['url'] not in seen_urls:
                        sources.append({
                            'title': chunk['title'],
                            'url': chunk['url'],
                            'similarity': chunk['similarity']
                        })
                        seen_urls.add(chunk['url'])
            
            response_time = time.time() - start_time
            
            # Log the query
            query_log = QueryLog(
                query=query,
                response=answer,
                response_time=response_time,
                chunks_used=len(chunks)
            )
            db.add(query_log)
            db.commit()
            
            return {
                'answer': answer,
                'sources': sources,
                'chunks_used': len(chunks),
                'response_time': response_time
            }
            
        except Exception as e:
            logger.error(f"Error in RAG pipeline: {e}")
            raise


# Global RAG pipeline instance
rag_pipeline = RAGPipeline()


def ask_question(query: str, db: Session) -> Dict[str, Any]:
    """Convenience function to ask a question using the RAG pipeline."""
    return rag_pipeline.answer_question(query, db) 