"""Advanced RAG system with AI reranking and hybrid search."""

import os
import logging
from typing import List, Dict, Any, Optional, Tuple
from openai import OpenAI
from sqlalchemy.orm import Session
from sqlalchemy import text
import json
import numpy as np
from dotenv import load_dotenv

from .rag import RAGPipeline, cosine_similarity
from .embeddings import generate_query_embedding
from .models import QueryLog

load_dotenv()

logger = logging.getLogger(__name__)

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
client = OpenAI(api_key=OPENAI_API_KEY)


class HybridSearchRAG(RAGPipeline):
    """Advanced RAG with hybrid search and AI reranking."""
    
    def __init__(self, top_k: int = 10, final_k: int = 5, similarity_threshold: float = 0.6):
        super().__init__(top_k=top_k, similarity_threshold=similarity_threshold)
        self.final_k = final_k  # Final number after reranking
        self.rerank_model = "gpt-3.5-turbo"
        self.chat_model = os.getenv("CHAT_MODEL", "gpt-4-1106-preview")
    
    def hybrid_retrieve(self, query: str, db: Session) -> List[Dict[str, Any]]:
        """
        Hybrid retrieval combining multiple search strategies.
        
        1. Vector similarity search
        2. Keyword/BM25-like search  
        3. Concept-based search
        4. AI reranking
        """
        
        # Step 1: Vector similarity search (existing method)
        vector_results = self.retrieve_relevant_chunks(query, db, top_k=self.top_k)
        
        # Step 2: Keyword-based search
        keyword_results = self._keyword_search(query, db)
        
        # Step 3: Concept-based search using AI
        concept_results = self._concept_search(query, db)
        
        # Step 4: Combine and deduplicate results
        combined_results = self._combine_search_results(
            vector_results, keyword_results, concept_results
        )
        
        # Step 5: AI reranking
        if len(combined_results) > self.final_k:
            reranked_results = self._ai_rerank(query, combined_results)
            return reranked_results[:self.final_k]
        
        return combined_results
    
    def _keyword_search(self, query: str, db: Session) -> List[Dict[str, Any]]:
        """Simple keyword-based search using PostgreSQL full-text search."""
        try:
            # Extract keywords from query
            keywords = self._extract_keywords(query)
            
            if not keywords:
                return []
            
            # Build search query
            search_terms = " | ".join(keywords)  # OR search
            
            sql_query = text("""
                SELECT 
                    c.id,
                    c.content,
                    c.chunk_index,
                    c.token_count,
                    d.title,
                    d.url,
                    ts_rank(to_tsvector('english', c.content), plainto_tsquery('english', :search_terms)) as rank
                FROM chunks c
                JOIN documents d ON c.document_id = d.id
                WHERE to_tsvector('english', c.content) @@ plainto_tsquery('english', :search_terms)
                ORDER BY rank DESC
                LIMIT 15
            """)
            
            result = db.execute(sql_query, {"search_terms": search_terms})
            
            chunks = []
            for row in result:
                chunks.append({
                    'id': row.id,
                    'content': row.content,
                    'chunk_index': row.chunk_index,
                    'token_count': row.token_count,
                    'title': row.title,
                    'url': row.url,
                    'similarity': float(row.rank),  # Use text rank as similarity
                    'search_type': 'keyword'
                })
            
            logger.info(f"Keyword search found {len(chunks)} results")
            return chunks
            
        except Exception as e:
            logger.warning(f"Keyword search failed: {e}")
            return []
    
    def _extract_keywords(self, query: str) -> List[str]:
        """Extract relevant keywords from query."""
        # Simple keyword extraction - could be enhanced with NLP
        import re
        
        # Remove common words
        stop_words = {
            'what', 'how', 'when', 'where', 'why', 'who', 'which', 'is', 'are', 
            'can', 'could', 'should', 'would', 'do', 'does', 'did', 'the', 'a', 
            'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with'
        }
        
        words = re.findall(r'\b\w+\b', query.lower())
        keywords = [word for word in words if len(word) > 2 and word not in stop_words]
        
        return keywords[:5]  # Limit to top 5 keywords
    
    def _concept_search(self, query: str, db: Session) -> List[Dict[str, Any]]:
        """AI-powered concept search to find semantically related content."""
        try:
            # Use AI to expand query concepts
            expanded_concepts = self._expand_query_concepts(query)
            
            if not expanded_concepts:
                return []
            
            # Search for chunks containing these concepts
            concept_chunks = []
            
            for concept in expanded_concepts:
                concept_embedding = generate_query_embedding(concept)
                
                # Find chunks related to this concept using parent class method
                chunks = self.retrieve_relevant_chunks(concept, db, top_k=3)
                for chunk in chunks:
                    chunk['search_type'] = 'concept'
                    chunk['concept'] = concept
                concept_chunks.extend(chunks)
            
            logger.info(f"Concept search found {len(concept_chunks)} results")
            return concept_chunks
            
        except Exception as e:
            logger.warning(f"Concept search failed: {e}")
            return []
    
    def _expand_query_concepts(self, query: str) -> List[str]:
        """Use AI to expand query into related concepts."""
        prompt = f"""Given this wasmCloud-related question, identify 2-3 key concepts or related terms that might help find relevant information:

Question: {query}

Return related wasmCloud concepts like:
- Technical terms (actors, capabilities, providers, WASM, OCI, NATS)
- Operations (deployment, scaling, configuration, monitoring)
- Platforms (Kubernetes, cloud, edge)

Return just the concepts, one per line, no explanations."""

        try:
            response = client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[{"role": "user", "content": prompt}],
                temperature=0.3,
                max_tokens=100
            )
            
            concepts = response.choices[0].message.content.strip().split('\n')
            return [c.strip('- ').strip() for c in concepts if c.strip()]
            
        except Exception as e:
            logger.error(f"Concept expansion failed: {e}")
            return []
    
    def _combine_search_results(self, *result_lists) -> List[Dict[str, Any]]:
        """Combine and deduplicate search results from multiple sources."""
        seen_ids = set()
        combined = []
        
        # Add results from each source, avoiding duplicates
        for results in result_lists:
            for result in results:
                if result['id'] not in seen_ids:
                    seen_ids.add(result['id'])
                    combined.append(result)
        
        # Sort by similarity score (higher is better)
        combined.sort(key=lambda x: x.get('similarity', 0), reverse=True)
        
        return combined
    
    def _ai_rerank(self, query: str, chunks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Use AI to rerank chunks based on relevance to the specific query."""
        
        if len(chunks) <= self.final_k:
            return chunks
        
        # Prepare chunk summaries for AI evaluation
        chunk_summaries = []
        for i, chunk in enumerate(chunks):
            summary = {
                'index': i,
                'title': chunk['title'],
                'content_preview': chunk['content'][:200] + '...' if len(chunk['content']) > 200 else chunk['content'],
                'search_type': chunk.get('search_type', 'vector'),
                'similarity': chunk.get('similarity', 0)
            }
            chunk_summaries.append(summary)
        
        prompt = f"""Rank these wasmCloud documentation chunks by relevance to the user's question.

Question: {query}

Chunks:
{json.dumps(chunk_summaries, indent=2)}

Consider:
1. Direct relevance to the question
2. Completeness of information
3. Accuracy and specificity
4. Practical value for the user

Return a JSON list of indices in order of relevance (most relevant first), like: [2, 0, 5, 1, 3]
Include only the top {self.final_k} most relevant chunks."""
        
        try:
            response = client.chat.completions.create(
                model=self.rerank_model,
                messages=[{"role": "user", "content": prompt}],
                response_format={"type": "json_object"},
                temperature=0.1
            )
            
            rerank_result = json.loads(response.choices[0].message.content)
            ranked_indices = rerank_result.get('ranking', rerank_result.get('indices', []))
            
            # Reorder chunks based on AI ranking
            reranked_chunks = []
            for idx in ranked_indices:
                if 0 <= idx < len(chunks):
                    chunk = chunks[idx].copy()
                    chunk['rerank_position'] = len(reranked_chunks) + 1
                    reranked_chunks.append(chunk)
            
            logger.info(f"AI reranked {len(chunks)} chunks to top {len(reranked_chunks)}")
            return reranked_chunks
            
        except Exception as e:
            logger.error(f"AI reranking failed: {e}")
            # Fallback to original order
            return chunks[:self.final_k]
    
    def answer_question_advanced(self, query: str, db: Session) -> Dict[str, Any]:
        """Enhanced question answering with hybrid search and reranking."""
        import time
        start_time = time.time()
        
        # Use hybrid retrieval
        relevant_chunks = self.hybrid_retrieve(query, db)
        
        if not relevant_chunks:
            return {
                'answer': "I couldn't find relevant information in the wasmCloud documentation to answer your question. Could you try rephrasing or asking about a different aspect of wasmCloud?",
                'sources': [],
                'chunks_used': 0,
                'response_time': time.time() - start_time,
                'search_method': 'hybrid_advanced'
            }
        
        # Build enhanced context with search metadata
        context = self._build_enhanced_context(relevant_chunks)
        
        # Generate response with enhanced prompt
        answer = self._generate_enhanced_response(query, context, relevant_chunks)
        
        # Prepare sources with additional metadata
        sources = []
        for chunk in relevant_chunks:
            sources.append({
                'title': chunk['title'],
                'url': chunk['url'],
                'similarity': chunk.get('similarity', 0),
                'search_type': chunk.get('search_type', 'vector'),
                'rerank_position': chunk.get('rerank_position'),
                'concept': chunk.get('concept')
            })
        
        response_time = time.time() - start_time
        
        # Log query for analysis
        try:
            log_entry = QueryLog(
                query=query,
                response=answer,
                chunks_used=len(relevant_chunks),
                response_time=response_time
            )
            db.add(log_entry)
            db.commit()
        except Exception as e:
            logger.warning(f"Failed to log query: {e}")
        
        return {
            'answer': answer,
            'sources': sources,
            'chunks_used': len(relevant_chunks),
            'response_time': response_time,
            'search_method': 'hybrid_advanced'
        }
    
    def _build_enhanced_context(self, chunks: List[Dict[str, Any]]) -> str:
        """Build enhanced context with search metadata."""
        if not chunks:
            return "No relevant information found."
        
        context_parts = []
        for i, chunk in enumerate(chunks, 1):
            search_info = f"[Found via {chunk.get('search_type', 'vector')} search"
            if chunk.get('concept'):
                search_info += f", concept: {chunk['concept']}"
            search_info += "]"
            
            context_parts.append(
                f"Source {i} {search_info}: {chunk['title']} - {chunk['url']}\n"
                f"{chunk['content']}\n"
            )
        
        return "\n---\n".join(context_parts)
    
    def _generate_enhanced_response(self, query: str, context: str, chunks: List[Dict[str, Any]]) -> str:
        """Generate enhanced response with better prompting."""
        
        search_methods = list(set(chunk.get('search_type', 'vector') for chunk in chunks))
        
        prompt = f"""You are an expert wasmCloud assistant with access to official documentation. Answer the user's question using the provided context.

SEARCH CONTEXT: Information gathered using {', '.join(search_methods)} search methods for comprehensive coverage.

GUIDELINES:
1. Provide accurate, specific information based on the context
2. Use examples and code snippets when available
3. Structure your response clearly with sections if needed
4. Mention specific wasmCloud concepts, tools, and best practices
5. If the context is incomplete, acknowledge limitations and suggest next steps
6. Always cite sources when making specific claims

CONTEXT:
{context}

QUESTION: {query}

ANSWER:"""
        
        try:
            response = client.chat.completions.create(
                model=self.chat_model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.1,
                max_tokens=2000
            )
            
            return response.choices[0].message.content.strip()
            
        except Exception as e:
            logger.error(f"Error generating enhanced response: {e}")
            return "I apologize, but I encountered an error while generating a response. Please try again."


# Global instance
advanced_rag = HybridSearchRAG()


def ask_question_advanced(query: str, db: Session) -> Dict[str, Any]:
    """Advanced question answering with hybrid search and AI reranking."""
    return advanced_rag.answer_question_advanced(query, db) 