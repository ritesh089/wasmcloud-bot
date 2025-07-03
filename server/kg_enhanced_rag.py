"""Knowledge Graph Enhanced RAG pipeline for wasmCloud bot."""

import os
import logging
import time
from typing import List, Dict, Any, Optional
from openai import OpenAI
from sqlalchemy.orm import Session
from dotenv import load_dotenv

from .advanced_rag import HybridSearchRAG
from .knowledge_graph import KnowledgeGraphRetriever
from .models import QueryLog

load_dotenv()

logger = logging.getLogger(__name__)

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
client = OpenAI(api_key=OPENAI_API_KEY)


class KGEnhancedRAG(HybridSearchRAG):
    """RAG pipeline enhanced with knowledge graph reasoning."""
    
    def __init__(self, top_k: int = 10, final_k: int = 5, kg_triplets: int = 15, similarity_threshold: float = 0.6):
        super().__init__(top_k=top_k, final_k=final_k, similarity_threshold=similarity_threshold)
        self.kg_triplets = kg_triplets
        self.kg_retriever = KnowledgeGraphRetriever()
        self.reasoning_model = "gpt-4-1106-preview"
    
    def answer_question_kg_enhanced(self, query: str, db: Session) -> Dict[str, Any]:
        """
        Answer a question using hybrid search + knowledge graph reasoning.
        
        Process:
        1. Hybrid retrieval (vector + keyword + concept search)
        2. Knowledge graph context retrieval
        3. AI reasoning with both text chunks and KG triplets
        4. Enhanced response generation
        """
        
        start_time = time.time()
        
        try:
            # Step 1: Get relevant chunks using hybrid search
            logger.info(f"Starting KG-enhanced query: {query}")
            relevant_chunks = self.hybrid_retrieve(query, db)
            
            # Step 2: Get knowledge graph context
            kg_context_data = self.kg_retriever.retrieve_kg_context(query, db, max_triplets=self.kg_triplets)
            
            # Step 3: Perform knowledge graph reasoning
            reasoning_result = self._perform_kg_reasoning(query, relevant_chunks, kg_context_data)
            
            # Step 4: Generate enhanced response
            response = self._generate_kg_enhanced_response(
                query, 
                relevant_chunks, 
                kg_context_data, 
                reasoning_result
            )
            
            # Calculate timing
            processing_time = time.time() - start_time
            
            # Prepare sources for API compatibility
            sources = []
            for chunk in relevant_chunks:
                sources.append({
                    'title': chunk.get('title', 'Unknown'),
                    'url': chunk.get('url', ''),
                    'similarity': chunk.get('similarity', 0),
                    'search_type': chunk.get('search_type', 'vector'),
                    'rerank_position': chunk.get('rerank_position'),
                    'concept': chunk.get('concept')
                })
            
            # Log the query
            self._log_kg_query(query, response, processing_time, len(relevant_chunks), 
                             kg_context_data["triplet_count"], db)
            
            return {
                "answer": response,
                "sources": sources,
                "chunks_used": len(relevant_chunks),
                "triplets_used": kg_context_data["triplet_count"],
                "entities_found": kg_context_data["entity_count"],
                "processing_time": processing_time,
                "reasoning_insights": reasoning_result.get("insights", []),
                "kg_context": kg_context_data.get("context", ""),
                "search_strategy": "kg_enhanced"
            }
            
        except Exception as e:
            logger.error(f"Error in KG-enhanced RAG: {e}")
            # Fallback to hybrid search
            return self.answer_question_advanced(query, db)
    
    def _perform_kg_reasoning(self, query: str, chunks: List[Dict[str, Any]], 
                            kg_context_data: Dict[str, Any]) -> Dict[str, Any]:
        """Perform AI reasoning using both chunks and knowledge graph data."""
        
        try:
            # Build reasoning context
            chunks_context = self._build_chunks_summary(chunks)
            kg_triplets = kg_context_data.get("triplets", [])
            
            # Create reasoning prompt
            reasoning_prompt = self._create_reasoning_prompt(query, chunks_context, kg_triplets)
            
            # Call GPT-4 for reasoning
            response = client.chat.completions.create(
                model=self.reasoning_model,
                messages=[
                    {"role": "system", "content": self._get_reasoning_system_prompt()},
                    {"role": "user", "content": reasoning_prompt}
                ],
                temperature=0.2,
                max_tokens=1000
            )
            
            reasoning_text = response.choices[0].message.content
            
            # Parse reasoning insights
            insights = self._parse_reasoning_insights(reasoning_text)
            
            return {
                "reasoning": reasoning_text,
                "insights": insights,
                "confidence": self._calculate_reasoning_confidence(kg_triplets, chunks)
            }
            
        except Exception as e:
            logger.error(f"Error in KG reasoning: {e}")
            return {"reasoning": "", "insights": [], "confidence": 0.5}
    
    def _get_reasoning_system_prompt(self) -> str:
        """System prompt for knowledge graph reasoning."""
        return """You are an expert wasmCloud consultant performing multi-step reasoning to answer technical questions.

You have access to:
1. Relevant text chunks from documentation
2. Knowledge graph triplets showing relationships between concepts

Your task:
1. Analyze the query to understand what information is needed
2. Examine text chunks for direct information
3. Use knowledge graph triplets to find related concepts and infer connections
4. Identify any gaps or contradictions between sources
5. Provide structured reasoning insights

Focus on:
- Technical relationships (what implements/provides/requires what)
- Architectural patterns and dependencies  
- Configuration and deployment relationships
- Platform and integration connections

Return reasoning in this format:
**Direct Evidence:** [what the text chunks directly state]
**Graph Connections:** [what the knowledge graph reveals about relationships]
**Inferred Insights:** [logical conclusions from combining both sources]
**Confidence Assessment:** [how confident you are in the reasoning]

Be concise but thorough. Highlight any missing information or areas of uncertainty."""
    
    def _create_reasoning_prompt(self, query: str, chunks_context: str, 
                               kg_triplets: List[Dict[str, Any]]) -> str:
        """Create reasoning prompt combining chunks and KG data."""
        
        # Format triplets for readability
        triplets_text = ""
        if kg_triplets:
            triplets_text = "\n**Knowledge Graph Relationships:**\n"
            for i, triplet in enumerate(kg_triplets[:10], 1):  # Limit to top 10
                triplets_text += f"{i}. {triplet['subject']} → {triplet['predicate']} → {triplet['object']} (confidence: {triplet['confidence']:.2f})\n"
        
        prompt = f"""Question: {query}

**Available Text Information:**
{chunks_context}

{triplets_text}

Perform reasoning to answer the question using both the text information and knowledge graph relationships. Consider how the relationships in the knowledge graph support, extend, or clarify the information in the text chunks."""
        
        return prompt
    
    def _build_chunks_summary(self, chunks: List[Dict[str, Any]]) -> str:
        """Build a concise summary of chunk contents for reasoning."""
        
        if not chunks:
            return "No relevant text chunks found."
        
        summary_parts = []
        for i, chunk in enumerate(chunks[:5], 1):  # Limit to top 5 chunks
            # Truncate very long chunks
            content = chunk['content']
            if len(content) > 500:
                content = content[:500] + "..."
            
            summary_parts.append(f"Chunk {i} (from {chunk['title']}):\n{content}\n")
        
        return "\n".join(summary_parts)
    
    def _parse_reasoning_insights(self, reasoning_text: str) -> List[str]:
        """Extract key insights from reasoning text."""
        
        insights = []
        
        # Look for specific sections
        sections = [
            "Direct Evidence:",
            "Graph Connections:",
            "Inferred Insights:",
            "Key Findings:",
            "Important:"
        ]
        
        for section in sections:
            if section in reasoning_text:
                start = reasoning_text.find(section) + len(section)
                # Find next section or end
                end = len(reasoning_text)
                for other_section in sections:
                    if other_section != section and other_section in reasoning_text[start:]:
                        next_pos = reasoning_text.find(other_section, start)
                        if next_pos < end:
                            end = next_pos
                
                section_content = reasoning_text[start:end].strip()
                if section_content:
                    insights.append(f"{section} {section_content}")
        
        # If no structured sections, extract sentences with key indicators
        if not insights:
            import re
            sentences = re.split(r'[.!?]+', reasoning_text)
            key_indicators = ['therefore', 'thus', 'this means', 'indicates', 'suggests', 'reveals']
            
            for sentence in sentences:
                if any(indicator in sentence.lower() for indicator in key_indicators):
                    insights.append(sentence.strip())
        
        return insights[:5]  # Limit to top 5 insights
    
    def _calculate_reasoning_confidence(self, kg_triplets: List[Dict[str, Any]], 
                                      chunks: List[Dict[str, Any]]) -> float:
        """Calculate confidence score for reasoning based on available data."""
        
        confidence_factors = []
        
        # Factor 1: Number of relevant sources
        source_factor = min(1.0, (len(chunks) + len(kg_triplets)) / 10)
        confidence_factors.append(source_factor)
        
        # Factor 2: Average KG triplet confidence
        if kg_triplets:
            avg_kg_confidence = sum(t["confidence"] for t in kg_triplets) / len(kg_triplets)
            confidence_factors.append(avg_kg_confidence)
        
        # Factor 3: Chunk relevance (based on similarity scores)
        if chunks:
            chunk_similarities = [c.get("similarity", 0.5) for c in chunks]
            avg_chunk_relevance = sum(chunk_similarities) / len(chunk_similarities)
            confidence_factors.append(avg_chunk_relevance)
        
        # Calculate weighted average
        if confidence_factors:
            return sum(confidence_factors) / len(confidence_factors)
        else:
            return 0.5  # Default medium confidence
    
    def _generate_kg_enhanced_response(self, query: str, chunks: List[Dict[str, Any]], 
                                     kg_context_data: Dict[str, Any], 
                                     reasoning_result: Dict[str, Any]) -> str:
        """Generate final response using all available information."""
        
        try:
            # Build comprehensive context
            context_parts = []
            
            # Add reasoning insights
            if reasoning_result.get("reasoning"):
                context_parts.append("**AI Reasoning Analysis:**")
                context_parts.append(reasoning_result["reasoning"])
                context_parts.append("")
            
            # Add chunk information
            if chunks:
                context_parts.append("**Documentation References:**")
                for i, chunk in enumerate(chunks[:3], 1):  # Top 3 chunks
                    context_parts.append(f"Source {i}: {chunk['title']}")
                    # Add truncated content
                    content = chunk['content']
                    if len(content) > 400:
                        content = content[:400] + "..."
                    context_parts.append(content)
                    context_parts.append("")
            
            # Add knowledge graph context
            if kg_context_data.get("context"):
                context_parts.append(kg_context_data["context"])
                context_parts.append("")
            
            full_context = "\n".join(context_parts)
            
            # Generate response
            response_prompt = f"""Based on the comprehensive analysis below, provide a clear, accurate answer to this wasmCloud question:

{query}

{full_context}

Instructions:
1. Provide a direct, actionable answer
2. Reference specific relationships from the knowledge graph when relevant
3. Include code examples or configuration details if applicable
4. Mention any limitations or areas needing more information
5. Structure the response clearly with headers if the answer has multiple parts

Make the answer practical and focused on what the user needs to know."""

            response = client.chat.completions.create(
                model=self.chat_model,
                messages=[
                    {"role": "system", "content": "You are a wasmCloud expert providing comprehensive, accurate answers based on documentation and relationship analysis."},
                    {"role": "user", "content": response_prompt}
                ],
                temperature=0.2,
                max_tokens=1500
            )
            
            return response.choices[0].message.content
            
        except Exception as e:
            logger.error(f"Error generating KG-enhanced response: {e}")
            # Fallback to basic response
            return self._generate_enhanced_response(query, full_context, chunks)
    
    def _log_kg_query(self, query: str, response: str, processing_time: float, 
                     chunks_used: int, triplets_used: int, db: Session):
        """Log the KG-enhanced query for analytics."""
        
        try:
            query_log = QueryLog(
                query=query,
                response=response,
                response_time=processing_time,
                chunks_used=chunks_used,
                triplets_used=triplets_used,
                search_strategy="kg_enhanced"
            )
            
            db.add(query_log)
            db.commit()
            
        except Exception as e:
            logger.error(f"Error logging KG query: {e}")
            # Rollback the transaction on error
            db.rollback()


# Utility function for external usage
def ask_question_kg_enhanced(query: str, db: Session) -> Dict[str, Any]:
    """
    Main entry point for KG-enhanced question answering.
    
    Args:
        query: User question
        db: Database session
        
    Returns:
        Dictionary with answer and metadata
    """
    
    rag_pipeline = KGEnhancedRAG()
    return rag_pipeline.answer_question_kg_enhanced(query, db)


def get_kg_rag_capabilities(db: Session) -> Dict[str, Any]:
    """Get information about KG-enhanced RAG capabilities."""
    
    from .knowledge_graph import get_knowledge_graph_stats
    
    try:
        kg_stats = get_knowledge_graph_stats(db)
        
        # Calculate coverage
        from .models import Chunk
        total_chunks = db.query(Chunk).count()
        chunks_with_triplets = db.query(Chunk).filter(Chunk.triplets.any()).count()
        
        coverage_percent = (chunks_with_triplets / total_chunks * 100) if total_chunks > 0 else 0
        
        return {
            "kg_enabled": kg_stats.get("triplet_count", 0) > 0,
            "knowledge_graph_stats": kg_stats,
            "coverage": {
                "total_chunks": total_chunks,
                "chunks_with_triplets": chunks_with_triplets,
                "coverage_percent": round(coverage_percent, 1)
            },
            "capabilities": [
                "Relationship-aware reasoning",
                "Multi-source information synthesis", 
                "Concept connection discovery",
                "Enhanced technical accuracy",
                "Structured knowledge analysis"
            ],
            "example_queries": [
                "How do wasmCloud actors interact with capability providers?",
                "What are the dependencies for deploying on Kubernetes?",
                "How does NATS messaging relate to wasmCloud's architecture?",
                "What configuration is needed for OCI registry integration?"
            ]
        }
        
    except Exception as e:
        logger.error(f"Error getting KG RAG capabilities: {e}")
        return {"kg_enabled": False, "error": str(e)} 