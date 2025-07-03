"""AI-enhanced chunking system for better semantic understanding."""

import os
import logging
from typing import List, Dict, Any, Optional
from openai import OpenAI
import tiktoken
from dotenv import load_dotenv

load_dotenv()

logger = logging.getLogger(__name__)

# Configuration
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
client = OpenAI(api_key=OPENAI_API_KEY)


class AIChunker:
    """AI-enhanced text chunking for better semantic coherence."""
    
    def __init__(self, model: str = "gpt-3.5-turbo", max_chunk_size: int = 1200):
        self.model = model
        self.max_chunk_size = max_chunk_size
        self.encoding = tiktoken.get_encoding("cl100k_base")
    
    def analyze_document_structure(self, content: str, title: str) -> Dict[str, Any]:
        """Use AI to analyze document structure and identify optimal chunk boundaries."""
        
        # Count tokens to decide strategy
        token_count = len(self.encoding.encode(content))
        
        if token_count < 500:
            # For small documents, return as single chunk
            return {
                "strategy": "single_chunk",
                "sections": [{"start": 0, "end": len(content), "topic": title}],
                "reasoning": "Document too small to benefit from chunking"
            }
        
        prompt = f"""Analyze this wasmCloud documentation and identify the best way to split it into coherent sections for a RAG system.

Document Title: {title}

Content:
{content[:3000]}{'...' if token_count > 3000 else ''}

Please identify 3-8 main sections that:
1. Cover distinct topics or concepts
2. Are self-contained and coherent
3. Would be useful for answering specific questions
4. Maintain important context

Return a JSON structure with:
- "sections": List of sections with "topic", "start_phrase", "end_phrase", and "rationale"
- "strategy": Brief description of the chunking approach
- "key_concepts": Main concepts covered in this document

Focus on wasmCloud concepts like: actors, capabilities, providers, WASM modules, deployment, configuration, etc."""

        try:
            response = client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                response_format={"type": "json_object"},
                temperature=0.1
            )
            
            analysis = response.choices[0].message.content
            import json
            return json.loads(analysis)
            
        except Exception as e:
            logger.error(f"AI analysis failed: {e}")
            # Fallback to simple analysis
            return self._fallback_analysis(content, title)
    
    def _fallback_analysis(self, content: str, title: str) -> Dict[str, Any]:
        """Fallback analysis when AI fails."""
        return {
            "strategy": "paragraph_based",
            "sections": self._simple_paragraph_split(content),
            "key_concepts": [title]
        }
    
    def _simple_paragraph_split(self, content: str) -> List[Dict[str, str]]:
        """Simple fallback paragraph-based splitting."""
        paragraphs = content.split('\n\n')
        sections = []
        
        for i, para in enumerate(paragraphs):
            if len(para.strip()) > 100:  # Skip very short paragraphs
                sections.append({
                    "topic": f"Section {i+1}",
                    "start_phrase": para[:50],
                    "end_phrase": para[-50:],
                    "rationale": "Paragraph-based split"
                })
        
        return sections
    
    def create_semantic_chunks(self, content: str, title: str, url: str) -> List[Dict[str, Any]]:
        """Create semantically coherent chunks using AI analysis."""
        
        # Get AI analysis of document structure
        analysis = self.analyze_document_structure(content, title)
        
        chunks = []
        
        if analysis["strategy"] == "single_chunk":
            # Single chunk for small documents
            chunks.append({
                'content': content,
                'token_count': len(self.encoding.encode(content)),
                'chunk_index': 0,
                'title': title,
                'url': url,
                'topic': title,
                'chunk_type': 'complete_document',
                'key_concepts': analysis.get('key_concepts', [title])
            })
        else:
            # Create chunks based on AI-identified sections
            sections = analysis.get('sections', [])
            
            for i, section in enumerate(sections):
                chunk_content = self._extract_section_content(
                    content, 
                    section.get('start_phrase', ''),
                    section.get('end_phrase', ''),
                    i,
                    len(sections)
                )
                
                if chunk_content and len(chunk_content.strip()) > 50:
                    chunks.append({
                        'content': chunk_content,
                        'token_count': len(self.encoding.encode(chunk_content)),
                        'chunk_index': i,
                        'title': title,
                        'url': url,
                        'topic': section.get('topic', f'Section {i+1}'),
                        'chunk_type': 'semantic_section',
                        'rationale': section.get('rationale', ''),
                        'key_concepts': analysis.get('key_concepts', [])
                    })
        
        # Ensure chunks aren't too large
        final_chunks = []
        for chunk in chunks:
            if chunk['token_count'] > self.max_chunk_size:
                # Split large chunks
                sub_chunks = self._split_large_chunk(chunk)
                final_chunks.extend(sub_chunks)
            else:
                final_chunks.append(chunk)
        
        logger.info(f"Created {len(final_chunks)} semantic chunks for '{title}'")
        return final_chunks
    
    def _extract_section_content(self, content: str, start_phrase: str, end_phrase: str, 
                                section_idx: int, total_sections: int) -> str:
        """Extract content for a specific section."""
        
        if not start_phrase:
            # For first section or when no specific phrase
            if section_idx == 0:
                end_pos = content.find(end_phrase) if end_phrase else len(content) // total_sections
                return content[:max(end_pos, len(content) // total_sections)]
            else:
                # Estimate based on section index
                section_size = len(content) // total_sections
                start_pos = section_idx * section_size
                end_pos = (section_idx + 1) * section_size
                return content[start_pos:end_pos]
        
        start_pos = content.find(start_phrase)
        if start_pos == -1:
            start_pos = 0
        
        if end_phrase:
            end_pos = content.find(end_phrase, start_pos + len(start_phrase))
            if end_pos == -1:
                end_pos = len(content)
            else:
                end_pos += len(end_phrase)
        else:
            end_pos = len(content)
        
        return content[start_pos:end_pos]
    
    def _split_large_chunk(self, chunk: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Split chunks that are too large."""
        content = chunk['content']
        max_size = self.max_chunk_size - 100  # Leave room for overlap
        
        if len(self.encoding.encode(content)) <= max_size:
            return [chunk]
        
        # Split by paragraphs first
        paragraphs = content.split('\n\n')
        
        chunks = []
        current_chunk = ""
        current_tokens = 0
        sub_idx = 0
        
        for para in paragraphs:
            para_tokens = len(self.encoding.encode(para))
            
            if current_tokens + para_tokens > max_size and current_chunk:
                # Finalize current chunk
                chunk_copy = chunk.copy()
                chunk_copy.update({
                    'content': current_chunk.strip(),
                    'token_count': current_tokens,
                    'chunk_index': f"{chunk['chunk_index']}.{sub_idx}",
                    'chunk_type': 'split_section'
                })
                chunks.append(chunk_copy)
                
                # Start new chunk
                current_chunk = para
                current_tokens = para_tokens
                sub_idx += 1
            else:
                current_chunk += "\n\n" + para if current_chunk else para
                current_tokens += para_tokens
        
        # Add final chunk
        if current_chunk.strip():
            chunk_copy = chunk.copy()
            chunk_copy.update({
                'content': current_chunk.strip(),
                'token_count': current_tokens,
                'chunk_index': f"{chunk['chunk_index']}.{sub_idx}",
                'chunk_type': 'split_section'
            })
            chunks.append(chunk_copy)
        
        return chunks


class HybridChunker:
    """Combines AI chunking with traditional methods for optimal performance."""
    
    def __init__(self, use_ai_for_large_docs: bool = True, ai_threshold: int = 1000):
        self.ai_chunker = AIChunker()
        self.ai_threshold = ai_threshold  # Token threshold for using AI
        self.use_ai_for_large_docs = use_ai_for_large_docs
        
        # Import traditional chunker
        from .embeddings import TextChunker
        self.traditional_chunker = TextChunker()
    
    def chunk_document(self, content: str, title: str, url: str) -> List[Dict[str, Any]]:
        """Choose optimal chunking strategy based on document characteristics."""
        
        token_count = len(self.ai_chunker.encoding.encode(content))
        
        # Use AI chunking for larger, more complex documents
        if self.use_ai_for_large_docs and token_count > self.ai_threshold:
            try:
                return self.ai_chunker.create_semantic_chunks(content, title, url)
            except Exception as e:
                logger.warning(f"AI chunking failed for '{title}': {e}, falling back to traditional")
                return self._traditional_chunks_with_metadata(content, title, url)
        else:
            # Use traditional chunking for smaller documents
            return self._traditional_chunks_with_metadata(content, title, url)
    
    def _traditional_chunks_with_metadata(self, content: str, title: str, url: str) -> List[Dict[str, Any]]:
        """Create traditional chunks with enhanced metadata."""
        chunks = self.traditional_chunker.chunk_text(content, {'title': title, 'url': url})
        
        # Enhance with additional metadata
        for chunk in chunks:
            chunk.update({
                'chunk_type': 'traditional',
                'topic': title,
                'key_concepts': [title]
            })
        
        return chunks


# Global instance
hybrid_chunker = HybridChunker()


def create_intelligent_chunks(content: str, title: str, url: str) -> List[Dict[str, Any]]:
    """Main function for creating intelligent chunks."""
    return hybrid_chunker.chunk_document(content, title, url) 