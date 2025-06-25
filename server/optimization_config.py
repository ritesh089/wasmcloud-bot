"""Optimization configuration for wasmCloud RAG Bot."""

import os
from typing import Dict, Any
from dataclasses import dataclass
from dotenv import load_dotenv

load_dotenv()


@dataclass
class OptimizationConfig:
    """Configuration for optimization features."""
    
    # AI Chunking
    enable_ai_chunking: bool = True
    ai_chunking_threshold: int = 1000  # Token threshold for using AI chunking
    ai_chunking_model: str = "gpt-3.5-turbo"
    
    # Advanced RAG
    enable_hybrid_search: bool = False  # Start with False for gradual rollout
    enable_keyword_search: bool = True
    enable_concept_search: bool = True
    enable_ai_reranking: bool = True
    
    # Search Parameters
    vector_search_k: int = 10  # Initial retrieval count
    final_results_k: int = 5   # Final results after reranking
    similarity_threshold: float = 0.6
    
    # Performance Settings
    embedding_batch_size: int = 10
    cache_embeddings: bool = True
    
    # Smart Document Processing
    enable_smart_scraping: bool = False
    quality_threshold: float = 0.7
    
    # Cost Management
    max_ai_calls_per_query: int = 3
    use_cheaper_models_for_analysis: bool = True
    
    @classmethod
    def from_env(cls) -> 'OptimizationConfig':
        """Create configuration from environment variables."""
        return cls(
            enable_ai_chunking=os.getenv('ENABLE_AI_CHUNKING', 'true').lower() == 'true',
            ai_chunking_threshold=int(os.getenv('AI_CHUNKING_THRESHOLD', '1000')),
            ai_chunking_model=os.getenv('AI_CHUNKING_MODEL', 'gpt-3.5-turbo'),
            
            enable_hybrid_search=os.getenv('ENABLE_HYBRID_SEARCH', 'false').lower() == 'true',
            enable_keyword_search=os.getenv('ENABLE_KEYWORD_SEARCH', 'true').lower() == 'true',
            enable_concept_search=os.getenv('ENABLE_CONCEPT_SEARCH', 'true').lower() == 'true',
            enable_ai_reranking=os.getenv('ENABLE_AI_RERANKING', 'true').lower() == 'true',
            
            vector_search_k=int(os.getenv('VECTOR_SEARCH_K', '10')),
            final_results_k=int(os.getenv('FINAL_RESULTS_K', '5')),
            similarity_threshold=float(os.getenv('SIMILARITY_THRESHOLD', '0.6')),
            
            embedding_batch_size=int(os.getenv('EMBEDDING_BATCH_SIZE', '10')),
            cache_embeddings=os.getenv('CACHE_EMBEDDINGS', 'true').lower() == 'true',
            
            enable_smart_scraping=os.getenv('ENABLE_SMART_SCRAPING', 'false').lower() == 'true',
            quality_threshold=float(os.getenv('QUALITY_THRESHOLD', '0.7')),
            
            max_ai_calls_per_query=int(os.getenv('MAX_AI_CALLS_PER_QUERY', '3')),
            use_cheaper_models_for_analysis=os.getenv('USE_CHEAPER_MODELS', 'true').lower() == 'true'
        )
    
    def get_chunking_strategy(self) -> str:
        """Get the chunking strategy to use."""
        if self.enable_ai_chunking:
            return "hybrid"  # Use HybridChunker
        else:
            return "traditional"  # Use TextChunker
    
    def get_rag_strategy(self) -> str:
        """Get the RAG strategy to use."""
        if self.enable_hybrid_search:
            return "advanced"  # Use HybridSearchRAG
        else:
            return "basic"  # Use basic RAGPipeline
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for API responses."""
        return {
            'chunking_strategy': self.get_chunking_strategy(),
            'rag_strategy': self.get_rag_strategy(),
            'ai_chunking_enabled': self.enable_ai_chunking,
            'hybrid_search_enabled': self.enable_hybrid_search,
            'keyword_search_enabled': self.enable_keyword_search,
            'concept_search_enabled': self.enable_concept_search,
            'ai_reranking_enabled': self.enable_ai_reranking,
            'vector_search_k': self.vector_search_k,
            'final_results_k': self.final_results_k,
            'similarity_threshold': self.similarity_threshold
        }


# Global configuration instance
config = OptimizationConfig.from_env()


def get_optimization_config() -> OptimizationConfig:
    """Get the current optimization configuration."""
    return config


def update_config(**kwargs) -> None:
    """Update configuration with new values."""
    global config
    for key, value in kwargs.items():
        if hasattr(config, key):
            setattr(config, key, value) 