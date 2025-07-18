"""Main FastAPI server for wasmCloud RAG bot."""

import os
import logging
from contextlib import contextmanager
from typing import Dict, Any, List
from fastapi import FastAPI, HTTPException, Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from sqlalchemy.orm import Session
from sqlalchemy import text
from dotenv import load_dotenv

from .database import get_db, init_database, check_database_connection, get_database_stats
from .rag import ask_question
from .advanced_rag import ask_question_advanced
from .kg_enhanced_rag import ask_question_kg_enhanced, get_kg_rag_capabilities
from .knowledge_graph import (
    extract_triplets_from_all_chunks, 
    get_knowledge_graph_stats,
    KnowledgeGraphExtractor
)
from .scraper import scrape_wasmcloud_docs
from .models import Document, Chunk
from .embeddings import chunk_document, generate_chunk_embedding
from .ai_chunking import create_intelligent_chunks
from .optimization_config import get_optimization_config

load_dotenv()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Configuration
SERVER_PORT = int(os.getenv("PORT", "8000"))


# Pydantic models for API
class QueryRequest(BaseModel):
    question: str
    include_sources: bool = True


class QueryResponse(BaseModel):
    answer: str
    sources: List[Dict[str, Any]]
    chunks_used: int
    response_time: float


class HealthResponse(BaseModel):
    status: str
    database_connected: bool
    message: str


class IngestResponse(BaseModel):
    status: str
    documents_processed: int
    chunks_created: int
    message: str


class StatsResponse(BaseModel):
    total_documents: int
    total_chunks: int
    database_stats: List[Dict[str, Any]]


# Create FastAPI app
app = FastAPI(
    title="wasmCloud RAG Bot",
    description="A RAG-based bot for answering questions about wasmCloud using official documentation",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure appropriately for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount static files
app.mount("/static", StaticFiles(directory="static"), name="static")

# Startup event
@app.on_event("startup")
def startup_event():
    """Initialize the application on startup."""
    logger.info("Starting wasmCloud RAG Bot server...")
    
    try:
        init_database()
        logger.info("Database initialized successfully")
    except Exception as e:
        logger.error(f"Failed to initialize database: {e}")
        raise


from fastapi.responses import FileResponse

@app.get("/")
def root():
    """Serve the web interface."""
    return FileResponse('static/index.html')

@app.get("/api", response_model=Dict[str, str])
def api_info():
    """API endpoint with basic information."""
    return {
        "message": "wasmCloud RAG Bot API",
        "version": "1.0.0",
        "description": "Ask questions about wasmCloud and get answers from official documentation"
    }


@app.get("/health", response_model=HealthResponse)
def health_check():
    """Health check endpoint."""
    try:
        db_connected = check_database_connection()
        
        if db_connected:
            return HealthResponse(
                status="healthy",
                database_connected=True,
                message="All systems operational"
            )
        else:
            return HealthResponse(
                status="unhealthy",
                database_connected=False,
                message="Database connection failed"
            )
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        return HealthResponse(
            status="unhealthy",
            database_connected=False,
            message=f"Health check error: {str(e)}"
        )


@app.post("/query", response_model=QueryResponse)
def query_bot(
    request: QueryRequest,
    db: Session = Depends(get_db)
):
    """
    Ask a question to the wasmCloud RAG bot.
    
    Args:
        request: Query request with question and options
        db: Database session
        
    Returns:
        Answer with sources and metadata
    """
    try:
        if not request.question.strip():
            raise HTTPException(status_code=400, detail="Question cannot be empty")
        
        logger.info(f"Processing query: {request.question[:100]}...")
        
        # Use optimized RAG if enabled
        config = get_optimization_config()
        
        if config.get_rag_strategy() == "kg_enhanced":
            result = ask_question_kg_enhanced(request.question, db)
        elif config.get_rag_strategy() == "advanced":
            result = ask_question_advanced(request.question, db)
        else:
            result = ask_question(request.question, db)
        
        return QueryResponse(
            answer=result['answer'],
            sources=result['sources'] if request.include_sources else [],
            chunks_used=result['chunks_used'],
            response_time=result.get('response_time', result.get('processing_time', 0))
        )
        
    except Exception as e:
        logger.error(f"Error processing query: {e}")
        raise HTTPException(status_code=500, detail=f"Error processing query: {str(e)}")


@app.post("/query/advanced", response_model=QueryResponse)
def query_bot_advanced(
    request: QueryRequest,
    db: Session = Depends(get_db)
):
    """
    Ask a question using advanced AI-enhanced RAG with hybrid search and reranking.
    """
    try:
        if not request.question.strip():
            raise HTTPException(status_code=400, detail="Question cannot be empty")
        
        logger.info(f"Processing advanced query: {request.question[:100]}...")
        
        result = ask_question_advanced(request.question, db)
        
        return QueryResponse(
            answer=result['answer'],
            sources=result['sources'] if request.include_sources else [],
            chunks_used=result['chunks_used'],
            response_time=result['response_time']
        )
        
    except Exception as e:
        logger.error(f"Error processing advanced query: {e}")
        raise HTTPException(status_code=500, detail=f"Error processing advanced query: {str(e)}")


@app.post("/query/kg", response_model=QueryResponse)
def query_bot_kg_enhanced(
    request: QueryRequest,
    db: Session = Depends(get_db)
):
    """
    Ask a question using Knowledge Graph Enhanced RAG.
    
    This combines traditional vector search with knowledge graph reasoning for more
    comprehensive and accurate answers with relationship-aware context.
    """
    try:
        if not request.question.strip():
            raise HTTPException(status_code=400, detail="Question cannot be empty")
        
        logger.info(f"Processing KG-enhanced query: {request.question[:100]}...")
        
        result = ask_question_kg_enhanced(request.question, db)
        
        # Format response for QueryResponse model
        return QueryResponse(
            answer=result['answer'],
            sources=result.get('sources', []) if request.include_sources else [],
            chunks_used=result['chunks_used'],
            response_time=result['processing_time']
        )
        
    except Exception as e:
        logger.error(f"Error processing KG-enhanced query: {e}")
        raise HTTPException(status_code=500, detail=f"Error processing KG-enhanced query: {str(e)}")


@app.post("/kg/extract")
def extract_knowledge_graph(
    batch_size: int = 50,
    db: Session = Depends(get_db)
):
    """
    Extract knowledge graph triplets from all document chunks using GPT-4.
    
    This is a batch operation that processes chunks without existing triplets
    and extracts structured knowledge relationships.
    """
    try:
        logger.info("Starting knowledge graph extraction...")
        
        result = extract_triplets_from_all_chunks(db, batch_size=batch_size)
        
        return {
            "status": "success",
            "chunks_processed": result["chunks_processed"],
            "triplets_extracted": result["triplets_extracted"],
            "message": f"Extracted {result['triplets_extracted']} triplets from {result['chunks_processed']} chunks"
        }
        
    except Exception as e:
        logger.error(f"Error extracting knowledge graph: {e}")
        raise HTTPException(status_code=500, detail=f"Knowledge graph extraction failed: {str(e)}")


@app.get("/kg/stats")
def get_knowledge_graph_stats(db: Session = Depends(get_db)):
    """Get knowledge graph statistics and capabilities."""
    try:
        kg_stats = get_knowledge_graph_stats(db)
        kg_capabilities = get_kg_rag_capabilities(db)
        
        return {
            "knowledge_graph": kg_stats,
            "rag_capabilities": kg_capabilities,
            "endpoints": {
                "kg_query": "/query/kg",
                "extract_kg": "/kg/extract",
                "kg_stats": "/kg/stats"
            }
        }
        
    except Exception as e:
        logger.error(f"Error getting KG stats: {e}")
        raise HTTPException(status_code=500, detail=f"Error getting KG stats: {str(e)}")


@app.post("/ingest", response_model=IngestResponse)
def ingest_documentation(db: Session = Depends(get_db)):
    """
    Manually trigger documentation ingestion.
    
    This endpoint scrapes wasmCloud documentation, processes it into chunks,
    generates embeddings, and stores everything in the database.
    """
    try:
        logger.info("Starting manual documentation ingestion...")
        
        # Scrape documentation
        documents = scrape_wasmcloud_docs()
        
        if not documents:
            return IngestResponse(
                status="error",
                documents_processed=0,
                chunks_created=0,
                message="No documents were scraped"
            )
        
        documents_processed = 0
        chunks_created = 0
        
        for doc_data in documents:
            try:
                logger.info(f"Processing: {doc_data['title']}")
                
                # Check if document already exists
                result = db.execute(
                    text("SELECT id, content_hash FROM documents WHERE url = :url"),
                    {"url": doc_data['url']}
                )
                existing = result.fetchone()
                
                # Skip if document hasn't changed
                if existing and existing[1] == doc_data['content_hash']:
                    logger.info(f"Skipping unchanged document: {doc_data['url']}")
                    continue
                
                # Delete existing document and its chunks if it exists
                if existing:
                    db.execute(
                        text("DELETE FROM documents WHERE id = :id"),
                        {"id": existing[0]}
                    )
                    logger.info(f"Updated existing document: {doc_data['url']}")
                
                # Create new document
                document = Document(
                    url=doc_data['url'],
                    title=doc_data['title'],
                    content=doc_data['content'],
                    content_hash=doc_data['content_hash']
                )
                db.add(document)
                db.flush()  # Get the document ID
                
                # Chunk the document using intelligent chunking if enabled
                config = get_optimization_config()
                if config.get_chunking_strategy() == "hybrid":
                    chunks_data = create_intelligent_chunks(
                        doc_data['content'],
                        doc_data['title'],
                        doc_data['url']
                    )
                else:
                    chunks_data = chunk_document(
                        doc_data['content'],
                        doc_data['title'],
                        doc_data['url']
                    )
                
                # Process chunks and generate embeddings
                for chunk_data in chunks_data:
                    # Generate embedding
                    embedding = generate_chunk_embedding(chunk_data['content'])
                    
                    # Create chunk
                    chunk = Chunk(
                        document_id=document.id,
                        content=chunk_data['content'],
                        chunk_index=chunk_data['chunk_index'],
                        token_count=chunk_data['token_count'],
                        embedding=embedding
                    )
                    db.add(chunk)
                    chunks_created += 1
                
                documents_processed += 1
                
                # Commit after each document
                db.commit()
                logger.info(f"Successfully processed: {doc_data['title']}")
                
            except Exception as e:
                logger.error(f"Error processing document {doc_data.get('url', 'unknown')}: {e}")
                db.rollback()
                continue
        
        return IngestResponse(
            status="success",
            documents_processed=documents_processed,
            chunks_created=chunks_created,
            message=f"Successfully processed {documents_processed} documents and created {chunks_created} chunks"
        )
        
    except Exception as e:
        logger.error(f"Error during documentation ingestion: {e}")
        raise HTTPException(status_code=500, detail=f"Ingestion failed: {str(e)}")


@app.get("/stats", response_model=StatsResponse)
def get_stats(db: Session = Depends(get_db)):
    """Get database statistics."""
    try:
        # Get document count
        doc_count_result = db.execute(text("SELECT COUNT(*) FROM documents"))
        doc_count = doc_count_result.scalar()
        
        # Get chunk count
        chunk_count_result = db.execute(text("SELECT COUNT(*) FROM chunks"))
        chunk_count = chunk_count_result.scalar()
        
        # Get detailed database stats
        db_stats = get_database_stats()
        
        return StatsResponse(
            total_documents=doc_count,
            total_chunks=chunk_count,
            database_stats=db_stats
        )
        
    except Exception as e:
        logger.error(f"Error getting stats: {e}")
        raise HTTPException(status_code=500, detail=f"Error getting stats: {str(e)}")


@app.get("/documents", response_model=List[Dict[str, Any]])
def list_documents(
    limit: int = 10,
    offset: int = 0,
    db: Session = Depends(get_db)
):
    """List documents in the database."""
    try:
        result = db.execute(
            text("""
                SELECT id, url, title, scraped_at, 
                       (SELECT COUNT(*) FROM chunks WHERE document_id = documents.id) as chunk_count
                FROM documents 
                ORDER BY scraped_at DESC 
                LIMIT :limit OFFSET :offset
            """),
            {"limit": limit, "offset": offset}
        )
        
        documents = []
        for row in result:
            documents.append({
                'id': row[0],
                'url': row[1],
                'title': row[2],
                'scraped_at': row[3].isoformat() if row[3] else None,
                'chunk_count': row[4]
            })
        
        return documents
        
    except Exception as e:
        logger.error(f"Error listing documents: {e}")
        raise HTTPException(status_code=500, detail=f"Error listing documents: {str(e)}")


@app.get("/optimization/config")
def get_optimization_status():
    """Get current optimization configuration and status."""
    try:
        config = get_optimization_config()
        return {
            "status": "success",
            "configuration": config.to_dict(),
            "description": {
                "ai_chunking": "Uses AI to create semantically coherent chunks instead of simple token-based splitting",
                "hybrid_search": "Combines vector similarity, keyword search, and concept search with AI reranking",
                "kg_enhanced": "Knowledge Graph Enhanced RAG that uses extracted triplets for relationship-aware reasoning",
                "keyword_search": "Traditional full-text search using PostgreSQL's built-in capabilities", 
                "concept_search": "AI-powered expansion of queries to find related concepts",
                "ai_reranking": "Uses AI to reorder search results based on relevance to specific queries",
                "kg_reasoning": "AI reasoning that combines text chunks with knowledge graph relationships"
            }
        }
    except Exception as e:
        logger.error(f"Error getting optimization config: {e}")
        raise HTTPException(status_code=500, detail=f"Error getting optimization config: {str(e)}")


@app.post("/optimization/enable/{feature}")
def enable_optimization_feature(feature: str):
    """Enable a specific optimization feature."""
    try:
        from .optimization_config import update_config
        
        valid_features = {
            "ai_chunking": "enable_ai_chunking",
            "hybrid_search": "enable_hybrid_search", 
            "kg_enhanced": "enable_kg_enhanced",
            "keyword_search": "enable_keyword_search",
            "concept_search": "enable_concept_search",
            "ai_reranking": "enable_ai_reranking",
            "kg_reasoning": "kg_enable_reasoning"
        }
        
        if feature not in valid_features:
            raise HTTPException(status_code=400, detail=f"Invalid feature. Valid options: {list(valid_features.keys())}")
        
        config_key = valid_features[feature]
        update_config(**{config_key: True})
        
        return {
            "status": "success",
            "message": f"Enabled {feature} optimization",
            "feature": feature,
            "enabled": True
        }
        
    except Exception as e:
        logger.error(f"Error enabling optimization feature {feature}: {e}")
        raise HTTPException(status_code=500, detail=f"Error enabling feature: {str(e)}")


@app.post("/optimization/disable/{feature}")
def disable_optimization_feature(feature: str):
    """Disable a specific optimization feature."""
    try:
        from .optimization_config import update_config
        
        valid_features = {
            "ai_chunking": "enable_ai_chunking",
            "hybrid_search": "enable_hybrid_search",
            "kg_enhanced": "enable_kg_enhanced",
            "keyword_search": "enable_keyword_search", 
            "concept_search": "enable_concept_search",
            "ai_reranking": "enable_ai_reranking",
            "kg_reasoning": "kg_enable_reasoning"
        }
        
        if feature not in valid_features:
            raise HTTPException(status_code=400, detail=f"Invalid feature. Valid options: {list(valid_features.keys())}")
        
        config_key = valid_features[feature]
        update_config(**{config_key: False})
        
        return {
            "status": "success",
            "message": f"Disabled {feature} optimization",
            "feature": feature,
            "enabled": False
        }
        
    except Exception as e:
        logger.error(f"Error disabling optimization feature {feature}: {e}")
        raise HTTPException(status_code=500, detail=f"Error disabling feature: {str(e)}")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=SERVER_PORT) 