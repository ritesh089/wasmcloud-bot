"""Database models for wasmCloud RAG bot."""

from sqlalchemy import Column, Integer, String, Text, DateTime, ForeignKey, Float, JSON
from sqlalchemy.orm import relationship
from sqlalchemy.sql import func
from .database import Base

# Try to import pgvector, fall back to JSON if not available
try:
    from pgvector.sqlalchemy import Vector
    PGVECTOR_AVAILABLE = True
except ImportError:
    PGVECTOR_AVAILABLE = False


class Document(Base):
    """Model for storing scraped documents."""
    
    __tablename__ = "documents"
    
    id = Column(Integer, primary_key=True, index=True)
    url = Column(String, unique=True, index=True, nullable=False)
    title = Column(String, nullable=False)
    content = Column(Text, nullable=False)
    content_hash = Column(String, nullable=False)  # For detecting changes
    scraped_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), onupdate=func.now())
    
    # Relationship to chunks
    chunks = relationship("Chunk", back_populates="document", cascade="all, delete-orphan")
    
    def __repr__(self):
        return f"<Document(id={self.id}, title='{self.title}', url='{self.url}')>"


class Chunk(Base):
    """Model for storing document chunks with embeddings."""
    
    __tablename__ = "chunks"
    
    id = Column(Integer, primary_key=True, index=True)
    document_id = Column(Integer, ForeignKey("documents.id"), nullable=False)
    content = Column(Text, nullable=False)
    chunk_index = Column(Integer, nullable=False)  # Order within document
    token_count = Column(Integer, nullable=False)
    
    # Vector embedding - use pgvector if available, otherwise JSON
    if PGVECTOR_AVAILABLE:
        embedding = Column(Vector(1536), nullable=False)
    else:
        embedding = Column(JSON, nullable=False)  # Store as JSON array
    
    # Metadata
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    
    # Relationship to document
    document = relationship("Document", back_populates="chunks")
    
    def __repr__(self):
        return f"<Chunk(id={self.id}, document_id={self.document_id}, chunk_index={self.chunk_index})>"


class QueryLog(Base):
    """Model for logging user queries and responses."""
    
    __tablename__ = "query_logs"
    
    id = Column(Integer, primary_key=True, index=True)
    query = Column(Text, nullable=False)
    response = Column(Text, nullable=False)
    response_time = Column(Float, nullable=False)  # Time in seconds
    chunks_used = Column(Integer, nullable=False)  # Number of chunks retrieved
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    
    def __repr__(self):
        return f"<QueryLog(id={self.id}, query='{self.query[:50]}...', response_time={self.response_time})>" 