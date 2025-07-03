"""Database models for wasmCloud RAG bot."""

from sqlalchemy import Column, Integer, String, Text, DateTime, ForeignKey, Float, JSON, UniqueConstraint, Index
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
    # Relationship to triplets
    triplets = relationship("Triplet", back_populates="source_chunk", cascade="all, delete-orphan")
    
    def __repr__(self):
        return f"<Chunk(id={self.id}, document_id={self.document_id}, chunk_index={self.chunk_index})>"


class Entity(Base):
    """Model for storing knowledge graph entities."""
    
    __tablename__ = "entities"
    
    id = Column(Integer, primary_key=True, index=True)
    name = Column(String, nullable=False, index=True)
    entity_type = Column(String, nullable=False, index=True)  # e.g., 'concept', 'technology', 'person', 'organization'
    description = Column(Text)  # Optional description extracted from context
    canonical_form = Column(String, nullable=False, index=True)  # Normalized form for deduplication
    frequency = Column(Integer, default=1)  # How often this entity appears
    
    # Vector embedding for entity
    if PGVECTOR_AVAILABLE:
        embedding = Column(Vector(1536))
    else:
        embedding = Column(JSON)  # Store as JSON array
    
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), onupdate=func.now())
    
    # Relationships
    subject_triplets = relationship("Triplet", foreign_keys="Triplet.subject_id", back_populates="subject")
    object_triplets = relationship("Triplet", foreign_keys="Triplet.object_id", back_populates="object")
    
    # Unique constraint on canonical form and type
    __table_args__ = (
        UniqueConstraint('canonical_form', 'entity_type', name='unique_entity'),
        Index('idx_entity_name_type', 'name', 'entity_type'),
    )
    
    def __repr__(self):
        return f"<Entity(id={self.id}, name='{self.name}', type='{self.entity_type}')>"


class Relation(Base):
    """Model for storing knowledge graph relations."""
    
    __tablename__ = "relations"
    
    id = Column(Integer, primary_key=True, index=True)
    name = Column(String, nullable=False, unique=True, index=True)
    description = Column(Text)  # Optional description of the relation
    relation_type = Column(String, nullable=False, index=True)  # e.g., 'functional', 'hierarchical', 'temporal'
    frequency = Column(Integer, default=1)  # How often this relation appears
    
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    
    # Relationships
    triplets = relationship("Triplet", back_populates="relation")
    
    def __repr__(self):
        return f"<Relation(id={self.id}, name='{self.name}', type='{self.relation_type}')>"


class Triplet(Base):
    """Model for storing knowledge graph triplets (subject-predicate-object)."""
    
    __tablename__ = "triplets"
    
    id = Column(Integer, primary_key=True, index=True)
    subject_id = Column(Integer, ForeignKey("entities.id"), nullable=False)
    relation_id = Column(Integer, ForeignKey("relations.id"), nullable=False)
    object_id = Column(Integer, ForeignKey("entities.id"), nullable=False)
    
    # Source information
    source_chunk_id = Column(Integer, ForeignKey("chunks.id"), nullable=False)
    confidence_score = Column(Float, default=0.8)  # Confidence in the extraction
    
    # Context and metadata
    context_sentence = Column(Text)  # The sentence where this triplet was found
    extracted_text = Column(Text)  # Original text that led to this triplet
    
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    
    # Relationships
    subject = relationship("Entity", foreign_keys=[subject_id], back_populates="subject_triplets")
    relation = relationship("Relation", back_populates="triplets")
    object = relationship("Entity", foreign_keys=[object_id], back_populates="object_triplets")
    source_chunk = relationship("Chunk", back_populates="triplets")
    
    # Constraints and indexes
    __table_args__ = (
        UniqueConstraint('subject_id', 'relation_id', 'object_id', 'source_chunk_id', name='unique_triplet'),
        Index('idx_triplet_subject', 'subject_id'),
        Index('idx_triplet_object', 'object_id'),
        Index('idx_triplet_relation', 'relation_id'),
        Index('idx_triplet_source', 'source_chunk_id'),
    )
    
    def __repr__(self):
        return f"<Triplet(id={self.id}, subject_id={self.subject_id}, relation_id={self.relation_id}, object_id={self.object_id})>"


class QueryLog(Base):
    """Model for logging user queries and responses."""
    
    __tablename__ = "query_logs"
    
    id = Column(Integer, primary_key=True, index=True)
    query = Column(Text, nullable=False)
    response = Column(Text, nullable=False)
    response_time = Column(Float, nullable=False)  # Time in seconds
    chunks_used = Column(Integer, nullable=False)  # Number of chunks retrieved
    triplets_used = Column(Integer, default=0)  # Number of triplets used in KG reasoning
    search_strategy = Column(String, default='basic')  # 'basic', 'advanced', 'kg_enhanced'
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    
    def __repr__(self):
        return f"<QueryLog(id={self.id}, query='{self.query[:50]}...', response_time={self.response_time})>" 