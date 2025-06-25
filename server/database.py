"""Database connection and configuration for wasmCloud RAG bot."""

import os
from typing import Generator
from sqlalchemy import create_engine, text, event
from sqlalchemy.orm import declarative_base, sessionmaker, Session
from dotenv import load_dotenv
import logging

load_dotenv()

logger = logging.getLogger(__name__)

# Database configuration
DATABASE_URL = os.getenv("DATABASE_URL", "postgresql://username:password@localhost:5432/wasmcloud_rag")

# Create engine
engine = create_engine(DATABASE_URL, echo=False)

# Session maker
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

# Base class for models
Base = declarative_base()


def get_db() -> Generator[Session, None, None]:
    """Get synchronous database session."""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


def init_database():
    """Initialize database with pgvector extension and create tables."""
    try:
        with engine.begin() as conn:
            # Enable pgvector extension
            conn.execute(text("CREATE EXTENSION IF NOT EXISTS vector"))
            logger.info("pgvector extension enabled")
            
            # Create all tables
            Base.metadata.create_all(bind=engine)
            logger.info("Database tables created successfully")
            
    except Exception as e:
        logger.error(f"Error initializing database: {e}")
        raise


def check_database_connection():
    """Check if database connection is working."""
    try:
        with engine.connect() as conn:
            conn.execute(text("SELECT 1"))
        return True
    except Exception as e:
        logger.error(f"Database connection failed: {e}")
        return False


def get_database_stats():
    """Get database statistics."""
    try:
        with engine.connect() as conn:
            result = conn.execute(text("""
                SELECT 
                    schemaname,
                    relname as tablename,
                    n_tup_ins as inserts,
                    n_tup_upd as updates,
                    n_tup_del as deletes,
                    n_live_tup as live_tuples
                FROM pg_stat_user_tables
                WHERE relname IN ('documents', 'chunks');
            """))
            return [dict(row._mapping) for row in result]
    except Exception as e:
        logger.error(f"Error getting database stats: {e}")
        return []


# Register pgvector types
@event.listens_for(engine, "connect")
def register_vector(dbapi_connection, connection_record):
    """Register pgvector types with psycopg2."""
    try:
        # Try to register vector type if pgvector is available
        with dbapi_connection.cursor() as cursor:
            cursor.execute("SELECT NULL::vector")
    except Exception:
        # pgvector not available, that's okay for now
        pass 