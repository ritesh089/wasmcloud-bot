#!/usr/bin/env python3
"""Documentation ingestion script for wasmCloud RAG bot."""

import sys
import os
import logging
from sqlalchemy import text

# Add the parent directory to the path so we can import server modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from server.database import get_db, check_database_connection, SessionLocal
from server.scraper import scrape_wasmcloud_docs
from server.models import Document, Chunk
from server.embeddings import chunk_document, generate_chunk_embedding
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def ingest_documents():
    """Ingest wasmCloud documentation into the database."""
    try:
        logger.info("Starting documentation ingestion...")
        
        # Check database connection
        if not check_database_connection():
            logger.error("Cannot connect to database. Please run init_db.py first.")
            return False
        
        # Scrape documentation
        logger.info("Scraping wasmCloud documentation...")
        documents = scrape_wasmcloud_docs()
        
        if not documents:
            logger.warning("No documents were scraped. Please check the scraper configuration.")
            return False
        
        logger.info(f"Scraped {len(documents)} documents")
        
        # Process documents
        db = SessionLocal()
        try:
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
                    
                    # Chunk the document
                    chunks_data = chunk_document(
                        doc_data['content'],
                        doc_data['title'],
                        doc_data['url']
                    )
                    
                    logger.info(f"Created {len(chunks_data)} chunks for document")
                    
                    # Process chunks and generate embeddings
                    for i, chunk_data in enumerate(chunks_data):
                        logger.info(f"Processing chunk {i+1}/{len(chunks_data)}")
                        
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
                    
                    # Commit after each document to avoid memory issues
                    db.commit()
                    logger.info(f"Successfully processed: {doc_data['title']}")
                    
                except Exception as e:
                    logger.error(f"Error processing document {doc_data.get('url', 'unknown')}: {e}")
                    db.rollback()
                    continue
            
            logger.info(f"Ingestion completed!")
            logger.info(f"Documents processed: {documents_processed}")
            logger.info(f"Chunks created: {chunks_created}")
            
            return True
            
        finally:
            db.close()
            
    except Exception as e:
        logger.error(f"Error during documentation ingestion: {e}")
        return False


def main():
    """Main function."""
    logger.info("wasmCloud Documentation Ingestion Script")
    logger.info("=" * 50)
    
    success = ingest_documents()
    
    if success:
        logger.info("Documentation ingestion completed successfully!")
        logger.info("You can now start the server and ask questions about wasmCloud.")
    else:
        logger.error("Documentation ingestion failed!")
        sys.exit(1)


if __name__ == "__main__":
    main() 