#!/usr/bin/env python3
"""Database initialization script for wasmCloud RAG bot."""

import sys
import os
import logging

# Add the parent directory to the path so we can import server modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from server.database import init_database, check_database_connection
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def main():
    """Initialize the database."""
    try:
        logger.info("Starting database initialization...")
        
        # Check if we can connect to the database
        logger.info("Checking database connection...")
        if not check_database_connection():
            logger.error("Cannot connect to database. Please check your DATABASE_URL configuration.")
            sys.exit(1)
        
        logger.info("Database connection successful!")
        
        # Initialize database (create tables and enable extensions)
        logger.info("Initializing database schema...")
        init_database()
        
        logger.info("Database initialization completed successfully!")
        logger.info("You can now run the ingestion script to populate the database with wasmCloud documentation.")
        
    except Exception as e:
        logger.error(f"Database initialization failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main() 