-- Initialize pgvector extension
CREATE EXTENSION IF NOT EXISTS vector;

-- Create indexes for better performance
-- These will be created after tables are set up by the application 