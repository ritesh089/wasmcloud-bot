# wasmCloud RAG Bot

A Retrieval-Augmented Generation (RAG) bot that uses wasmCloud documentation to answer questions about wasmCloud. Built with FastAPI, PostgreSQL with pgvector, and OpenAI GPT-4.

## Features

- **Documentation Scraping**: Automatically scrapes and processes wasmCloud documentation
- **Vector Database**: Uses PostgreSQL with pgvector for efficient similarity search
- **MCP Integration**: Model Context Protocol client and server for AI assistant integration
- **GPT-4 Integration**: Uses OpenAI GPT-4 for intelligent responses
- **Chunking Pipeline**: Smart text chunking with overlap for better context
- **Web Interface**: Beautiful chat interface for easy interaction

## Setup

### 🚀 Quick Start for Git Users

**Just cloned this repository?** See [SETUP_GUIDE.md](SETUP_GUIDE.md) for a 5-minute setup guide!

### Prerequisites

- Python 3.9+
- PostgreSQL with pgvector extension
- OpenAI API key
- **Required**: OpenAI API key (get from [OpenAI Platform](https://platform.openai.com/api-keys))

### Quick Start

**Complete Setup with Virtual Environment** (recommended):
```zsh
./scripts/setup.zsh
# or
make setup
```

This will:
- Create a Python virtual environment (`venv/`)
- Install all dependencies in isolation
- Set up PostgreSQL with Docker
- Initialize the database
- Create activation scripts

**Start All Services** (using zsh):
```zsh
./scripts/start.zsh    # Auto-activates virtual environment
# or
make start
```

**Stop All Services**:
```zsh
./scripts/stop.zsh
# or
make stop
```

**Manual Virtual Environment Usage**:
```bash
source venv/bin/activate          # Activate virtual environment
# or
source activate_wasmcloud_rag.zsh # Use helper script
```

**Alternative Setup** (Python script):
```bash
python3 setup.py  # Creates virtual environment automatically
```

### Virtual Environment Benefits:
- **Dependency Isolation**: No conflicts with system Python packages
- **Reproducible Environment**: Consistent dependencies across systems
- **Easy Cleanup**: Remove `venv/` directory to clean up completely
- **Automatic Management**: Scripts handle activation automatically

### Manual Installation

1. Create and activate virtual environment:
```bash
python3 -m venv venv
source venv/bin/activate
```

2. Install dependencies:
```bash
pip3 install -r requirements.txt
```

2. Set up PostgreSQL with pgvector:
```bash
# Using Docker (recommended)
docker-compose up -d postgres

# Or install PostgreSQL manually and add pgvector extension
CREATE EXTENSION IF NOT EXISTS vector;
```

3. Configure environment:
```bash
cp config.env.example .env
# Edit .env with your OpenAI API key and database settings
```

## Environment Configuration

### Required Environment Variables

Before running the application, you **must** configure the following environment variables:

1. **Copy the example configuration:**
   ```bash
   cp config.env.example .env
   ```

2. **Set your OpenAI API key:**
   - Get your API key from [OpenAI Platform](https://platform.openai.com/api-keys)
   - Edit `.env` and replace `your_openai_api_key_here` with your actual key:
   ```bash
   OPENAI_API_KEY=sk-proj-your-actual-api-key-here
   ```

3. **Database configuration (default values work with Docker):**
   ```bash
   # These values match docker-compose.yml - no changes needed if using Docker
   DATABASE_URL=postgresql://wasmcloud_user:wasmcloud_password@localhost:5432/wasmcloud_rag
   PGVECTOR_USER=wasmcloud_user
   PGVECTOR_PASSWORD=wasmcloud_password
   ```

4. **Optional: Customize AI models and processing:**
   ```bash
   # Use different OpenAI models if preferred
   EMBEDDING_MODEL=text-embedding-3-small  # or text-embedding-ada-002
   CHAT_MODEL=gpt-4-1106-preview           # or gpt-3.5-turbo
   
   # Adjust text processing
   CHUNK_SIZE=1000      # Larger = more context, slower processing
   CHUNK_OVERLAP=200    # Overlap between text chunks
   ```

### Environment Setup Examples

**Quick setup for local development:**
```bash
# Copy and edit the configuration
cp config.env.example .env

# Edit with your favorite editor
nano .env
# or
code .env
# or
vim .env

# Update the OpenAI API key line:
# OPENAI_API_KEY=sk-proj-your-actual-api-key-here
```

**Production setup:**
```bash
# Set environment variables directly (for deployment)
export OPENAI_API_KEY="your-actual-api-key"
export DATABASE_URL="postgresql://user:pass@host:5432/db"
export CHAT_MODEL="gpt-4-1106-preview"
```

### 🚨 Important Security Notes

- **Never commit `.env` files to git** - they contain sensitive API keys
- **Use different API keys for development and production**
- **Rotate API keys regularly for security**
- **Set up billing alerts in OpenAI dashboard to monitor usage**

### Troubleshooting Environment Issues

**OpenAI API Key Issues:**
```bash
# Test your API key
curl -H "Authorization: Bearer YOUR_API_KEY" \
     -H "Content-Type: application/json" \
     https://api.openai.com/v1/models
```

**Database Connection Issues:**
```bash
# Test database connection
python3 -c "
from server.database import check_database_connection
print('Database connected:', check_database_connection())
"
```

**Check all environment variables are loaded:**
```bash
python3 -c "
import os
from dotenv import load_dotenv
load_dotenv()
print('OpenAI Key configured:', bool(os.getenv('OPENAI_API_KEY')))
print('Database URL:', os.getenv('DATABASE_URL'))
print('Embedding Model:', os.getenv('EMBEDDING_MODEL'))
"
```

### Database Setup

Run the database initialization:
```bash
python3 scripts/init_db.py
```

### Data Ingestion

Scrape and ingest wasmCloud documentation:
```bash
python3 scripts/ingest_docs.py
```

### Running the Server

**Using zsh scripts** (recommended):
```zsh
./scripts/start.zsh                    # Start all services
./scripts/start.zsh --with-mcp         # Include MCP server
./scripts/dev.zsh                      # Development environment
./scripts/dev.zsh --with-tests --with-mcp  # Full development setup
```

**Manual start**:
```bash
python3 -m server.main
# or
make run
```

The server will be available at:
- **Web Interface**: `http://localhost:8000` 
- **API Documentation**: `http://localhost:8000/docs`
- **Database Admin**: `http://localhost:8080` (Adminer)

## API Endpoints

- `POST /query` - Ask questions about wasmCloud
- `GET /health` - Health check
- `POST /ingest` - Manually trigger documentation ingestion
- `GET /stats` - Get database statistics

## Usage

### Web Interface
Open `http://localhost:8000` in your browser for an interactive chat interface.

### API Usage
```python
import requests

response = requests.post("http://localhost:8000/query", json={
    "question": "What is wasmCloud and how does it work?"
})

print(response.json()["answer"])
```

### Test Client
```bash
python3 test_client.py
```

### MCP Integration
Use with AI assistants like Claude Desktop:

```bash
# Test MCP client
python3 mcp_client.py

# Start MCP server for AI assistant integration  
python3 mcp_server.py
```

## 📖 Documentation

- **[Knowledge Graph Guide](KNOWLEDGE_GRAPH_GUIDE.md)** - Advanced knowledge graph enhancement for relationship-aware reasoning
- **[System Design](SYSTEM_DESIGN.md)** - Comprehensive architecture, data flow, and implementation details
- **[Optimization Analysis](OPTIMIZATION_ANALYSIS.md)** - AI enhancement features and performance improvements  
- **[Setup Guide](SETUP_GUIDE.md)** - Complete installation and configuration instructions
- **[MCP Usage Guide](MCP_USAGE_GUIDE.md)** - Model Context Protocol integration for AI assistants

## Architecture

```
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│   Documentation │───▶│   Text Chunking  │───▶│   Embeddings    │
│     Scraper     │    │   & Processing   │    │   Generation    │
└─────────────────┘    └──────────────────┘    └─────────────────┘
                                                         │
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│   User Query    │───▶│  Vector Search   │◀───│  PostgreSQL +   │
│                 │    │   & Retrieval    │    │    pgvector     │
└─────────────────┘    └──────────────────┘    └─────────────────┘
         │                       │
         │              ┌─────────────────┐
         └─────────────▶│   GPT-4 RAG     │
                        │   Generation    │
                        └─────────────────┘
```

## zsh Scripts

The project includes comprehensive zsh scripts for easy management:

### Setup Script (`scripts/setup.zsh`)
Complete automated setup with colored output and error handling:
```zsh
./scripts/setup.zsh
```

### Start Script (`scripts/start.zsh`)
Start all services with monitoring:
```zsh
./scripts/start.zsh               # Basic startup
./scripts/start.zsh --with-mcp    # Include MCP server
```

### Stop Script (`scripts/stop.zsh`)
Stop services with cleanup options:
```zsh
./scripts/stop.zsh                # Stop services
./scripts/stop.zsh --clean-logs   # Stop and clean logs
./scripts/stop.zsh --clean-data   # Stop and clean database
./scripts/stop.zsh --all          # Stop and clean everything
```

### Development Script (`scripts/dev.zsh`)
Interactive development environment with hot reloading:
```zsh
./scripts/dev.zsh                       # Basic dev environment
./scripts/dev.zsh --with-tests          # Include continuous testing
./scripts/dev.zsh --with-mcp            # Include MCP server
./scripts/dev.zsh --with-tests --with-mcp  # Full development setup
```

Features:
- Hot reloading for Python files
- Real-time log monitoring
- Interactive development console
- Service status dashboard
- Continuous testing (optional)

## Project Structure

```
wasmcloud-bot/
├── server/
│   ├── main.py              # MCP server entry point
│   ├── models.py            # Database models
│   ├── database.py          # Database connection
│   ├── embeddings.py        # Embedding utilities
│   ├── rag.py              # RAG pipeline
│   └── scraper.py          # Documentation scraper
├── scripts/
│   ├── init_db.py          # Database initialization
│   ├── ingest_docs.py      # Documentation ingestion
│   ├── setup.zsh           # Complete setup automation
│   ├── start.zsh           # Service startup management
│   ├── stop.zsh            # Service shutdown management
│   └── dev.zsh             # Development environment
├── requirements.txt
├── .env.example
└── README.md
``` 