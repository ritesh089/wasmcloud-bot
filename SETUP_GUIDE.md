# wasmCloud RAG Bot - Quick Setup Guide

This guide helps you set up the wasmCloud RAG Bot after cloning from git.

## 🚀 Quick Start (5 minutes)

### 1. Clone and Enter Directory
```bash
git clone <repository-url>
cd wasmcloud-bot
```

### 2. Set Up Environment Variables
```bash
# Copy the example configuration
cp config.env.example .env

# Edit the .env file with your OpenAI API key
nano .env  # or use your preferred editor
```

**Required: Add your OpenAI API key to `.env`:**
```bash
OPENAI_API_KEY=sk-proj-your-actual-api-key-here
```

### 3. Run Automated Setup
```bash
# This handles everything: virtual environment, dependencies, database
./scripts/setup.zsh
```

### 4. Start the Application
```bash
# Start all services (database, server, web interface)
./scripts/start.zsh
```

### 5. Access the Interface
- **Web Interface**: http://localhost:8000
- **API Documentation**: http://localhost:8000/docs
- **Database Admin**: http://localhost:8080

## 🔑 Environment Variables Setup

### Required Variables

| Variable | Description | Where to Get |
|----------|-------------|--------------|
| `OPENAI_API_KEY` | OpenAI API key for embeddings and chat | [OpenAI Platform](https://platform.openai.com/api-keys) |

### Optional Variables (have sensible defaults)

| Variable | Default | Description |
|----------|---------|-------------|
| `DATABASE_URL` | `postgresql://wasmcloud_user:wasmcloud_password@localhost:5432/wasmcloud_rag` | PostgreSQL connection string |
| `EMBEDDING_MODEL` | `text-embedding-3-small` | OpenAI embedding model |
| `CHAT_MODEL` | `gpt-4-1106-preview` | OpenAI chat model |
| `CHUNK_SIZE` | `1000` | Text chunk size in tokens |
| `CHUNK_OVERLAP` | `200` | Overlap between chunks |
| `PORT` | `8000` | Main server port |

### Edit Environment File

```bash
# Use your preferred editor
nano .env        # Simple editor
code .env        # VS Code
vim .env         # Vim
emacs .env       # Emacs
```

**Minimum required `.env` content:**
```bash
OPENAI_API_KEY=sk-proj-your-actual-api-key-here
```

## 🛠️ Manual Setup (Alternative)

If you prefer manual setup or the scripts don't work:

### 1. Create Virtual Environment
```bash
python3 -m venv venv
source venv/bin/activate
```

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

### 3. Start Database
```bash
docker-compose up -d postgres
```

### 4. Initialize Database
```bash
python3 scripts/init_db.py
```

### 5. Ingest Documentation
```bash
python3 scripts/ingest_docs.py
```

### 6. Start Server
```bash
python3 -m server.main
```

## 🔍 Verification

### Check Everything is Working

1. **Health Check:**
   ```bash
   curl http://localhost:8000/health
   ```

2. **Test Query:**
   ```bash
   curl -X POST http://localhost:8000/query \
     -H "Content-Type: application/json" \
     -d '{"question": "What is wasmCloud?"}'
   ```

3. **Check Stats:**
   ```bash
   curl http://localhost:8000/stats
   ```

### Expected Responses

- **Health Check**: `{"status": "healthy", "database_connected": true}`
- **Stats**: Should show documents and chunks > 0 after ingestion
- **Query**: Should return a detailed answer about wasmCloud

## 🚨 Common Issues

### "No such file or directory" for scripts
```bash
# Make scripts executable
chmod +x scripts/*.zsh
```

### "OpenAI API key not found"
```bash
# Check your .env file
cat .env | grep OPENAI_API_KEY

# Make sure it starts with sk-
```

### "Database connection failed"
```bash
# Start the database
docker-compose up -d postgres

# Wait a moment for startup
sleep 5

# Test connection
python3 scripts/init_db.py
```

### "No documents found" or "0 chunks"
```bash
# Run the ingestion process
python3 scripts/ingest_docs.py

# This takes 5-15 minutes depending on your internet speed
```

## 🎯 Next Steps

1. **Try the Web Interface**: Open http://localhost:8000
2. **Ask Questions**: Try asking about wasmCloud features
3. **Explore API**: Check http://localhost:8000/docs
4. **Customize**: Edit models and settings in `.env`

## 🔧 Management Commands

```bash
# Start services
./scripts/start.zsh

# Start with MCP server for AI assistant integration
./scripts/start.zsh --with-mcp

# Stop services
./scripts/stop.zsh

# Development mode with hot reloading
./scripts/dev.zsh

# Clean restart (stops, cleans, starts)
./scripts/stop.zsh --clean-data && ./scripts/start.zsh
```

## 📚 Additional Resources

- **Full Documentation**: See `README.md`
- **MCP Integration**: See `MCP_USAGE_GUIDE.md`
- **API Reference**: http://localhost:8000/docs (when running)
- **Setup Status**: See `SETUP_STATUS.md` for detailed status

## 🆘 Getting Help

If you encounter issues:

1. **Check logs**: `tail -f rag_server.log`
2. **Verify environment**: Run verification commands above
3. **Clean restart**: `./scripts/stop.zsh --all && ./scripts/start.zsh`
4. **Check GitHub issues**: Look for similar problems
5. **Create new issue**: Include error messages and setup details 