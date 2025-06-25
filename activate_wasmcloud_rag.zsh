#!/usr/bin/env zsh

# wasmCloud RAG Bot Virtual Environment Activation Script

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENV_DIR="$SCRIPT_DIR/venv"
VENV_ACTIVATE="$VENV_DIR/bin/activate"

if [[ -f $VENV_ACTIVATE ]]; then
    echo "🚀 Activating wasmCloud RAG Bot virtual environment..."
    source $VENV_ACTIVATE
    echo "✅ Virtual environment activated"
    echo ""
    echo "Available commands:"
    echo "  python3 -m server.main          # Start RAG server"
    echo "  python3 scripts/ingest_docs.py  # Ingest documentation"
    echo "  python3 test_client.py          # Test the API"
    echo "  python3 mcp_client.py           # Test MCP client"
    echo "  python3 mcp_server.py           # Start MCP server"
    echo ""
    echo "Or use make commands:"
    echo "  make run       # Start server"
    echo "  make test      # Run tests"
    echo "  make start     # Start all services"
    echo ""
    echo "To deactivate: deactivate"
else
    echo "❌ Virtual environment not found at $VENV_ACTIVATE"
    echo "Run ./scripts/setup.zsh to create it"
    exit 1
fi
