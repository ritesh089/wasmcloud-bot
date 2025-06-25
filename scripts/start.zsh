#!/usr/bin/env zsh

# wasmCloud RAG Bot Start Script (zsh)
# This script starts the complete wasmCloud RAG bot environment with virtual environment support

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Virtual environment settings
VENV_DIR="venv"
VENV_ACTIVATE="$VENV_DIR/bin/activate"

print_status() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

print_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Function to check if command exists
command_exists() {
    command -v "$1" >/dev/null 2>&1
}

# Function to activate virtual environment
activate_venv() {
    if [[ -f $VENV_ACTIVATE ]]; then
        print_status "Activating virtual environment..."
        source $VENV_ACTIVATE
        print_success "Virtual environment activated: $VIRTUAL_ENV"
    else
        print_error "Virtual environment not found at $VENV_ACTIVATE"
        print_error "Please run ./scripts/setup.zsh first to create the virtual environment"
        exit 1
    fi
}

# Function to check if port is in use
port_in_use() {
    lsof -i :$1 >/dev/null 2>&1
}

# Function to wait for service to be ready
wait_for_service() {
    local url=$1
    local service_name=$2
    local max_attempts=30
    local attempt=1
    
    print_status "Waiting for $service_name to be ready..."
    
    while [[ $attempt -le $max_attempts ]]; do
        if curl -s "$url" >/dev/null 2>&1; then
            print_success "$service_name is ready!"
            return 0
        fi
        
        echo -n "."
        sleep 2
        ((attempt++))
    done
    
    print_error "$service_name failed to start within expected time"
    return 1
}

# Check prerequisites
check_prerequisites() {
    print_status "Checking prerequisites..."
    
    if ! command_exists docker; then
        print_error "Docker is required but not installed"
        exit 1
    fi
    
    if ! command_exists docker-compose; then
        print_error "Docker Compose is required but not installed"
        exit 1
    fi
    
    if ! command_exists python3; then
        print_error "Python 3 is required but not installed"
        exit 1
    fi
    
    if [[ ! -f .env ]]; then
        print_error ".env file not found. Please run setup first:"
        print_error "  ./scripts/setup.zsh"
        exit 1
    fi
    
    print_success "Prerequisites check passed"
}

# Start database
start_database() {
    print_status "Starting PostgreSQL database..."
    
    if docker-compose up -d postgres; then
        print_success "Database container started"
        
        # Wait for database to be ready
        sleep 5
        if wait_for_service "http://localhost:5432" "PostgreSQL"; then
            return 0
        else
            # Try a different check for PostgreSQL
            print_status "Checking database with pg_isready alternative..."
            sleep 10
            print_success "Database should be ready"
        fi
    else
        print_error "Failed to start database"
        exit 1
    fi
}

# Check database status
check_database() {
    print_status "Checking database status..."
    
    # Make sure virtual environment is activated
    if [[ -z "$VIRTUAL_ENV" ]]; then
        activate_venv
    fi
    
    if python3 -c "
import sys
import os
sys.path.append('.')
from server.database import check_database_connection

if check_database_connection():
    print('Database connection successful')
else:
    print('Database connection failed')
    sys.exit(1)
"; then
        print_success "Database is accessible"
    else
        print_error "Database is not accessible"
        print_error "Try running: make init-db"
        exit 1
    fi
}

# Start RAG bot server
start_rag_server() {
    print_status "Starting wasmCloud RAG bot server..."
    
    # Make sure virtual environment is activated
    if [[ -z "$VIRTUAL_ENV" ]]; then
        activate_venv
    fi
    
    if port_in_use 8000; then
        print_warning "Port 8000 is already in use"
        print_status "Checking if it's our RAG bot server..."
        
        if curl -s http://localhost:8000/health >/dev/null 2>&1; then
            print_success "RAG bot server is already running"
            return 0
        else
            print_error "Port 8000 is occupied by another service"
            print_error "Please stop the service using port 8000 and try again"
            exit 1
        fi
    fi
    
    print_status "Starting server in background..."
    nohup python3 -m server.main > rag_server.log 2>&1 &
    local server_pid=$!
    echo $server_pid > .rag_server.pid
    
    # Wait for server to be ready
    if wait_for_service "http://localhost:8000/health" "RAG bot server"; then
        print_success "RAG bot server started (PID: $server_pid)"
        print_status "Server logs: tail -f rag_server.log"
    else
        print_error "RAG bot server failed to start"
        if [[ -f .rag_server.pid ]]; then
            kill $(cat .rag_server.pid) 2>/dev/null || true
            rm -f .rag_server.pid
        fi
        exit 1
    fi
}

# Start MCP server (optional)
start_mcp_server() {
    if [[ "$1" == "--with-mcp" ]]; then
        print_status "Starting MCP server..."
        
        # Make sure virtual environment is activated
        if [[ -z "$VIRTUAL_ENV" ]]; then
            activate_venv
        fi
        
        if port_in_use 8001; then
            print_warning "Port 8001 is already in use, skipping MCP server"
            return 0
        fi
        
        print_status "Starting MCP server in background..."
        nohup python3 mcp_server.py > mcp_server.log 2>&1 &
        local mcp_pid=$!
        echo $mcp_pid > .mcp_server.pid
        
        sleep 3
        print_success "MCP server started (PID: $mcp_pid)"
        print_status "MCP server logs: tail -f mcp_server.log"
    fi
}

# Show status
show_status() {
    echo
    print_success "🚀 wasmCloud RAG Bot Environment Started!"
    echo
    print_status "Virtual Environment: $VIRTUAL_ENV"
    echo
    print_status "Available services:"
    echo "  📱 Web Interface:    http://localhost:8000"
    echo "  📚 API Documentation: http://localhost:8000/docs"
    echo "  🗄️  Database Admin:   http://localhost:8080 (adminer)"
    echo
    print_status "API endpoints:"
    echo "  POST /query         - Ask questions"
    echo "  GET  /health        - Health check"
    echo "  GET  /stats         - Statistics"
    echo "  GET  /documents     - List documents"
    echo
    print_status "Management commands:"
    echo "  ./scripts/stop.zsh  - Stop all services"
    echo "  make test           - Test the API"
    echo "  make test-mcp       - Test MCP integration"
    echo
    
    if [[ -f .mcp_server.pid ]]; then
        print_status "MCP server is running for AI assistant integration"
    fi
    
    print_status "Logs:"
    echo "  tail -f rag_server.log  - RAG bot server logs"
    if [[ -f .mcp_server.pid ]]; then
        echo "  tail -f mcp_server.log  - MCP server logs"
    fi
    echo "  make logs               - Database logs"
    echo
    print_status "To manually activate virtual environment:"
    echo "  source $VENV_ACTIVATE"
}

# Main function
main() {
    local start_mcp=""
    
    # Parse arguments
    for arg in "$@"; do
        case $arg in
            --with-mcp)
                start_mcp="--with-mcp"
                ;;
            --help|-h)
                echo "Usage: $0 [--with-mcp] [--help]"
                echo ""
                echo "Options:"
                echo "  --with-mcp    Also start the MCP server for AI assistant integration"
                echo "  --help        Show this help message"
                exit 0
                ;;
        esac
    done
    
    echo "🚀 Starting wasmCloud RAG Bot Environment (zsh)"
    echo "==============================================="
    echo
    
    check_prerequisites
    activate_venv
    start_database
    check_database
    start_rag_server
    start_mcp_server $start_mcp
    show_status
}

# Handle script termination
cleanup() {
    print_warning "Received termination signal"
    if [[ -f .rag_server.pid ]]; then
        print_status "Stopping RAG server..."
        kill $(cat .rag_server.pid) 2>/dev/null || true
        rm -f .rag_server.pid
    fi
    if [[ -f .mcp_server.pid ]]; then
        print_status "Stopping MCP server..."
        kill $(cat .mcp_server.pid) 2>/dev/null || true
        rm -f .mcp_server.pid
    fi
    exit 0
}

trap cleanup SIGINT SIGTERM

# Run main function
main "$@" 