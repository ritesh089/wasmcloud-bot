#!/usr/bin/env zsh

# wasmCloud RAG Bot Setup Script (zsh)
# This script automates the complete setup process with virtual environment

set -e  # Exit on any error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Virtual environment settings
VENV_DIR="venv"
VENV_ACTIVATE="$VENV_DIR/bin/activate"

# Function to print colored output
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
        print_success "Virtual environment activated"
    else
        print_error "Virtual environment not found at $VENV_ACTIVATE"
        return 1
    fi
}

# Function to create virtual environment
create_virtual_environment() {
    print_status "Setting up Python virtual environment..."
    
    if [[ -d $VENV_DIR ]]; then
        print_warning "Virtual environment already exists at $VENV_DIR"
        print_status "Removing existing virtual environment..."
        rm -rf $VENV_DIR
    fi
    
    # Create virtual environment
    if python3 -m venv $VENV_DIR; then
        print_success "Virtual environment created at $VENV_DIR"
    else
        print_error "Failed to create virtual environment"
        print_error "Make sure python3-venv is installed:"
        print_error "  Ubuntu/Debian: sudo apt install python3-venv"
        print_error "  macOS: python3 -m pip install --user virtualenv"
        exit 1
    fi
    
    # Activate virtual environment
    activate_venv
    
    # Upgrade pip in virtual environment
    print_status "Upgrading pip in virtual environment..."
    python3 -m pip install --upgrade pip
    
    print_success "Virtual environment setup completed"
}

# Check prerequisites
check_prerequisites() {
    print_status "Checking prerequisites..."
    
    local missing_deps=()
    
    if ! command_exists python3; then
        missing_deps+=("python3")
    fi
    
    if ! command_exists docker; then
        missing_deps+=("docker")
    fi
    
    if ! command_exists docker-compose; then
        missing_deps+=("docker-compose")
    fi
    
    if [[ ${#missing_deps[@]} -gt 0 ]]; then
        print_error "Missing required dependencies: ${missing_deps[*]}"
        print_error "Please install the missing dependencies and try again."
        exit 1
    fi
    
    # Check if python3-venv is available
    if ! python3 -m venv --help >/dev/null 2>&1; then
        print_error "python3-venv module not available"
        print_error "Please install python3-venv:"
        print_error "  Ubuntu/Debian: sudo apt install python3-venv"
        print_error "  macOS: Should be included with Python 3"
        exit 1
    fi
    
    print_success "All prerequisites satisfied"
}

# Install Python dependencies in virtual environment
install_dependencies() {
    print_status "Installing Python dependencies in virtual environment..."
    
    # Make sure we're in the virtual environment
    if [[ -z "$VIRTUAL_ENV" ]]; then
        print_error "Virtual environment not activated"
        exit 1
    fi
    
    if python3 -m pip install -r requirements.txt; then
        print_success "Python dependencies installed in virtual environment"
        print_status "Installed packages:"
        python3 -m pip list | head -10
    else
        print_error "Failed to install Python dependencies"
        exit 1
    fi
}

# Setup environment file
setup_environment() {
    print_status "Setting up environment configuration..."
    
    if [[ -f .env ]]; then
        print_warning ".env file already exists, skipping environment setup"
        return 0
    fi
    
    if [[ ! -f config.env.example ]]; then
        print_error "config.env.example not found"
        exit 1
    fi
    
    # Copy and update environment file
    cp config.env.example .env
    
    # Update database URL for Docker Compose
    if command_exists sed; then
        sed -i.bak 's/username:password@localhost:5432\/wasmcloud_rag/wasmcloud_user:wasmcloud_password@localhost:5432\/wasmcloud_rag/' .env
        rm -f .env.bak
    fi
    
    print_success "Environment file created (.env)"
    print_warning "Please edit .env and add your OpenAI API key!"
}

# Start database
start_database() {
    print_status "Starting PostgreSQL database with Docker Compose..."
    
    if docker-compose up -d postgres; then
        print_success "Database started"
        print_status "Waiting for database to be ready..."
        sleep 10
    else
        print_error "Failed to start database"
        exit 1
    fi
}

# Initialize database
initialize_database() {
    print_status "Initializing database schema..."
    
    # Make sure we're in the virtual environment
    if [[ -z "$VIRTUAL_ENV" ]]; then
        activate_venv
    fi
    
    if python3 scripts/init_db.py; then
        print_success "Database initialized"
    else
        print_error "Failed to initialize database"
        exit 1
    fi
}

# Check if OpenAI API key is set
check_openai_key() {
    if [[ -f .env ]]; then
        if grep -q "OPENAI_API_KEY=your_openai_api_key_here" .env; then
            print_warning "OpenAI API key is not set in .env file"
            print_warning "Please edit .env and add your OpenAI API key before proceeding"
            return 1
        fi
        
        if grep -q "OPENAI_API_KEY=$" .env; then
            print_warning "OpenAI API key appears to be empty"
            return 1
        fi
    else
        print_error ".env file not found"
        return 1
    fi
    
    return 0
}

# Ingest documentation
ingest_documentation() {
    print_status "Checking OpenAI API key..."
    
    if ! check_openai_key; then
        print_error "Cannot proceed with documentation ingestion without OpenAI API key"
        print_error "Please edit .env file and add your OpenAI API key, then run:"
        print_error "  source $VENV_ACTIVATE && python3 scripts/ingest_docs.py"
        return 1
    fi
    
    print_status "Ingesting wasmCloud documentation..."
    print_status "This may take several minutes..."
    
    # Make sure we're in the virtual environment
    if [[ -z "$VIRTUAL_ENV" ]]; then
        activate_venv
    fi
    
    if python3 scripts/ingest_docs.py; then
        print_success "Documentation ingested successfully"
    else
        print_error "Failed to ingest documentation"
        return 1
    fi
}

# Create activation script
create_activation_script() {
    print_status "Creating virtual environment activation script..."
    
    cat > activate_wasmcloud_rag.zsh << 'EOF'
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
EOF

    chmod +x activate_wasmcloud_rag.zsh
    print_success "Created activation script: activate_wasmcloud_rag.zsh"
}

# Main setup function
main() {
    echo "🚀 wasmCloud RAG Bot Setup with Virtual Environment (zsh)"
    echo "========================================================"
    echo
    
    check_prerequisites
    create_virtual_environment
    install_dependencies
    setup_environment
    start_database
    initialize_database
    create_activation_script
    
    echo
    print_success "Setup completed successfully!"
    echo
    print_status "Virtual environment created at: $VENV_DIR"
    print_status "To activate the virtual environment:"
    echo "  source $VENV_ACTIVATE"
    echo "  # or"
    echo "  source activate_wasmcloud_rag.zsh"
    echo
    print_status "Next steps:"
    echo "1. Edit .env and add your OpenAI API key"
    echo "2. Activate virtual environment: source $VENV_ACTIVATE"
    echo "3. Run: python3 scripts/ingest_docs.py  (or make ingest)"
    echo "4. Run: python3 -m server.main  (or make run)"
    echo "5. Open: http://localhost:8000"
    echo
    print_status "Or use the zsh scripts (they will auto-activate the virtual environment):"
    echo "  ./scripts/start.zsh    # Start all services"
    echo "  ./scripts/dev.zsh      # Development environment"
    echo
    
    # Try to ingest documentation if API key is available
    if ingest_documentation; then
        echo
        print_success "🎉 Complete setup finished!"
        print_status "Virtual environment is ready at: $VENV_DIR"
        print_status "You can now start the server with: ./scripts/start.zsh"
    else
        echo
        print_warning "Setup completed, but documentation ingestion was skipped"
        print_warning "Add your OpenAI API key to .env and run:"
        print_warning "  source $VENV_ACTIVATE && python3 scripts/ingest_docs.py"
    fi
}

# Run main function
main "$@" 