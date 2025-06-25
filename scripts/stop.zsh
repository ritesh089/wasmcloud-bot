#!/usr/bin/env zsh

# wasmCloud RAG Bot Stop Script (zsh)
# This script stops all wasmCloud RAG bot services

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

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

# Function to stop process by PID file
stop_process() {
    local pid_file=$1
    local service_name=$2
    
    if [[ -f $pid_file ]]; then
        local pid=$(cat $pid_file)
        if kill -0 $pid 2>/dev/null; then
            print_status "Stopping $service_name (PID: $pid)..."
            if kill $pid 2>/dev/null; then
                # Wait for process to stop
                local attempts=0
                while kill -0 $pid 2>/dev/null && [[ $attempts -lt 10 ]]; do
                    sleep 1
                    ((attempts++))
                done
                
                if kill -0 $pid 2>/dev/null; then
                    print_warning "Process didn't stop gracefully, force killing..."
                    kill -9 $pid 2>/dev/null || true
                fi
                
                print_success "$service_name stopped"
            else
                print_warning "Failed to stop $service_name"
            fi
        else
            print_warning "$service_name process not running"
        fi
        rm -f $pid_file
    else
        print_status "$service_name PID file not found (not running)"
    fi
}

# Stop RAG bot server
stop_rag_server() {
    print_status "Stopping wasmCloud RAG bot server..."
    stop_process ".rag_server.pid" "RAG bot server"
    
    # Clean up log file if requested
    if [[ "$1" == "--clean-logs" ]]; then
        if [[ -f rag_server.log ]]; then
            rm -f rag_server.log
            print_status "Removed RAG server log file"
        fi
    fi
}

# Stop MCP server
stop_mcp_server() {
    print_status "Stopping MCP server..."
    stop_process ".mcp_server.pid" "MCP server"
    
    # Clean up log file if requested
    if [[ "$1" == "--clean-logs" ]]; then
        if [[ -f mcp_server.log ]]; then
            rm -f mcp_server.log
            print_status "Removed MCP server log file"
        fi
    fi
}

# Stop database
stop_database() {
    print_status "Stopping PostgreSQL database..."
    
    if docker-compose down; then
        print_success "Database stopped"
    else
        print_warning "Failed to stop database or it wasn't running"
    fi
    
    # Clean up volumes if requested
    if [[ "$1" == "--clean-data" ]]; then
        print_warning "Removing database data volumes..."
        if docker-compose down -v; then
            print_success "Database volumes removed"
        else
            print_warning "Failed to remove database volumes"
        fi
    fi
}

# Stop all Docker containers
stop_all_containers() {
    if [[ "$1" == "--clean-data" ]]; then
        print_status "Stopping all containers and removing volumes..."
        docker-compose down -v
    else
        print_status "Stopping all containers..."
        docker-compose down
    fi
}

# Kill any remaining processes on our ports
cleanup_ports() {
    local ports=(8000 8001 5432 8080)
    
    for port in $ports; do
        local pids=$(lsof -ti :$port 2>/dev/null || true)
        if [[ -n $pids ]]; then
            print_warning "Found processes on port $port: $pids"
            if [[ "$1" == "--force" ]]; then
                print_status "Force killing processes on port $port..."
                echo $pids | xargs kill -9 2>/dev/null || true
            else
                print_warning "Use --force to kill these processes"
            fi
        fi
    done
}

# Show help
show_help() {
    echo "Usage: $0 [OPTIONS]"
    echo ""
    echo "Stop wasmCloud RAG bot services"
    echo ""
    echo "Options:"
    echo "  --clean-logs    Remove log files"
    echo "  --clean-data    Remove database data volumes"
    echo "  --force         Force kill processes on used ports"
    echo "  --all           Stop everything (equivalent to --clean-logs --clean-data)"
    echo "  --help          Show this help message"
    echo ""
    echo "Examples:"
    echo "  $0                    # Stop all services"
    echo "  $0 --clean-logs       # Stop services and remove logs"
    echo "  $0 --clean-data       # Stop services and remove database data"
    echo "  $0 --all              # Stop everything and clean up"
    echo "  $0 --force            # Stop services and force kill remaining processes"
}

# Main function
main() {
    local clean_logs=""
    local clean_data=""
    local force=""
    
    # Parse arguments
    for arg in "$@"; do
        case $arg in
            --clean-logs)
                clean_logs="--clean-logs"
                ;;
            --clean-data)
                clean_data="--clean-data"
                ;;
            --force)
                force="--force"
                ;;
            --all)
                clean_logs="--clean-logs"
                clean_data="--clean-data"
                ;;
            --help|-h)
                show_help
                exit 0
                ;;
            *)
                print_error "Unknown option: $arg"
                show_help
                exit 1
                ;;
        esac
    done
    
    echo "🛑 Stopping wasmCloud RAG Bot Environment (zsh)"
    echo "==============================================="
    echo
    
    # Stop services in reverse order
    stop_mcp_server $clean_logs
    stop_rag_server $clean_logs
    stop_database $clean_data
    
    # Clean up ports if requested
    if [[ -n $force ]]; then
        cleanup_ports $force
    fi
    
    echo
    print_success "🛑 wasmCloud RAG Bot Environment Stopped!"
    
    if [[ -n $clean_logs ]]; then
        print_status "Log files cleaned up"
    fi
    
    if [[ -n $clean_data ]]; then
        print_warning "Database data has been removed"
        print_status "You'll need to run setup and ingestion again:"
        print_status "  ./scripts/setup.zsh"
        print_status "  make ingest"
    fi
    
    echo
    print_status "To start again:"
    echo "  ./scripts/start.zsh           # Start services"
    echo "  ./scripts/start.zsh --with-mcp # Start with MCP server"
    echo "  make dev                      # Start development environment"
}

# Run main function
main "$@" 