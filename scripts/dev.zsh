#!/usr/bin/env zsh

# wasmCloud RAG Bot Development Script (zsh)
# This script provides a development environment with monitoring and hot reloading using virtual environment

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
PURPLE='\033[0;35m'
CYAN='\033[0;36m'
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

print_dev() {
    echo -e "${PURPLE}[DEV]${NC} $1"
}

print_monitor() {
    echo -e "${CYAN}[MONITOR]${NC} $1"
}

# Function to check if command exists
command_exists() {
    command -v "$1" >/dev/null 2>&1
}

# Function to activate virtual environment
activate_venv() {
    if [[ -f $VENV_ACTIVATE ]]; then
        print_dev "Activating virtual environment..."
        source $VENV_ACTIVATE
        print_success "Virtual environment activated: $VIRTUAL_ENV"
    else
        print_error "Virtual environment not found at $VENV_ACTIVATE"
        print_error "Please run ./scripts/setup.zsh first to create the virtual environment"
        exit 1
    fi
}

# Function to install development dependencies
install_dev_deps() {
    print_dev "Checking development dependencies..."
    
    # Make sure virtual environment is activated
    if [[ -z "$VIRTUAL_ENV" ]]; then
        activate_venv
    fi
    
    local missing_deps=()
    
    if ! python3 -c "import watchdog" 2>/dev/null; then
        missing_deps+=("watchdog")
    fi
    
    if ! command_exists entr; then
        if [[ "$OSTYPE" == "darwin"* ]]; then
            if ! command_exists brew; then
                print_warning "Homebrew not found, cannot install entr"
            else
                print_dev "Installing entr via Homebrew..."
                brew install entr
            fi
        else
            print_warning "entr not found, file watching may not work optimally"
        fi
    fi
    
    # Install Python development dependencies
    if [[ ${#missing_deps[@]} -gt 0 ]]; then
        print_dev "Installing Python development dependencies: ${missing_deps[*]}"
        python3 -m pip install watchdog
    fi
}

# Function to start development server with hot reload
start_dev_server() {
    print_dev "Starting development server with hot reload..."
    
    # Make sure virtual environment is activated
    if [[ -z "$VIRTUAL_ENV" ]]; then
        activate_venv
    fi
    
    if python3 -c "import watchdog" 2>/dev/null; then
        print_dev "Using watchdog for file monitoring"
        watchmedo auto-restart --directory=./server --pattern="*.py" --recursive -- python3 -m server.main &
        echo $! > .dev_server.pid
    else
        print_dev "Starting server without hot reload"
        python3 -m server.main &
        echo $! > .dev_server.pid
    fi
    
    print_success "Development server started (PID: $(cat .dev_server.pid))"
}

# Function to monitor logs
monitor_logs() {
    print_monitor "Setting up log monitoring..."
    
    # Create log monitoring script
    cat > .monitor_logs.zsh << 'EOF'
#!/usr/bin/env zsh

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
PURPLE='\033[0;35m'
CYAN='\033[0;36m'
NC='\033[0m'

print_log() {
    local level=$1
    local message=$2
    local timestamp=$(date '+%H:%M:%S')
    
    case $level in
        "ERROR")
            echo -e "${RED}[$timestamp ERROR]${NC} $message"
            ;;
        "WARNING")
            echo -e "${YELLOW}[$timestamp WARN]${NC} $message"
            ;;
        "INFO")
            echo -e "${BLUE}[$timestamp INFO]${NC} $message"
            ;;
        "DEBUG")
            echo -e "${CYAN}[$timestamp DEBUG]${NC} $message"
            ;;
        *)
            echo -e "${NC}[$timestamp]${NC} $message"
            ;;
    esac
}

# Monitor multiple log files
monitor_file() {
    local file=$1
    local prefix=$2
    
    if [[ -f $file ]]; then
        tail -f $file | while read line; do
            if [[ $line =~ "ERROR" ]]; then
                print_log "ERROR" "[$prefix] $line"
            elif [[ $line =~ "WARNING" ]]; then
                print_log "WARNING" "[$prefix] $line"
            elif [[ $line =~ "INFO" ]]; then
                print_log "INFO" "[$prefix] $line"
            else
                print_log "DEBUG" "[$prefix] $line"
            fi
        done &
    fi
}

# Start monitoring
echo "Starting log monitoring..."
monitor_file "rag_server.log" "RAG"
monitor_file "mcp_server.log" "MCP"

# Monitor Docker logs
if command -v docker-compose >/dev/null 2>&1; then
    docker-compose logs -f postgres 2>/dev/null | while read line; do
        print_log "INFO" "[DB] $line"
    done &
fi

wait
EOF

    chmod +x .monitor_logs.zsh
    ./.monitor_logs.zsh &
    echo $! > .monitor.pid
    
    print_success "Log monitoring started (PID: $(cat .monitor.pid))"
}

# Function to run tests continuously
run_continuous_tests() {
    print_dev "Setting up continuous testing..."
    
    # Create test runner script that uses virtual environment
    cat > .test_runner.zsh << EOF
#!/usr/bin/env zsh

# Activate virtual environment for tests
VENV_ACTIVATE="$VENV_ACTIVATE"
if [[ -f \$VENV_ACTIVATE ]]; then
    source \$VENV_ACTIVATE
fi

run_tests() {
    echo "🧪 Running tests at \$(date)"
    echo "=========================="
    
    # Health check
    if curl -s http://localhost:8000/health >/dev/null 2>&1; then
        echo "✅ Health check passed"
    else
        echo "❌ Health check failed"
        return 1
    fi
    
    # API test
    if python3 test_client.py >/dev/null 2>&1; then
        echo "✅ API tests passed"
    else
        echo "❌ API tests failed"
    fi
    
    # MCP test
    if python3 mcp_client.py >/dev/null 2>&1; then
        echo "✅ MCP tests passed"
    else
        echo "❌ MCP tests failed"
    fi
    
    echo "=========================="
    echo
}

# Run tests every 30 seconds
while true; do
    run_tests
    sleep 30
done
EOF

    chmod +x .test_runner.zsh
    
    if [[ "$1" == "--with-tests" ]]; then
        ./.test_runner.zsh &
        echo $! > .test_runner.pid
        print_success "Continuous testing started (PID: $(cat .test_runner.pid))"
    fi
}

# Function to show development dashboard
show_dashboard() {
    clear
    echo "🚀 wasmCloud RAG Bot Development Environment"
    echo "============================================"
    echo
    echo "🐍 Virtual Environment: $VIRTUAL_ENV"
    echo
    echo "📊 Services Status:"
    
    # Check RAG server
    if [[ -f .dev_server.pid ]] && kill -0 $(cat .dev_server.pid) 2>/dev/null; then
        echo "  ✅ RAG Server: Running (PID: $(cat .dev_server.pid))"
    else
        echo "  ❌ RAG Server: Not running"
    fi
    
    # Check database
    if docker-compose ps postgres | grep -q "Up"; then
        echo "  ✅ Database: Running"
    else
        echo "  ❌ Database: Not running"
    fi
    
    # Check MCP server
    if [[ -f .mcp_server.pid ]] && kill -0 $(cat .mcp_server.pid) 2>/dev/null; then
        echo "  ✅ MCP Server: Running (PID: $(cat .mcp_server.pid))"
    else
        echo "  ⚪ MCP Server: Not running"
    fi
    
    # Check monitoring
    if [[ -f .monitor.pid ]] && kill -0 $(cat .monitor.pid) 2>/dev/null; then
        echo "  ✅ Log Monitor: Running (PID: $(cat .monitor.pid))"
    else
        echo "  ⚪ Log Monitor: Not running"
    fi
    
    # Check continuous testing
    if [[ -f .test_runner.pid ]] && kill -0 $(cat .test_runner.pid) 2>/dev/null; then
        echo "  ✅ Continuous Tests: Running (PID: $(cat .test_runner.pid))"
    else
        echo "  ⚪ Continuous Tests: Not running"
    fi
    
    echo
    echo "🌐 Endpoints:"
    echo "  📱 Web Interface:     http://localhost:8000"
    echo "  📚 API Docs:          http://localhost:8000/docs"
    echo "  🗄️ Database Admin:    http://localhost:8080"
    echo
    echo "🔧 Development Commands:"
    echo "  r  - Restart RAG server"
    echo "  m  - Start/Stop MCP server"
    echo "  t  - Run tests manually"
    echo "  l  - Show recent logs"
    echo "  s  - Show statistics"
    echo "  h  - Show this help"
    echo "  q  - Quit development environment"
    echo
}

# Function to handle user input
handle_input() {
    while true; do
        echo -n "dev> "
        read -r command
        
        case $command in
            "r"|"restart")
                print_dev "Restarting RAG server..."
                if [[ -f .dev_server.pid ]]; then
                    kill $(cat .dev_server.pid) 2>/dev/null || true
                    rm -f .dev_server.pid
                fi
                sleep 2
                start_dev_server
                ;;
            "m"|"mcp")
                if [[ -f .mcp_server.pid ]]; then
                    print_dev "Stopping MCP server..."
                    kill $(cat .mcp_server.pid) 2>/dev/null || true
                    rm -f .mcp_server.pid
                else
                    print_dev "Starting MCP server..."
                    # Make sure virtual environment is activated
                    if [[ -z "$VIRTUAL_ENV" ]]; then
                        activate_venv
                    fi
                    python3 mcp_server.py > mcp_server.log 2>&1 &
                    echo $! > .mcp_server.pid
                    print_success "MCP server started (PID: $(cat .mcp_server.pid))"
                fi
                ;;
            "t"|"test")
                print_dev "Running manual tests..."
                # Make sure virtual environment is activated
                if [[ -z "$VIRTUAL_ENV" ]]; then
                    activate_venv
                fi
                python3 test_client.py
                ;;
            "l"|"logs")
                print_dev "Recent logs:"
                if [[ -f rag_server.log ]]; then
                    echo "=== RAG Server Logs ==="
                    tail -20 rag_server.log
                fi
                if [[ -f mcp_server.log ]]; then
                    echo "=== MCP Server Logs ==="
                    tail -20 mcp_server.log
                fi
                ;;
            "s"|"stats")
                print_dev "Getting statistics..."
                curl -s http://localhost:8000/stats | python3 -m json.tool
                ;;
            "h"|"help")
                show_dashboard
                ;;
            "q"|"quit"|"exit")
                print_dev "Stopping development environment..."
                break
                ;;
            "")
                # Refresh dashboard on empty input
                show_dashboard
                ;;
            *)
                print_warning "Unknown command: $command (type 'h' for help)"
                ;;
        esac
    done
}

# Function to cleanup on exit
cleanup() {
    print_dev "Cleaning up development environment..."
    
    # Stop all background processes
    for pid_file in .dev_server.pid .mcp_server.pid .monitor.pid .test_runner.pid; do
        if [[ -f $pid_file ]]; then
            local pid=$(cat $pid_file)
            kill $pid 2>/dev/null || true
            rm -f $pid_file
        fi
    done
    
    # Clean up temporary scripts
    rm -f .monitor_logs.zsh .test_runner.zsh
    
    print_success "Development environment stopped"
    exit 0
}

# Function to show help
show_help() {
    echo "Usage: $0 [OPTIONS]"
    echo ""
    echo "Start wasmCloud RAG bot development environment"
    echo ""
    echo "Options:"
    echo "  --with-tests    Enable continuous testing"
    echo "  --with-mcp      Start MCP server"
    echo "  --no-monitor    Disable log monitoring"
    echo "  --help          Show this help message"
    echo ""
    echo "Features:"
    echo "  • Automatic virtual environment activation"
    echo "  • Hot reloading for Python files"
    echo "  • Real-time log monitoring"
    echo "  • Continuous testing (optional)"
    echo "  • Interactive development console"
    echo "  • Service status dashboard"
}

# Main function
main() {
    local with_tests=""
    local with_mcp=""
    local no_monitor=""
    
    # Parse arguments
    for arg in "$@"; do
        case $arg in
            --with-tests)
                with_tests="--with-tests"
                ;;
            --with-mcp)
                with_mcp="--with-mcp"
                ;;
            --no-monitor)
                no_monitor="--no-monitor"
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
    
    # Set up signal handlers
    trap cleanup SIGINT SIGTERM
    
    print_dev "Starting wasmCloud RAG Bot Development Environment..."
    
    # Activate virtual environment first
    activate_venv
    
    # Install development dependencies
    install_dev_deps
    
    # Start database if not running
    if ! docker-compose ps postgres | grep -q "Up"; then
        print_dev "Starting database..."
        docker-compose up -d postgres
        sleep 5
    fi
    
    # Start development server
    start_dev_server
    
    # Start MCP server if requested
    if [[ -n $with_mcp ]]; then
        print_dev "Starting MCP server..."
        python3 mcp_server.py > mcp_server.log 2>&1 &
        echo $! > .mcp_server.pid
        print_success "MCP server started (PID: $(cat .mcp_server.pid))"
    fi
    
    # Start log monitoring
    if [[ -z $no_monitor ]]; then
        monitor_logs
    fi
    
    # Start continuous testing
    run_continuous_tests $with_tests
    
    # Wait for server to be ready
    sleep 3
    
    # Show dashboard and start interactive mode
    show_dashboard
    handle_input
    
    # Cleanup on exit
    cleanup
}

# Run main function
main "$@" 