#!/usr/bin/env python3
"""Setup script for wasmCloud RAG bot."""

import os
import sys
import subprocess
import asyncio
from pathlib import Path


def run_command(command, description):
    """Run a shell command and handle errors."""
    print(f"🔄 {description}...")
    try:
        result = subprocess.run(command, shell=True, check=True, capture_output=True, text=True)
        print(f"✅ {description} completed successfully")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ {description} failed: {e.stderr}")
        return False


def check_requirements():
    """Check if required tools are installed."""
    print("🔍 Checking requirements...")
    
    requirements = {
        'python3': 'python3 --version',
        'pip3': 'pip3 --version',
        'docker': 'docker --version',
        'docker-compose': 'docker-compose --version'
    }
    
    missing = []
    for tool, command in requirements.items():
        if not run_command(command, f"Checking {tool}"):
            missing.append(tool)
    
    if missing:
        print(f"❌ Missing requirements: {', '.join(missing)}")
        print("Please install the missing tools and try again.")
        return False
    
    print("✅ All requirements satisfied")
    return True


def setup_environment():
    """Set up environment variables."""
    env_file = Path('.env')
    env_example = Path('config.env.example')
    
    if env_file.exists():
        print("⚠️  .env file already exists, skipping environment setup")
        return True
    
    if not env_example.exists():
        print("❌ config.env.example not found")
        return False
    
    # Copy example to .env
    try:
        with open(env_example, 'r') as src, open(env_file, 'w') as dst:
            content = src.read()
            # Update with Docker Compose defaults
            content = content.replace('username:password@localhost:5432/wasmcloud_rag', 
                                    'wasmcloud_user:wasmcloud_password@localhost:5432/wasmcloud_rag')
            dst.write(content)
        
        print("✅ Environment file created (.env)")
        print("⚠️  Please edit .env and add your OpenAI API key")
        return True
    except Exception as e:
        print(f"❌ Failed to create .env file: {e}")
        return False


def main():
    """Main setup function."""
    print("🚀 wasmCloud RAG Bot Setup")
    print("=" * 40)
    
    # Check requirements
    if not check_requirements():
        sys.exit(1)
    
    # Install Python dependencies
    if not run_command("pip3 install -r requirements.txt", "Installing Python dependencies"):
        sys.exit(1)
    
    # Set up environment
    if not setup_environment():
        sys.exit(1)
    
    # Start database
    if not run_command("docker-compose up -d postgres", "Starting PostgreSQL database"):
        sys.exit(1)
    
    print("⏳ Waiting for database to be ready...")
    import time
    time.sleep(10)
    
    # Initialize database
    if not run_command("python3 scripts/init_db.py", "Initializing database"):
        sys.exit(1)
    
    print("\n🎉 Setup completed successfully!")
    print("\nNext steps:")
    print("1. Edit .env and add your OpenAI API key")
    print("2. Run: python3 scripts/ingest_docs.py")
    print("3. Run: python3 -m server.main")
    print("4. Open: http://localhost:8000")
    print("\nOr use the Makefile:")
    print("- make ingest    # Ingest documentation")
    print("- make run       # Start the server")
    print("- make test      # Test the API")


if __name__ == "__main__":
    main() 