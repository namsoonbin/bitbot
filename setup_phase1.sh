#!/bin/bash
# HATS Trading System - Phase 1 Quick Setup Script (Linux/macOS)
# This script automates the Phase 1 infrastructure setup

set -e  # Exit on error

echo "========================================"
echo "HATS Trading System - Phase 1 Setup"
echo "========================================"
echo ""

# Check if Docker is installed
echo "[1/6] Checking Docker..."
if ! command -v docker &> /dev/null; then
    echo "ERROR: Docker is not installed"
    echo "Please install Docker: https://docs.docker.com/get-docker/"
    exit 1
fi
echo "OK: Docker is installed ($(docker --version))"
echo ""

# Check if Docker is running
if ! docker ps &> /dev/null; then
    echo "ERROR: Docker is not running"
    echo "Please start Docker and try again"
    exit 1
fi
echo "OK: Docker is running"
echo ""

# Check if Python is installed
echo "[2/6] Checking Python..."
if ! command -v python3 &> /dev/null; then
    echo "ERROR: Python 3 is not installed"
    echo "Please install Python 3.11+: https://www.python.org/downloads/"
    exit 1
fi
echo "OK: Python is installed ($(python3 --version))"
echo ""

# Create .env file if it doesn't exist
echo "[3/6] Setting up environment variables..."
if [ ! -f .env ]; then
    echo "Creating .env file from .env.example..."
    cp .env.example .env
    echo ""
    echo "IMPORTANT: Please edit .env file and add your API tokens:"
    echo "- CRYPTOPANIC_API_TOKEN (get from https://cryptopanic.com/developers/api/)"
    echo "- OPENAI_API_KEY or ANTHROPIC_API_KEY (for future agent implementation)"
    echo ""
    read -p "Press Enter to continue after editing .env file..."
else
    echo "OK: .env file already exists"
fi
echo ""

# Start Docker Compose
echo "[4/6] Starting Docker containers..."
docker-compose up -d
echo "OK: Docker containers started"
echo ""

# Wait for databases to be ready
echo "[5/6] Waiting for databases to initialize (15 seconds)..."
sleep 15
echo "OK: Databases should be ready"
echo ""

# Create virtual environment if it doesn't exist
echo "[6/6] Setting up Python environment..."
if [ ! -d venv ]; then
    echo "Creating virtual environment..."
    python3 -m venv venv
fi

echo "Activating virtual environment..."
source venv/bin/activate

echo "Installing dependencies..."
pip install --quiet --upgrade pip
pip install --quiet -r backend/requirements_agent.txt
echo "OK: Dependencies installed"
echo ""

echo "========================================"
echo "Phase 1 Setup Complete!"
echo "========================================"
echo ""
echo "Docker containers running:"
docker-compose ps
echo ""
echo "Web interfaces:"
echo "- Adminer (PostgreSQL):  http://localhost:8080"
echo "- Mongo Express (MongoDB): http://localhost:8081"
echo ""
echo "Next steps:"
echo "1. Run infrastructure tests: python backend/tests/test_phase1_infrastructure.py"
echo "2. Collect historical data: python backend/data/ccxt_collector.py --days 365"
echo "3. Collect news data: python backend/data/news_collector.py --days 7"
echo ""
echo "For more details, see: backend/README_PHASE1.md"
echo ""
