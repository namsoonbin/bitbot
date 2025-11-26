@echo off
REM HATS Trading System - Phase 1 Quick Setup Script (Windows)
REM This script automates the Phase 1 infrastructure setup

echo ========================================
echo HATS Trading System - Phase 1 Setup
echo ========================================
echo.

REM Check if Docker is running
echo [1/6] Checking Docker...
docker --version >nul 2>&1
if %errorlevel% neq 0 (
    echo ERROR: Docker is not installed or not in PATH
    echo Please install Docker Desktop: https://www.docker.com/products/docker-desktop/
    pause
    exit /b 1
)
echo OK: Docker is installed
echo.

REM Check if Docker is running
docker ps >nul 2>&1
if %errorlevel% neq 0 (
    echo ERROR: Docker is not running
    echo Please start Docker Desktop and try again
    pause
    exit /b 1
)
echo OK: Docker is running
echo.

REM Check if Python is installed
echo [2/6] Checking Python...
python --version >nul 2>&1
if %errorlevel% neq 0 (
    echo ERROR: Python is not installed or not in PATH
    echo Please install Python 3.11+: https://www.python.org/downloads/
    pause
    exit /b 1
)
python --version
echo OK: Python is installed
echo.

REM Create .env file if it doesn't exist
echo [3/6] Setting up environment variables...
if not exist .env (
    echo Creating .env file from .env.example...
    copy .env.example .env
    echo.
    echo IMPORTANT: Please edit .env file and add your API tokens:
    echo - CRYPTOPANIC_API_TOKEN (get from https://cryptopanic.com/developers/api/)
    echo - OPENAI_API_KEY or ANTHROPIC_API_KEY (for future agent implementation)
    echo.
    echo Press any key to continue after editing .env file...
    pause >nul
) else (
    echo OK: .env file already exists
)
echo.

REM Start Docker Compose
echo [4/6] Starting Docker containers...
docker-compose up -d
if %errorlevel% neq 0 (
    echo ERROR: Failed to start Docker containers
    echo Check: docker-compose logs
    pause
    exit /b 1
)
echo OK: Docker containers started
echo.

REM Wait for databases to be ready
echo [5/6] Waiting for databases to initialize (15 seconds)...
timeout /t 15 /nobreak >nul
echo OK: Databases should be ready
echo.

REM Create virtual environment if it doesn't exist
echo [6/6] Setting up Python environment...
if not exist venv (
    echo Creating virtual environment...
    python -m venv venv
    if %errorlevel% neq 0 (
        echo ERROR: Failed to create virtual environment
        pause
        exit /b 1
    )
)

echo Activating virtual environment...
call venv\Scripts\activate.bat

echo Installing dependencies...
pip install --quiet --upgrade pip
pip install --quiet -r backend\requirements_agent.txt
if %errorlevel% neq 0 (
    echo ERROR: Failed to install dependencies
    echo Try manually: pip install -r backend\requirements_agent.txt
    pause
    exit /b 1
)
echo OK: Dependencies installed
echo.

echo ========================================
echo Phase 1 Setup Complete!
echo ========================================
echo.
echo Docker containers running:
docker-compose ps
echo.
echo Web interfaces:
echo - Adminer (PostgreSQL):  http://localhost:8080
echo - Mongo Express (MongoDB): http://localhost:8081
echo.
echo Next steps:
echo 1. Run infrastructure tests: python backend\tests\test_phase1_infrastructure.py
echo 2. Collect historical data: python backend\data\ccxt_collector.py --days 365
echo 3. Collect news data: python backend\data\news_collector.py --days 7
echo.
echo For more details, see: backend\README_PHASE1.md
echo.
pause
