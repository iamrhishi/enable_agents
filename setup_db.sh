#!/bin/bash

###############################################################################
# Database Setup Script
# 
# This script initializes the database on a remote server after pulling code
# from git. It should be run once after deployment.
#
# Usage: ./setup_db.sh
###############################################################################

set -e  # Exit on error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo -e "${YELLOW}========================================${NC}"
echo -e "${YELLOW}Enable Agents - Database Setup${NC}"
echo -e "${YELLOW}========================================${NC}\n"

# Get the directory where the script is located
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
PROJECT_ROOT="$SCRIPT_DIR"

# Paths
VENV_PATH="$PROJECT_ROOT/venv"
TOOLS_DIR="$PROJECT_ROOT/tools"

# Check if virtual environment exists
if [ ! -d "$VENV_PATH" ]; then
    echo -e "${YELLOW}Virtual environment not found. Creating with Python 3.12...${NC}"
    python3.12 -m venv "$VENV_PATH"
    echo -e "${GREEN}✓ Virtual environment created${NC}\n"
fi

# Activate virtual environment
echo -e "${YELLOW}Activating virtual environment...${NC}"
source "$VENV_PATH/bin/activate"
echo -e "${GREEN}✓ Virtual environment activated${NC}\n"

# Install dependencies
echo -e "${YELLOW}Installing Python dependencies...${NC}"
pip install --upgrade pip --no-cache-dir > /dev/null 2>&1
pip install --no-cache-dir -r "$TOOLS_DIR/requirements.txt" > /dev/null 2>&1
echo -e "${GREEN}✓ Dependencies installed${NC}\n"

# Run database migrations
echo -e "${YELLOW}Setting up database...${NC}"
cd "$TOOLS_DIR"

python3.12 << 'EOF'
import sys
from app import app, db
from migrations import init_db

try:
    with app.app_context():
        init_db()
    print("\n")
    sys.exit(0)
except Exception as e:
    print(f"\033[0;31m✗ Database setup failed: {e}\033[0m\n")
    sys.exit(1)
EOF

if [ $? -eq 0 ]; then
    echo -e "${GREEN}✓ Database setup completed successfully${NC}\n"
    echo -e "${GREEN}========================================${NC}"
    echo -e "${GREEN}Database is ready! You can now start the application.${NC}"
    echo -e "${GREEN}Run: ./start.sh${NC}"
    echo -e "${GREEN}========================================${NC}\n"
else
    echo -e "${RED}✗ Database setup failed${NC}\n"
    exit 1
fi
