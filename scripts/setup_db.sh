#!/bin/bash

###############################################################################
# Database Setup Script
# 
# This script initializes the environment after deployment:
# 1. Creates and activates Python virtual environment
# 2. Installs all dependencies
# 3. Initializes database (if migrations exist)
# 4. Validates configuration files
# 
# Usage: ./scripts/setup_db.sh
# 
# This should be run once after cloning the repository and configuring
# .env files (or before running start.sh for the first time)
###############################################################################

set -e  # Exit on error

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
PROJECT_ROOT="$( cd "$SCRIPT_DIR/.." && pwd )"
# shellcheck source=scripts/lib/common.sh
source "$SCRIPT_DIR/lib/common.sh"

print_banner "Enable Agents - Database Setup"

# Paths
VENV_PATH="$PROJECT_ROOT/venv"
TOOLS_DIR="$PROJECT_ROOT/backend"
APP_DIR="$PROJECT_ROOT/frontend"
LOG_DIR="$PROJECT_ROOT/.logs"

# Create log directory
if [ ! -d "$LOG_DIR" ]; then
    mkdir -p "$LOG_DIR"
    echo -e "${GREEN}✓ Created logs directory${NC}\n"
fi

# Check if virtual environment exists
echo -e "${BLUE}[1/5] Setting up Python virtual environment...${NC}"
if [ ! -d "$VENV_PATH" ]; then
    echo -e "${BLUE}   Creating new environment...${NC}"
    python3 -m venv "$VENV_PATH"
    echo -e "${GREEN}✓ Virtual environment created${NC}"
else
    echo -e "${GREEN}✓ Virtual environment exists${NC}"
fi

echo

# Activate virtual environment
echo -e "${BLUE}[2/5] Activating Python environment...${NC}"
source "$VENV_PATH/bin/activate"
echo -e "${GREEN}✓ Environment activated${NC}\n"

# Install dependencies
echo -e "${BLUE}[3/5] Installing Python dependencies...${NC}"
cd "$TOOLS_DIR"

pip install --upgrade pip --no-cache-dir > /dev/null
echo -e "${BLUE}   pip upgraded${NC}"

if [ -f "requirements.txt" ]; then
    pip install --no-cache-dir -r requirements.txt
    echo -e "${GREEN}✓ Python dependencies installed${NC}"
else
    echo -e "${RED}✗ requirements.txt not found in $TOOLS_DIR${NC}"
    exit 1
fi

echo

# Run database migrations if they exist
echo -e "${BLUE}[4/5] Checking for database migrations...${NC}"
if [ -f "$TOOLS_DIR/migrations.py" ]; then
    echo -e "${BLUE}   Running migrations...${NC}"
    python migrations.py
    echo -e "${GREEN}✓ Database migrations completed${NC}"
else
    echo -e "${YELLOW}~ No migrations.py found (skipping)${NC}"
fi

echo

# Validate configuration
echo -e "${BLUE}[5/5] Validating configuration files...${NC}"
validate_local_env "$TOOLS_DIR" "$APP_DIR" || exit 1
echo -e "${GREEN}✓ Configuration files validated${NC}\n"

print_success "Setup completed successfully!"

# Get environment info
ENVIRONMENT=$(grep "^ENVIRONMENT=" "$TOOLS_DIR/.env" | cut -d'=' -f2 || echo "development")
PUBLIC_URL=$(grep "^PUBLIC_URL=" "$TOOLS_DIR/.env" | cut -d'=' -f2)
API_URL=$(grep "^REACT_APP_API_URL=" "$APP_DIR/.env" | cut -d'=' -f2)

echo -e "${BLUE}Configuration:${NC}"
echo -e "  Environment: $ENVIRONMENT"
echo -e "  Backend URL: $PUBLIC_URL"
echo -e "  API URL:     $API_URL\n"

echo -e "${BLUE}Next steps:${NC}"
echo -e "  Start services:    ${YELLOW}./scripts/start.sh${NC}"
if [ "$ENVIRONMENT" = "development" ]; then
    echo -e "  View frontend:     http://localhost:3000"
    echo -e "  View backend:      http://localhost:5000"
else
    echo -e "  View frontend:     http://agents.enableyou.co"
    echo -e "  View backend:      $PUBLIC_URL"
fi
echo -e "  View logs:         ${YELLOW}tail -f .logs/python.log${NC}\n"
echo -e "${YELLOW}Setting up database...${NC}"
cd "$TOOLS_DIR"

python3 << 'EOF'
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
    echo -e "${GREEN}Run: ./scripts/start.sh${NC}"
    echo -e "${GREEN}========================================${NC}\n"
else
    echo -e "${RED}✗ Database setup failed${NC}\n"
    exit 1
fi
