#!/bin/bash

###############################################################################
# Start Script
# 
# This script starts both the React frontend and Python backend services.
#
# Usage: ./start.sh
# 
# The script:
# 1. Activates the Python virtual environment
# 2. Installs React dependencies (if needed)
# 3. Starts the React app (npm start) on port 3000
# 4. Starts the Python backend (python app.py) on port 5000
# 5. Creates log files in .logs/ directory
# 6. Saves process IDs for the stop script
###############################################################################

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Get the directory where the script is located
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
PROJECT_ROOT="$SCRIPT_DIR"

# Paths
VENV_PATH="$PROJECT_ROOT/venv"
TOOLS_DIR="$PROJECT_ROOT/tools"
APP_DIR="$PROJECT_ROOT/agent-app"
LOG_DIR="$PROJECT_ROOT/.logs"
PID_FILE="$PROJECT_ROOT/.pids"

echo -e "${YELLOW}========================================${NC}"
echo -e "${YELLOW}Enable Agents - Restarting Services${NC}"
echo -e "${YELLOW}========================================${NC}\n"

# Validate configuration files exist
echo -e "${BLUE}Validating configuration...${NC}"

if [ ! -f "$TOOLS_DIR/.env" ]; then
    echo -e "${RED}✗ Missing: $TOOLS_DIR/.env${NC}"
    echo -e "${RED}Please copy from .env.example and configure for your environment${NC}\n"
    exit 1
fi

if [ ! -f "$APP_DIR/.env" ]; then
    echo -e "${RED}✗ Missing: $APP_DIR/.env${NC}"
    echo -e "${RED}Please copy from .env.example and configure for your environment${NC}\n"
    exit 1
fi

# Check that PUBLIC_URL is set in backend .env
if ! grep -q "^PUBLIC_URL=" "$TOOLS_DIR/.env"; then
    echo -e "${RED}✗ PUBLIC_URL not set in $TOOLS_DIR/.env${NC}"
    echo -e "${RED}Add: PUBLIC_URL=http://localhost:5000 (or your remote URL)${NC}\n"
    exit 1
fi

# Check that REACT_APP_API_URL is set in frontend .env
if ! grep -q "^REACT_APP_API_URL=" "$APP_DIR/.env"; then
    echo -e "${RED}✗ REACT_APP_API_URL not set in $APP_DIR/.env${NC}"
    echo -e "${RED}Add: REACT_APP_API_URL=http://localhost:5000 (or your remote URL)${NC}\n"
    exit 1
fi

echo -e "${GREEN}✓ Configuration files validated${NC}\n"

# Kill any existing processes on ports 3000 and 5000
echo -e "${BLUE}Cleaning up existing processes on ports 3000 and 5000...${NC}"

# Kill ALL app.py processes regardless of path
pkill -9 -f "app\.py" 2>/dev/null || true
sleep 1

# Kill all npm/node processes
pkill -9 -f "npm.*start" 2>/dev/null || true
pkill -9 -f "node.*3000" 2>/dev/null || true
sleep 1

echo -e "${GREEN}✓ All processes killed${NC}\n"

# Check if virtual environment exists
if [ ! -d "$VENV_PATH" ]; then
    echo -e "${RED}✗ Virtual environment not found at $VENV_PATH${NC}"
    echo -e "${RED}Please run ./setup_db.sh first${NC}\n"
    exit 1
fi

# Check if .logs directory exists, create if not
if [ ! -d "$LOG_DIR" ]; then
    mkdir -p "$LOG_DIR"
    echo -e "${BLUE}✓ Created logs directory${NC}"
fi

# Check if .pids file exists, remove if it does
if [ -f "$PID_FILE" ]; then
    rm -f "$PID_FILE"
fi

# Activate virtual environment
echo -e "${BLUE}Activating Python virtual environment...${NC}"
source "$VENV_PATH/bin/activate"
if [ $? -ne 0 ]; then
    echo -e "${RED}✗ Failed to activate virtual environment${NC}\n"
    exit 1
fi
echo -e "${GREEN}✓ Virtual environment activated${NC}\n"

# Start Python backend
echo -e "${BLUE}Starting Python backend on port 5000...${NC}"
cd "$TOOLS_DIR"
nohup python app.py > "$LOG_DIR/python.log" 2>&1 &
BACKEND_PID=$!
sleep 2

# Check if backend started successfully
if ! kill -0 $BACKEND_PID 2>/dev/null; then
    echo -e "${RED}✗ Failed to start Python backend${NC}"
    echo -e "${RED}Check logs at: $LOG_DIR/python.log${NC}\n"
    exit 1
fi

echo -e "${GREEN}✓ Backend started (PID: $BACKEND_PID)${NC}"
echo "$BACKEND_PID" >> "$PID_FILE"

# Install React dependencies (if needed)
echo -e "${BLUE}Checking React dependencies...${NC}"
cd "$APP_DIR"
if [ ! -d "node_modules" ]; then
    echo -e "${BLUE}Installing React dependencies...${NC}"
    npm install -q
    if [ $? -ne 0 ]; then
        echo -e "${RED}✗ Failed to install React dependencies${NC}\n"
        kill $BACKEND_PID
        exit 1
    fi
    echo -e "${GREEN}✓ React dependencies installed${NC}"
else
    echo -e "${GREEN}✓ React dependencies already installed${NC}"
fi

# Start React frontend
echo -e "${BLUE}Starting React frontend on port 3000...${NC}"
cd "$APP_DIR"
nohup npm start > "$LOG_DIR/react.log" 2>&1 &
FRONTEND_PID=$!
sleep 3

# Check if frontend started successfully
if ! kill -0 $FRONTEND_PID 2>/dev/null; then
    echo -e "${RED}✗ Failed to start React frontend${NC}"
    echo -e "${RED}Check logs at: $LOG_DIR/react.log${NC}\n"
    kill $BACKEND_PID
    exit 1
fi

echo -e "${GREEN}✓ Frontend started (PID: $FRONTEND_PID)${NC}"
echo "$FRONTEND_PID" >> "$PID_FILE"

# Final status
echo -e "\n${GREEN}========================================${NC}"
echo -e "${GREEN}✓ All services started successfully!${NC}"
echo -e "${GREEN}========================================${NC}\n"

# Print configuration
PUBLIC_URL=$(grep "^PUBLIC_URL=" "$TOOLS_DIR/.env" | cut -d'=' -f2)
API_URL=$(grep "^REACT_APP_API_URL=" "$APP_DIR/.env" | cut -d'=' -f2)
ENVIRONMENT=$(grep "^ENVIRONMENT=" "$TOOLS_DIR/.env" | cut -d'=' -f2 || echo "development")

echo -e "${BLUE}CONFIGURATION:${NC}"
echo -e "  Environment:    ${YELLOW}$ENVIRONMENT${NC}"
echo -e "  Backend URL:    ${YELLOW}$PUBLIC_URL${NC}"
echo -e "  Frontend API:   ${YELLOW}$API_URL${NC}\n"

echo -e "${BLUE}Services running:${NC}"
echo -e "  Frontend: ${GREEN}$API_URL${NC} (port 3000)"
echo -e "  Backend:  ${GREEN}$PUBLIC_URL${NC} (port 5000)\n"

echo -e "${BLUE}Endpoints:${NC}"
echo -e "  Google Auth: ${GREEN}$PUBLIC_URL/auth/google/start${NC}"
echo -e "  API Health:  ${GREEN}$PUBLIC_URL/health${NC} (if implemented)\n"

echo -e "${BLUE}Log files:${NC}"
echo -e "  React:   $LOG_DIR/react.log"
echo -e "  Python:  $LOG_DIR/python.log\n"

echo -e "${BLUE}To view logs in real-time:${NC}"
echo -e "  ${YELLOW}tail -f $LOG_DIR/react.log${NC}"
echo -e "  ${YELLOW}tail -f $LOG_DIR/python.log${NC}\n"

echo -e "${BLUE}Troubleshooting:${NC}"
echo -e "  View Python errors:    ${YELLOW}cat $LOG_DIR/python.log${NC}"
echo -e "  View React errors:     ${YELLOW}cat $LOG_DIR/react.log${NC}"
echo -e "  Check if running:      ${YELLOW}lsof -i :5000${NC}"
echo -e "  Stop services:         ${YELLOW}./stop.sh${NC}\n"

echo -e "${BLUE}To stop services:${NC}"
echo -e "  ${YELLOW}./stop.sh${NC}\n"

# Keep running (optional - you can remove this if you want the script to exit)
wait
