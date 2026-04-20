#!/bin/bash

###############################################################################
# Start Script
# 
# This script starts both the React frontend and Python backend in the 
# background. Both services will run continuously until stopped.
#
# Usage: ./start.sh
# 
# The script:
# 1. Creates log files for both services
# 2. Activates the Python virtual environment
# 3. Starts the React app (npm start) in background
# 4. Starts the Python app (python app.py) in background
# 5. Saves PID information for the stop script to use
###############################################################################

set -e  # Exit on error

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
AGENT_APP_DIR="$PROJECT_ROOT/agent-app"
TOOLS_DIR="$PROJECT_ROOT/tools"
LOG_DIR="$PROJECT_ROOT/.logs"
PID_FILE="$PROJECT_ROOT/.pids"

# Create logs directory if it doesn't exist
mkdir -p "$LOG_DIR"

# Initialize PID file (empty)
> "$PID_FILE"

echo -e "${YELLOW}========================================${NC}"
echo -e "${YELLOW}Enable Agents - Starting Services${NC}"
echo -e "${YELLOW}========================================${NC}\n"

# Check if virtual environment exists
if [ ! -d "$VENV_PATH" ]; then
    echo -e "${RED}✗ Virtual environment not found${NC}"
    echo -e "${YELLOW}Please run ./setup_db.sh first${NC}\n"
    exit 1
fi

# Activate virtual environment
source "$VENV_PATH/bin/activate"

# Check if React app dependencies are installed
if [ ! -d "$AGENT_APP_DIR/node_modules" ]; then
    echo -e "${YELLOW}Installing React dependencies...${NC}"
    cd "$AGENT_APP_DIR"
    npm install > "$LOG_DIR/npm-install.log" 2>&1
    echo -e "${GREEN}✓ React dependencies installed${NC}\n"
fi

# Start React App
echo -e "${BLUE}Starting React frontend...${NC}"
cd "$AGENT_APP_DIR"
npm start > "$LOG_DIR/react.log" 2>&1 &
REACT_PID=$!
echo $REACT_PID >> "$PID_FILE"
echo -e "${GREEN}✓ React app started (PID: $REACT_PID)${NC}"
echo -e "${BLUE}  Log file: $LOG_DIR/react.log${NC}\n"

# Wait a moment for React to start
sleep 3

# Start Python App
echo -e "${BLUE}Starting Python backend...${NC}"
cd "$TOOLS_DIR"
python3 app.py > "$LOG_DIR/python.log" 2>&1 &
PYTHON_PID=$!
echo $PYTHON_PID >> "$PID_FILE"
echo -e "${GREEN}✓ Python app started (PID: $PYTHON_PID)${NC}"
echo -e "${BLUE}  Log file: $LOG_DIR/python.log${NC}\n"

# Wait a moment for Python to start
sleep 2

# Verify both processes are running
if kill -0 $REACT_PID 2>/dev/null && kill -0 $PYTHON_PID 2>/dev/null; then
    echo -e "${GREEN}========================================${NC}"
    echo -e "${GREEN}✓ All services started successfully!${NC}"
    echo -e "${GREEN}========================================${NC}\n"
    
    echo -e "${BLUE}Frontend:${NC}"
    echo -e "  URL: ${YELLOW}http://localhost:3000${NC}"
    echo -e "  Log: ${YELLOW}$LOG_DIR/react.log${NC}\n"
    
    echo -e "${BLUE}Backend:${NC}"
    echo -e "  URL: ${YELLOW}http://localhost:5000${NC}"
    echo -e "  Log: ${YELLOW}$LOG_DIR/python.log${NC}\n"
    
    echo -e "${YELLOW}To view logs:${NC}"
    echo -e "  tail -f $LOG_DIR/react.log"
    echo -e "  tail -f $LOG_DIR/python.log\n"
    
    echo -e "${YELLOW}To stop services:${NC}"
    echo -e "  ./stop.sh\n"
else
    echo -e "${RED}✗ Failed to start services${NC}\n"
    kill $REACT_PID 2>/dev/null || true
    kill $PYTHON_PID 2>/dev/null || true
    exit 1
fi
