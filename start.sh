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

# Kill any existing processes on ports 3000 and 5000
echo -e "${BLUE}Cleaning up existing processes on ports 3000 and 5000...${NC}"

# Function to kill process on a specific port
kill_port() {
    local PORT=$1
    local PID=$(lsof -ti:$PORT 2>/dev/null)
    
    if [ ! -z "$PID" ]; then
        echo -e "${YELLOW}Found process on port $PORT (PID: $PID), terminating...${NC}"
        kill -TERM $PID 2>/dev/null || true
        
        # Wait up to 3 seconds for graceful shutdown
        for i in {1..30}; do
            if ! kill -0 $PID 2>/dev/null; then
                echo -e "${GREEN}✓ Process on port $PORT terminated${NC}"
                return 0
            fi
            sleep 0.1
        done
        
        # Force kill if still running
        if kill -0 $PID 2>/dev/null; then
            echo -e "${YELLOW}Force killing process on port $PORT${NC}"
            kill -9 $PID 2>/dev/null || true
            echo -e "${GREEN}✓ Process on port $PORT force killed${NC}"
        fi
    else
        echo -e "${GREEN}✓ Port $PORT is clean${NC}"
    fi
}

# Kill processes on both ports
kill_port 3000
kill_port 5000

echo -e "${GREEN}✓ Ports cleaned${NC}\n"

# Check if .logs directory exists, create if not
if [ ! -d "$LOG_DIR" ]; then
    mkdir -p "$LOG_DIR"
    echo -e "${BLUE}✓ Created logs directory${NC}"
fi

# Check if .pids file exists, remove if it does
if [ -f "$PID_FILE" ]; then
    rm -f "$PID_FILE"
fi

# Check if virtual environment exists, create if not
if [ ! -d "$VENV_PATH" ]; then
    echo -e "${BLUE}Creating Python virtual environment...${NC}"
    python3 -m venv "$VENV_PATH"
    if [ $? -ne 0 ]; then
        echo -e "${RED}✗ Failed to create virtual environment${NC}\n"
        exit 1
    fi
    echo -e "${GREEN}✓ Virtual environment created${NC}"
fi

# Activate virtual environment
echo -e "${BLUE}Activating Python virtual environment...${NC}"
source "$VENV_PATH/bin/activate"
if [ $? -ne 0 ]; then
    echo -e "${RED}✗ Failed to activate virtual environment${NC}\n"
    exit 1
fi
echo -e "${GREEN}✓ Virtual environment activated${NC}"

# Install Python dependencies
echo -e "${BLUE}Installing Python dependencies...${NC}"
cd "$TOOLS_DIR"
pip install --upgrade pip --no-cache-dir -q
pip install --no-cache-dir -r requirements.txt
if [ $? -ne 0 ]; then
    echo -e "${RED}✗ Failed to install Python dependencies${NC}\n"
    exit 1
fi
echo -e "${GREEN}✓ Python dependencies installed${NC}"

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
export REACT_APP_API_URL=http://localhost:5000
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

echo -e "${BLUE}Services running:${NC}"
echo -e "  Frontend: ${GREEN}http://localhost:3000${NC}"
echo -e "  Backend:  ${GREEN}http://localhost:5000${NC}\n"

echo -e "${BLUE}Log files:${NC}"
echo -e "  React:   $LOG_DIR/react.log"
echo -e "  Python:  $LOG_DIR/python.log\n"

echo -e "${BLUE}To view logs in real-time:${NC}"
echo -e "  ${YELLOW}tail -f $LOG_DIR/react.log${NC}"
echo -e "  ${YELLOW}tail -f $LOG_DIR/python.log${NC}\n"

echo -e "${BLUE}To stop services:${NC}"
echo -e "  ${YELLOW}./stop.sh${NC}\n"

# Keep running (optional - you can remove this if you want the script to exit)
wait
