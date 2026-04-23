#!/bin/bash

###############################################################################
# Stop Script
# 
# This script stops both the React frontend and Python backend services
# that were started with the start.sh script.
#
# Usage: ./stop.sh
# 
# The script:
# 1. Reads the saved PID information
# 2. Gracefully terminates both services
# 3. Verifies they have stopped
# 4. Cleans up PID file
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
PID_FILE="$PROJECT_ROOT/.pids"
LOG_DIR="$PROJECT_ROOT/.logs"

echo -e "${YELLOW}========================================${NC}"
echo -e "${YELLOW}Enable Agents - Stopping Services${NC}"
echo -e "${YELLOW}========================================${NC}\n"

# Function to kill process gracefully
kill_process() {
    local PID=$1
    local NAME=$2
    
    if kill -0 "$PID" 2>/dev/null; then
        echo -e "${BLUE}Stopping $NAME (PID: $PID)...${NC}"
        
        # First, try graceful termination
        kill -TERM "$PID" 2>/dev/null || true
        
        # Wait up to 5 seconds for graceful shutdown
        for i in {1..50}; do
            if ! kill -0 "$PID" 2>/dev/null; then
                echo -e "${GREEN}✓ $NAME stopped successfully${NC}"
                return 0
            fi
            sleep 0.1
        done
        
        # If still running, force kill
        if kill -0 "$PID" 2>/dev/null; then
            echo -e "${YELLOW}Force stopping $NAME (PID: $PID)...${NC}"
            kill -9 "$PID" 2>/dev/null || true
            echo -e "${GREEN}✓ $NAME force stopped${NC}"
            return 0
        fi
    fi
    return 0
}

STOPPED=0

# Try from PID file first
if [ -f "$PID_FILE" ]; then
    PIDS=($(cat "$PID_FILE"))
    
    if [ ${#PIDS[@]} -gt 0 ]; then
        for PID in "${PIDS[@]}"; do
            kill_process "$PID" "Process"
            ((STOPPED++))
        done
    fi
    rm -f "$PID_FILE"
fi

# Fallback: Kill by process name if any are still running
echo -e "${BLUE}Searching for React and Python app processes...${NC}"

# Kill React (npm start)
REACT_PIDS=$(pgrep -f "npm start" | grep -v grep)
if [ ! -z "$REACT_PIDS" ]; then
    echo "$REACT_PIDS" | while read PID; do
        kill_process "$PID" "React (npm start)"
        ((STOPPED++))
    done
fi

# Kill Python app.py
PYTHON_PIDS=$(pgrep -f "python.*app.py")
if [ ! -z "$PYTHON_PIDS" ]; then
    echo "$PYTHON_PIDS" | while read PID; do
        kill_process "$PID" "Python backend (app.py)"
        ((STOPPED++))
    done
fi

echo -e "\n${GREEN}========================================${NC}"
echo -e "${GREEN}✓ Services stopped successfully!${NC}"
echo -e "${GREEN}========================================${NC}\n"

if [ -d "$LOG_DIR" ]; then
    echo -e "${BLUE}Log files saved in: $LOG_DIR${NC}"
    echo -e "${BLUE}View logs with: tail -f $LOG_DIR/{react,python}.log${NC}\n"
fi
