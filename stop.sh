#!/bin/bash

###############################################################################
# Stop Script - Stops React frontend (port 3000) and Python backend (port 5000)
# Usage: ./stop.sh
###############################################################################

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
PID_FILE="$SCRIPT_DIR/.pids"

echo -e "${YELLOW}========================================${NC}"
echo -e "${YELLOW}  Enable Agents - Stopping Services${NC}"
echo -e "${YELLOW}========================================${NC}\n"

# --- Kill process on a specific port ---
kill_port() {
    local PORT=$1
    local PIDS
    PIDS=$(lsof -ti:"$PORT" 2>/dev/null || true)
    if [ -n "$PIDS" ]; then
        echo -e "${BLUE}Killing process(es) on port $PORT: $PIDS${NC}"
        echo "$PIDS" | xargs kill -9 2>/dev/null || true
        sleep 0.5
        REMAINING=$(lsof -ti:"$PORT" 2>/dev/null || true)
        if [ -z "$REMAINING" ]; then
            echo -e "${GREEN}✓ Port $PORT freed${NC}"
        else
            echo -e "${RED}✗ Port $PORT still in use after kill attempt${NC}"
        fi
    else
        echo -e "${GREEN}✓ Port $PORT already free${NC}"
    fi
}

# --- Kill by PIDs from PID file ---
if [ -f "$PID_FILE" ]; then
    while IFS= read -r PID; do
        if [ -n "$PID" ] && kill -0 "$PID" 2>/dev/null; then
            echo -e "${BLUE}Stopping PID $PID...${NC}"
            kill -TERM "$PID" 2>/dev/null || true
            sleep 0.5
            kill -9 "$PID" 2>/dev/null || true
        fi
    done < "$PID_FILE"
    rm -f "$PID_FILE"
fi

# --- Kill by port (catches any processes not in PID file) ---
echo -e "${BLUE}\nChecking ports...${NC}"
kill_port 3000
kill_port 5000

# --- Kill by process name as final sweep ---
pkill -f "react-scripts start" 2>/dev/null || true
pkill -f "npm start" 2>/dev/null || true
pkill -f "python3\?.*app\.py" 2>/dev/null || true

echo -e "\n${GREEN}========================================${NC}"
echo -e "${GREEN}✓ All services stopped${NC}"
echo -e "${GREEN}========================================${NC}\n"


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
