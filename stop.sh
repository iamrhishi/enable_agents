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

# Check if PID file exists
if [ ! -f "$PID_FILE" ]; then
    echo -e "${RED}✗ No running services found (PID file not found)${NC}\n"
    exit 0
fi

# Read PIDs from file
PIDS=($(cat "$PID_FILE"))

if [ ${#PIDS[@]} -eq 0 ]; then
    echo -e "${RED}✗ No running services found (empty PID file)${NC}\n"
    rm -f "$PID_FILE"
    exit 0
fi

# Try to stop each process gracefully
STOPPED=0
FAILED=0

for PID in "${PIDS[@]}"; do
    if kill -0 "$PID" 2>/dev/null; then
        echo -e "${BLUE}Stopping process (PID: $PID)...${NC}"
        
        # First, try graceful termination
        kill -TERM "$PID" 2>/dev/null || true
        
        # Wait up to 5 seconds for graceful shutdown
        for i in {1..50}; do
            if ! kill -0 "$PID" 2>/dev/null; then
                echo -e "${GREEN}✓ Process stopped successfully${NC}"
                ((STOPPED++))
                break
            fi
            sleep 0.1
        done
        
        # If still running, force kill
        if kill -0 "$PID" 2>/dev/null; then
            echo -e "${YELLOW}Force stopping process (PID: $PID)...${NC}"
            kill -9 "$PID" 2>/dev/null || true
            echo -e "${GREEN}✓ Process force stopped${NC}"
            ((STOPPED++))
        fi
    else
        echo -e "${YELLOW}Process (PID: $PID) not running${NC}"
        ((STOPPED++))
    fi
done

# Clean up
rm -f "$PID_FILE"

echo -e "\n${GREEN}========================================${NC}"
echo -e "${GREEN}✓ All services stopped successfully!${NC}"
echo -e "${GREEN}========================================${NC}\n"

if [ -d "$LOG_DIR" ]; then
    echo -e "${BLUE}Log files saved in: $LOG_DIR${NC}"
    echo -e "${BLUE}View logs with: tail -f $LOG_DIR/{react,python}.log${NC}\n"
fi
