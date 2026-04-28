#!/bin/bash

###############################################################################
# Stop Script
# 
# This script stops all running services started by start.sh
#
# For PRODUCTION: Stops Flask backend and nginx
# For DEVELOPMENT: Stops Flask backend and npm dev server
#
# Usage: ./scripts/stop.sh
###############################################################################

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Get the directory where the script is located
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
PROJECT_ROOT="$( cd "$SCRIPT_DIR/.." && pwd )"

# Paths
PID_FILE="$PROJECT_ROOT/.pids"
LOG_DIR="$PROJECT_ROOT/.logs"
TOOLS_DIR="$PROJECT_ROOT/tools"

echo -e "${YELLOW}========================================${NC}"
echo -e "${YELLOW}Enable Agents - Stopping Services${NC}"
echo -e "${YELLOW}========================================${NC}\n"

# Get environment
ENVIRONMENT="development"
if [ -f "$TOOLS_DIR/.env" ]; then
    ENVIRONMENT=$(grep "^ENVIRONMENT=" "$TOOLS_DIR/.env" | cut -d'=' -f2 || echo "development")
fi

# Stop main processes from PID file
STOPPED=0

if [ -f "$PID_FILE" ]; then
    echo -e "${BLUE}Stopping tracked processes...${NC}"
    
    while IFS= read -r PID; do
        if [ -z "$PID" ]; then
            continue
        fi
        
        if kill -0 "$PID" 2>/dev/null; then
            echo -e "${BLUE}  Stopping PID $PID...${NC}"
            
            # Graceful shutdown
            kill -TERM "$PID" 2>/dev/null || true
            
            # Wait up to 5 seconds
            for i in {1..50}; do
                if ! kill -0 "$PID" 2>/dev/null; then
                    echo -e "${GREEN}  ✓ PID $PID stopped${NC}"
                    ((STOPPED++))
                    break
                fi
                sleep 0.1
            done
            
            # Force kill if still running
            if kill -0 "$PID" 2>/dev/null; then
                echo -e "${YELLOW}  Force stopping PID $PID...${NC}"
                kill -9 "$PID" 2>/dev/null || true
                echo -e "${GREEN}  ✓ PID $PID force stopped${NC}"
                ((STOPPED++))
            fi
        fi
    done < "$PID_FILE"
    
    rm -f "$PID_FILE"
fi

# Stop any remaining Python processes
echo -e "${BLUE}Cleaning up any remaining services...${NC}"

PYTHON_PIDS=$(pgrep -f "app\.py" 2>/dev/null || true)
if [ -n "$PYTHON_PIDS" ]; then
    echo -e "${BLUE}  Stopping backend processes...${NC}"
    pkill -9 -f "app\.py" 2>/dev/null || true
    sleep 1
    echo -e "${GREEN}  ✓ Backend stopped${NC}"
fi

NPM_PIDS=$(pgrep -f "npm.*start" 2>/dev/null || true)
if [ -n "$NPM_PIDS" ]; then
    echo -e "${BLUE}  Stopping React dev server...${NC}"
    pkill -9 -f "npm.*start" 2>/dev/null || true
    sleep 1
    echo -e "${GREEN}  ✓ React dev server stopped${NC}"
fi

# Stop nginx on production
if [ "$ENVIRONMENT" = "production" ]; then
    NGINX_PID=$(pgrep -f "nginx" 2>/dev/null || true)
    if [ -n "$NGINX_PID" ]; then
        echo -e "${BLUE}  Stopping nginx...${NC}"
        sudo nginx -s quit 2>/dev/null || sudo systemctl stop nginx 2>/dev/null || true
        sleep 1
        echo -e "${GREEN}  ✓ Nginx stopped${NC}"
    fi
fi

# Final status
echo -e "\n${GREEN}========================================${NC}"
echo -e "${GREEN}✓ All services stopped!${NC}"
echo -e "${GREEN}========================================${NC}\n"

if [ -d "$LOG_DIR" ]; then
    echo -e "${BLUE}Log files available:${NC}"
    echo -e "  $LOG_DIR/python.log"
    if [ "$ENVIRONMENT" != "production" ]; then
        echo -e "  $LOG_DIR/react.log"
    fi
    echo
fi

echo -e "${BLUE}To view logs:${NC}"
echo -e "  ${YELLOW}tail -f $LOG_DIR/python.log${NC}"
if [ "$ENVIRONMENT" != "production" ]; then
    echo -e "  ${YELLOW}tail -f $LOG_DIR/react.log${NC}"
fi
echo

echo -e "${BLUE}To restart services:${NC}"
echo -e "  ${YELLOW}./scripts/start.sh${NC}\n"
