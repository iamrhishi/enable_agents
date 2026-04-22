#!/bin/bash

###############################################################################
# Start Script
# 
# This script starts backend and frontend services.
#
# Usage: ./start.sh
# 
# For PRODUCTION:
# 1. Starts Python backend (app.py) on port 5000
# 2. Builds React frontend
# 3. Starts nginx proxy on port 80 (serves React + proxies API)
# 4. Verifies all services are responsive before claiming success
#
# For DEVELOPMENT (localhost):
# 1. Starts Python backend on port 5000
# 2. Starts React dev server on port 3000
# 3. Verifies services are responsive
###############################################################################

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

set -e  # Exit on any error

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
echo -e "${YELLOW}Enable Agents - Starting Services${NC}"
echo -e "${YELLOW}========================================${NC}\n"

# Validate configuration files exist
echo -e "${BLUE}[1/7] Validating configuration...${NC}"

if [ ! -f "$TOOLS_DIR/.env" ]; then
    echo -e "${RED}✗ Missing: $TOOLS_DIR/.env${NC}"
    echo -e "${RED}Create one based on .env.production or .env.development${NC}\n"
    exit 1
fi

if [ ! -f "$APP_DIR/.env" ]; then
    echo -e "${RED}✗ Missing: $APP_DIR/.env${NC}"
    echo -e "${RED}Create one based on .env.production or .env.development${NC}\n"
    exit 1
fi

# Check that PUBLIC_URL is set in backend .env
if ! grep -q "^PUBLIC_URL=" "$TOOLS_DIR/.env"; then
    echo -e "${RED}✗ PUBLIC_URL not set in $TOOLS_DIR/.env${NC}"
    exit 1
fi

# Check that REACT_APP_API_URL is set in frontend .env
if ! grep -q "^REACT_APP_API_URL=" "$APP_DIR/.env"; then
    echo -e "${RED}✗ REACT_APP_API_URL not set in $APP_DIR/.env${NC}"
    exit 1
fi

# Get environment
ENVIRONMENT=$(grep "^ENVIRONMENT=" "$TOOLS_DIR/.env" | cut -d'=' -f2 || echo "development")
PUBLIC_URL=$(grep "^PUBLIC_URL=" "$TOOLS_DIR/.env" | cut -d'=' -f2)
API_URL=$(grep "^REACT_APP_API_URL=" "$APP_DIR/.env" | cut -d'=' -f2)

echo -e "${GREEN}✓ Configuration valid${NC}"
echo -e "  Environment: $ENVIRONMENT"
echo -e "  Backend URL: $PUBLIC_URL"
echo -e "  API URL:     $API_URL\n"

# Kill any existing processes
echo -e "${BLUE}[2/7] Stopping any existing services...${NC}"

pkill -9 -f "app\.py" 2>/dev/null || true
pkill -9 -f "npm.*start" 2>/dev/null || true
pkill -9 -f "nginx" 2>/dev/null || true
sleep 1

echo -e "${GREEN}✓ Old processes cleaned up${NC}\n"

# Check if virtual environment exists
if [ ! -d "$VENV_PATH" ]; then
    echo -e "${RED}✗ Virtual environment not found${NC}"
    echo -e "${RED}Run: ./setup_db.sh${NC}\n"
    exit 1
fi

# Create log directory
if [ ! -d "$LOG_DIR" ]; then
    mkdir -p "$LOG_DIR"
fi

if [ -f "$PID_FILE" ]; then
    rm -f "$PID_FILE"
fi

# Activate virtual environment
echo -e "${BLUE}[3/7] Activating Python environment...${NC}"
source "$VENV_PATH/bin/activate"
echo -e "${GREEN}✓ Environment activated${NC}\n"

# Start Python backend
echo -e "${BLUE}[4/7] Starting backend on port 5000...${NC}"
cd "$TOOLS_DIR"
nohup python app.py > "$LOG_DIR/python.log" 2>&1 &
BACKEND_PID=$!
sleep 2

# Verify backend is running
if ! kill -0 $BACKEND_PID 2>/dev/null; then
    echo -e "${RED}✗ Backend failed to start${NC}"
    echo -e "${RED}Errors:${NC}"
    tail -20 "$LOG_DIR/python.log"
    exit 1
fi

echo -e "${GREEN}✓ Backend started (PID: $BACKEND_PID)${NC}"
echo "$BACKEND_PID" >> "$PID_FILE"

# Wait for backend to be responsive
echo -e "${BLUE}   Waiting for backend to be responsive...${NC}"
for i in {1..30}; do
    if curl -s http://localhost:5000/ > /dev/null 2>&1; then
        echo -e "${GREEN}✓ Backend is responsive${NC}"
        break
    fi
    if [ $i -eq 30 ]; then
        echo -e "${RED}✗ Backend not responsive after 30s${NC}"
        tail -20 "$LOG_DIR/python.log"
        kill $BACKEND_PID
        exit 1
    fi
    sleep 1
done
echo

# Handle production vs development
if [ "$ENVIRONMENT" = "production" ]; then
    echo -e "${BLUE}[5/7] Building React for production...${NC}"
    cd "$APP_DIR"
    
    if [ ! -d "node_modules" ]; then
        echo -e "${BLUE}   Installing dependencies...${NC}"
        npm install -q
    fi
    
    npm run build -q
    if [ $? -ne 0 ]; then
        echo -e "${RED}✗ React build failed${NC}"
        kill $BACKEND_PID
        exit 1
    fi
    
    echo -e "${GREEN}✓ React built successfully${NC}\n"
    
    # Deploy to nginx
    echo -e "${BLUE}[6/7] Starting nginx reverse proxy (port 80)...${NC}"
    
    if [ ! -d "/usr/share/nginx/html" ]; then
        sudo mkdir -p /usr/share/nginx/html
    fi
    
    sudo cp -r "$APP_DIR/build"/* /usr/share/nginx/html/ 2>/dev/null || true
    sudo cp "$APP_DIR/nginx.conf" /etc/nginx/sites-enabled/agents.conf 2>/dev/null || true
    
    sudo nginx -s reload 2>/dev/null || sudo systemctl restart nginx 2>/dev/null || true
    sleep 1
    
    echo -e "${GREEN}✓ Nginx started${NC}\n"
    
    # Verify nginx is running
    echo -e "${BLUE}[7/7] Verifying all services...${NC}"
    
    NGINX_OK=0
    for i in {1..10}; do
        if curl -s http://localhost/ > /dev/null 2>&1; then
            NGINX_OK=1
            break
        fi
        sleep 1
    done
    
    if [ $NGINX_OK -ne 1 ]; then
        echo -e "${RED}✗ Nginx not responding${NC}"
        kill $BACKEND_PID
        exit 1
    fi
    
    echo -e "${GREEN}✓ Nginx is responsive${NC}\n"
    
else
    # Development: start npm dev server
    echo -e "${BLUE}[5/7] Checking React dependencies...${NC}"
    cd "$APP_DIR"
    
    if [ ! -d "node_modules" ]; then
        echo -e "${BLUE}   Installing dependencies...${NC}"
        npm install -q
    fi
    
    echo -e "${GREEN}✓ Dependencies ready${NC}\n"
    
    echo -e "${BLUE}[6/7] Starting React dev server on port 3000...${NC}"
    nohup npm start > "$LOG_DIR/react.log" 2>&1 &
    FRONTEND_PID=$!
    sleep 3
    
    if ! kill -0 $FRONTEND_PID 2>/dev/null; then
        echo -e "${RED}✗ React dev server failed${NC}"
        echo -e "${RED}Errors:${NC}"
        tail -20 "$LOG_DIR/react.log"
        kill $BACKEND_PID
        exit 1
    fi
    
    echo -e "${GREEN}✓ React dev server started (PID: $FRONTEND_PID)${NC}"
    echo "$FRONTEND_PID" >> "$PID_FILE"
    
    # Verify React is responsive
    echo -e "${BLUE}[7/7] Verifying React responsiveness...${NC}"
    for i in {1..30}; do
        if curl -s http://localhost:3000/ > /dev/null 2>&1; then
            echo -e "${GREEN}✓ React is responsive${NC}\n"
            break
        fi
        if [ $i -eq 30 ]; then
            echo -e "${RED}✗ React not responsive after 30s${NC}"
            tail -20 "$LOG_DIR/react.log"
            kill $BACKEND_PID $FRONTEND_PID
            exit 1
        fi
        sleep 1
    done
fi

# Final status
echo -e "${GREEN}========================================${NC}"
echo -e "${GREEN}✓ All services started successfully!${NC}"
echo -e "${GREEN}========================================${NC}\n"

echo -e "${BLUE}CONFIGURATION:${NC}"
echo -e "  Environment:     ${YELLOW}$ENVIRONMENT${NC}"
echo -e "  Backend API:     ${YELLOW}$PUBLIC_URL${NC}"
echo -e "  Frontend API:    ${YELLOW}$API_URL${NC}\n"

if [ "$ENVIRONMENT" = "production" ]; then
    echo -e "${BLUE}Services running:${NC}"
    echo -e "  Nginx (frontend + proxy): http://agents.enableyou.co (port 80)"
    echo -e "  Backend:                  $PUBLIC_URL (port 5000)\n"
    
    echo -e "${BLUE}Test endpoints:${NC}"
    echo -e "  Frontend: ${GREEN}curl http://agents.enableyou.co${NC}"
    echo -e "  Backend:  ${GREEN}curl $PUBLIC_URL${NC}"
    echo -e "  OAuth:    ${GREEN}curl $PUBLIC_URL/auth/google/start${NC}\n"
else
    echo -e "${BLUE}Services running:${NC}"
    echo -e "  React dev server: http://localhost:3000"
    echo -e "  Backend:          http://localhost:5000\n"
    
    echo -e "${BLUE}Test endpoints:${NC}"
    echo -e "  Frontend: ${GREEN}curl http://localhost:3000${NC}"
    echo -e "  Backend:  ${GREEN}curl http://localhost:5000${NC}"
    echo -e "  OAuth:    ${GREEN}curl http://localhost:5000/auth/google/start${NC}\n"
fi

echo -e "${BLUE}Log files:${NC}"
echo -e "  Backend:  $LOG_DIR/python.log"
if [ "$ENVIRONMENT" != "production" ]; then
    echo -e "  Frontend: $LOG_DIR/react.log"
fi
echo

echo -e "${BLUE}Commands:${NC}"
echo -e "  View backend logs:    ${YELLOW}tail -f $LOG_DIR/python.log${NC}"
if [ "$ENVIRONMENT" != "production" ]; then
    echo -e "  View frontend logs:   ${YELLOW}tail -f $LOG_DIR/react.log${NC}"
fi
echo -e "  Stop services:        ${YELLOW}./stop.sh${NC}"
echo -e "  Restart services:     ${YELLOW}./start.sh${NC}\n"

echo -e "${BLUE}To stop services:${NC}"
echo -e "  ${YELLOW}./stop.sh${NC}\n"

# Keep running (optional - you can remove this if you want the script to exit)
wait
