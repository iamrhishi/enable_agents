#!/bin/bash

###############################################################################
# Start Script - Starts React frontend (port 3000) and Python backend (port 5000)
# Usage: ./start.sh
###############################################################################

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
PROJECT_ROOT="$SCRIPT_DIR"
VENV_PATH="$PROJECT_ROOT/venv"
AGENT_APP_DIR="$PROJECT_ROOT/agent-app"
TOOLS_DIR="$PROJECT_ROOT/tools"
LOG_DIR="$PROJECT_ROOT/.logs"
PID_FILE="$PROJECT_ROOT/.pids"

mkdir -p "$LOG_DIR"
> "$PID_FILE"

echo -e "${YELLOW}========================================${NC}"
echo -e "${YELLOW}  Enable Agents - Starting Services${NC}"
echo -e "${YELLOW}========================================${NC}\n"

# =============================================================================
# PRE-FLIGHT: Check that setup_db.sh tasks have been completed
# =============================================================================
echo -e "${BLUE}--- Pre-flight checks ---${NC}"
SETUP_OK=true

# 1. Virtual environment
if [ ! -d "$VENV_PATH" ]; then
    echo -e "${RED}  ✗ Virtual environment not found at $VENV_PATH${NC}"
    SETUP_OK=false
else
    echo -e "${GREEN}  ✓ Virtual environment exists${NC}"
fi

# 2. Python dependencies (check flask as key package)
if [ -d "$VENV_PATH" ]; then
    if ! "$VENV_PATH/bin/pip" show flask > /dev/null 2>&1; then
        echo -e "${RED}  ✗ Python dependencies not installed (flask not found)${NC}"
        SETUP_OK=false
    else
        echo -e "${GREEN}  ✓ Python dependencies installed${NC}"
    fi
fi

# 3. Database initialised
DB_FILE="$TOOLS_DIR/instance/enable_agents.db"
if [ ! -f "$DB_FILE" ]; then
    echo -e "${RED}  ✗ Database not found at $DB_FILE${NC}"
    SETUP_OK=false
else
    echo -e "${GREEN}  ✓ Database exists${NC}"
fi

# 4. React node_modules (non-fatal — will be installed automatically)
if [ ! -d "$AGENT_APP_DIR/node_modules" ]; then
    echo -e "${YELLOW}  ⚠ React node_modules not found — will install below${NC}"
else
    echo -e "${GREEN}  ✓ React node_modules installed${NC}"
fi

echo ""

if [ "$SETUP_OK" = false ]; then
    echo -e "${RED}========================================${NC}"
    echo -e "${RED}  Setup incomplete. Please run:${NC}"
    echo -e "${RED}    ./setup_db.sh${NC}"
    echo -e "${RED}  then try ./start.sh again.${NC}"
    echo -e "${RED}========================================${NC}\n"
    exit 1
fi

# --- Kill any process on a given port ---
kill_port() {
    local PORT=$1
    local PIDS
    PIDS=$(lsof -ti:"$PORT" 2>/dev/null || true)
    if [ -n "$PIDS" ]; then
        echo -e "${YELLOW}Port $PORT in use — killing existing process(es): $PIDS${NC}"
        echo "$PIDS" | xargs kill -9 2>/dev/null || true
        sleep 1
        # Verify it's gone
        REMAINING=$(lsof -ti:"$PORT" 2>/dev/null || true)
        if [ -n "$REMAINING" ]; then
            echo -e "${RED}✗ Could not free port $PORT. Aborting.${NC}"
            exit 1
        fi
        echo -e "${GREEN}✓ Port $PORT is now free${NC}"
    else
        echo -e "${GREEN}✓ Port $PORT is already free${NC}"
    fi
}

echo -e "${BLUE}Checking ports...${NC}"
kill_port 3000
kill_port 5000
echo ""

source "$VENV_PATH/bin/activate"

# --- Install React deps if needed ---
if [ ! -d "$AGENT_APP_DIR/node_modules" ]; then
    echo -e "${YELLOW}Installing React dependencies...${NC}"
    cd "$AGENT_APP_DIR" && npm install > "$LOG_DIR/npm-install.log" 2>&1
    echo -e "${GREEN}✓ React dependencies installed${NC}\n"
fi

# --- Load environment ---
ENV_FILE="$TOOLS_DIR/.env"
ENVIRONMENT="development"
if [ -f "$ENV_FILE" ]; then
    ENVIRONMENT=$(grep '^ENVIRONMENT=' "$ENV_FILE" | cut -d= -f2 | tr -d '[:space:]')
fi

if [ "$ENVIRONMENT" = "production" ]; then
    # --- Production: build React and serve via nginx ---

    # Clean old build directory (fixes permission errors on rebuild)
    if [ -d "$AGENT_APP_DIR/build" ]; then
        echo -e "${YELLOW}Cleaning old build directory...${NC}"
        chmod -R u+w "$AGENT_APP_DIR/build" 2>/dev/null || true
        rm -rf "$AGENT_APP_DIR/build"
        echo -e "${GREEN}✓ Old build removed${NC}\n"
    fi

    echo -e "${BLUE}Building React for production...${NC}"
    cd "$AGENT_APP_DIR"
    npm run build > "$LOG_DIR/react-build.log" 2>&1
    if [ $? -ne 0 ]; then
        echo -e "${RED}✗ React build failed. See $LOG_DIR/react-build.log${NC}"
        tail -20 "$LOG_DIR/react-build.log"
        exit 1
    fi
    echo -e "${GREEN}✓ React built successfully${NC}\n"

    # Deploy nginx config (remove ALL old configs first to avoid conflicts)
    echo -e "${BLUE}Configuring nginx...${NC}"
    NGINX_CONF="/etc/nginx/sites-enabled/enable_agents"
    sudo rm -f /etc/nginx/sites-enabled/*
    sudo cp "$AGENT_APP_DIR/nginx.conf" "$NGINX_CONF"
    sudo sed -i "s|REACT_BUILD_PATH|$AGENT_APP_DIR/build|g" "$NGINX_CONF"
    sudo nginx -t 2>/dev/null && (sudo nginx -s reload 2>/dev/null || sudo nginx)
    echo -e "${GREEN}✓ Nginx configured and started (port 80)${NC}\n"

    REACT_PID=0
else
    # --- Development: start React dev server ---
    echo -e "${BLUE}Starting React frontend (dev)...${NC}"
    cd "$AGENT_APP_DIR"
    BROWSER=none HOST=0.0.0.0 npm start > "$LOG_DIR/react.log" 2>&1 &
    REACT_PID=$!
    echo "$REACT_PID" >> "$PID_FILE"
    echo -e "${GREEN}✓ React started (PID: $REACT_PID)${NC}"
    echo -e "${BLUE}  Log: $LOG_DIR/react.log${NC}\n"
fi

# --- Start Python ---
echo -e "${BLUE}Starting Python backend...${NC}"
cd "$TOOLS_DIR"
python3 app.py > "$LOG_DIR/python.log" 2>&1 &
PYTHON_PID=$!
echo "$PYTHON_PID" >> "$PID_FILE"
echo -e "${GREEN}✓ Python started (PID: $PYTHON_PID)${NC}"
echo -e "${BLUE}  Log: $LOG_DIR/python.log${NC}\n"

# --- Wait and verify ---
echo -e "${YELLOW}Waiting for services to initialize...${NC}"
sleep 6

PYTHON_OK=false
kill -0 "$PYTHON_PID" 2>/dev/null && PYTHON_OK=true

if [ "$ENVIRONMENT" = "production" ]; then
    # In production, check nginx and python
    NGINX_OK=false
    curl -s --max-time 3 http://127.0.0.1 > /dev/null 2>&1 && NGINX_OK=true

    if [ "$NGINX_OK" = true ] && [ "$PYTHON_OK" = true ]; then
        echo -e "${GREEN}========================================${NC}"
        echo -e "${GREEN}✓ All services running!${NC}"
        echo -e "${GREEN}========================================${NC}\n"
        echo -e "  Frontend : ${YELLOW}http://agents.enableyou.co${NC} (nginx port 80)"
        echo -e "  Backend  : ${YELLOW}http://agents.enableyou.co:5000${NC}"
        echo -e "  Logs     : ${YELLOW}$LOG_DIR/${NC}\n"
        echo -e "${YELLOW}To stop: ./stop.sh${NC}\n"
    else
        echo -e "${RED}✗ One or more services failed to start.${NC}\n"
        [ "$NGINX_OK" = false ] && echo -e "${RED}Nginx not responding on port 80${NC}"
        [ "$PYTHON_OK" = false ] && echo -e "${RED}Python log:${NC}" && tail -n 20 "$LOG_DIR/python.log" 2>/dev/null
        kill "$PYTHON_PID" 2>/dev/null || true
        exit 1
    fi
else
    REACT_OK=false
    kill -0 "$REACT_PID" 2>/dev/null && REACT_OK=true

    if [ "$REACT_OK" = true ] && [ "$PYTHON_OK" = true ]; then
        echo -e "${GREEN}========================================${NC}"
        echo -e "${GREEN}✓ All services running!${NC}"
        echo -e "${GREEN}========================================${NC}\n"
        echo -e "  Frontend : ${YELLOW}http://localhost:3000${NC}"
        echo -e "  Backend  : ${YELLOW}http://localhost:5000${NC}"
        echo -e "  Logs     : ${YELLOW}$LOG_DIR/${NC}\n"
        echo -e "${YELLOW}To stop: ./stop.sh${NC}\n"
    else
        echo -e "${RED}✗ One or more services failed to start.${NC}\n"
        [ "$REACT_OK" = false ] && echo -e "${RED}React log:${NC}" && tail -n 20 "$LOG_DIR/react.log" 2>/dev/null
        [ "$PYTHON_OK" = false ] && echo -e "${RED}Python log:${NC}" && tail -n 20 "$LOG_DIR/python.log" 2>/dev/null
        kill "$REACT_PID" 2>/dev/null || true
        kill "$PYTHON_PID" 2>/dev/null || true
        exit 1
    fi
fi

