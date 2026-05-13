#!/bin/bash

###############################################################################
# Deployment Script
#
# Setup and start services on remote (manual code pull):
# - Sets up Python environment (create venv, install dependencies)
# - Starts React frontend (if not running)
# - Starts Python backend (if not running)
# - Runs both in background (returns terminal access)
#
# NOTE: You must pull code manually before running this script.
# For Docker + Compose on the VM, use scripts/run.sh (remote / remote-ssl / remote-rebuild).
#
# Usage: ./deploy.sh <remote_user> <remote_ip>
# Example: ./deploy.sh rhishi 34.70.101.143
###############################################################################

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
# shellcheck source=scripts/lib/common.sh
source "$SCRIPT_DIR/lib/common.sh"

# Check arguments
if [ $# -lt 2 ]; then
    echo -e "${RED}Usage: ./deploy.sh <remote_user> <remote_ip>${NC}"
    echo -e "Example: ./deploy.sh rhishi 34.70.101.143"
    exit 1
fi

REMOTE_USER=$1
REMOTE_IP=$2
REMOTE_HOST="$REMOTE_USER@$REMOTE_IP"
REMOTE_DIR="~/enable_agents"

print_banner "Enable Agents - Remote Deployment"

echo -e "${BLUE}Target: $REMOTE_HOST${NC}"
echo -e "${BLUE}Directory: $REMOTE_DIR\n${NC}"

# Step 1: Setup database (create venv, install deps)
echo -e "${BLUE}[1/4] Setting up Python environment...${NC}"
ssh "$REMOTE_HOST" "cd $REMOTE_DIR && bash -c 'set -e; \
if [ ! -d venv ]; then python3 -m venv venv; fi; \
source venv/bin/activate; \
pip install -q --upgrade pip; \
pip install -q --no-cache-dir -r backend/requirements.txt; \
if [ -f backend/migrations.py ]; then python backend/migrations.py; fi'" > /dev/null 2>&1

if [ $? -eq 0 ]; then
    echo -e "${GREEN}✓ Python environment ready${NC}\n"
else
    echo -e "${RED}✗ Failed to setup Python${NC}\n"
    exit 1
fi

# Step 2: Configure .env files
echo -e "${BLUE}[2/4] Checking configuration files...${NC}"

# Check and create tools/.env if missing
ssh "$REMOTE_HOST" "cd $REMOTE_DIR && bash -c '\
if [ ! -f backend/.env ]; then
  echo \"ENVIRONMENT=production\" > backend/.env
  echo \"PUBLIC_URL=http://agents.enableyou.co:5000\" >> backend/.env
  echo \"REACT_APP_API_URL=http://agents.enableyou.co:5000\" >> backend/.env
  echo \"OAUTHLIB_INSECURE_TRANSPORT=1\" >> backend/.env
fi
if ! grep -q \"^REACT_APP_API_URL=\" backend/.env; then
  echo \"REACT_APP_API_URL=http://agents.enableyou.co:5000\" >> backend/.env
fi'" > /dev/null 2>&1

# Check and create frontend/.env if missing
ssh "$REMOTE_HOST" "cd $REMOTE_DIR && bash -c '\
if [ ! -f frontend/.env ]; then
  echo \"REACT_APP_API_URL=http://agents.enableyou.co:5000\" > frontend/.env
fi'" > /dev/null 2>&1

echo -e "${GREEN}✓ Configuration files ready${NC}\n"

# Step 3: Start backend if not running
echo -e "${BLUE}[3/4] Starting backend...${NC}"

BACKEND_RUNNING=$(ssh "$REMOTE_HOST" "pgrep -f 'python.*app\.py' 2>/dev/null | wc -l")

if [ "$BACKEND_RUNNING" -gt 0 ]; then
    echo -e "${GREEN}✓ Backend already running${NC}"
else
    echo -e "${BLUE}   Starting Python backend...${NC}"
    ssh "$REMOTE_HOST" "cd $REMOTE_DIR && bash -c '\
mkdir -p .logs
source venv/bin/activate
cd backend
nohup python app.py > ../.logs/python.log 2>&1 &
sleep 2'" > /dev/null 2>&1
    
    BACKEND_RUNNING=$(ssh "$REMOTE_HOST" "pgrep -f 'python.*app\.py' 2>/dev/null | wc -l")
    if [ "$BACKEND_RUNNING" -gt 0 ]; then
        echo -e "${GREEN}✓ Backend started${NC}"
    else
        echo -e "${RED}✗ Failed to start backend${NC}"
        echo -e "${RED}   Check logs: ssh $REMOTE_HOST tail .logs/python.log${NC}"
    fi
fi

echo

# Step 4: Start React frontend if not running
echo -e "${BLUE}[4/4] Starting frontend...${NC}"

NPM_RUNNING=$(ssh "$REMOTE_HOST" "pgrep -f 'npm.*start' 2>/dev/null | wc -l")

if [ "$NPM_RUNNING" -gt 0 ]; then
    echo -e "${GREEN}✓ Frontend already running${NC}"
else
    echo -e "${BLUE}   Starting React dev server...${NC}"
    ssh "$REMOTE_HOST" "cd $REMOTE_DIR/frontend && bash -c '\
if [ ! -d node_modules ]; then npm install -q; fi
nohup npm start > ../.logs/react.log 2>&1 &
sleep 3'" > /dev/null 2>&1
    
    NPM_RUNNING=$(ssh "$REMOTE_HOST" "pgrep -f 'npm.*start' 2>/dev/null | wc -l")
    if [ "$NPM_RUNNING" -gt 0 ]; then
        echo -e "${GREEN}✓ Frontend started${NC}"
    else
        echo -e "${RED}✗ Failed to start frontend${NC}"
        echo -e "${RED}   Check logs: ssh $REMOTE_HOST tail .logs/react.log${NC}"
    fi
fi

echo

# Final status
print_success "Deployment complete!"

# Get status
BACKEND_STATUS=$(ssh "$REMOTE_HOST" "pgrep -f 'python.*app\.py' >/dev/null 2>&1 && echo 'running' || echo 'stopped'")
FRONTEND_STATUS=$(ssh "$REMOTE_HOST" "pgrep -f 'npm.*start' >/dev/null 2>&1 && echo 'running' || echo 'stopped'")

echo -e "${BLUE}Services Status:${NC}"
echo -e "  Backend:  ${GREEN}$BACKEND_STATUS${NC} (port 5000)"
echo -e "  Frontend: ${GREEN}$FRONTEND_STATUS${NC} (port 3000)\n"

echo -e "${BLUE}Access:${NC}"
echo -e "  Frontend: http://34.70.101.143:3000"
echo -e "  Backend:  http://34.70.101.143:5000\n"

echo -e "${BLUE}Useful commands:${NC}"
echo -e "  SSH:         ${YELLOW}ssh $REMOTE_HOST${NC}"
echo -e "  View logs:   ${YELLOW}ssh $REMOTE_HOST tail -f .logs/python.log${NC}"
echo -e "  Stop all:    ${YELLOW}ssh $REMOTE_HOST 'pkill -f \"app\\.py|npm.*start\"'${NC}"
echo -e "  Restart:     ${YELLOW}./deploy.sh $REMOTE_USER $REMOTE_IP${NC}\n"
