#!/bin/bash
# Shared shell utilities — source this file from any script in scripts/.
#
# Prerequisites: the sourcing script must set SCRIPT_DIR and PROJECT_ROOT
# before sourcing this file:
#
#   SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
#   PROJECT_ROOT="$( cd "$SCRIPT_DIR/.." && pwd )"
#   source "$SCRIPT_DIR/lib/common.sh"

# ── Colours ────────────────────────────────────────────────────────────────
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

# ── Banner helpers ──────────────────────────────────────────────────────────
print_banner() {
  local title="${1:-Enable Agents}"
  echo -e "${YELLOW}========================================${NC}"
  echo -e "${YELLOW}${title}${NC}"
  echo -e "${YELLOW}========================================${NC}\n"
}

print_success() {
  echo -e "\n${GREEN}========================================${NC}"
  echo -e "${GREEN}✓ ${1}${NC}"
  echo -e "${GREEN}========================================${NC}\n"
}

# ── Local-mode env validation ───────────────────────────────────────────────
# validate_local_env <backend_dir> <frontend_dir>
# Returns non-zero and prints an error if any required file/variable is missing.
validate_local_env() {
  local backend_dir="$1"
  local frontend_dir="$2"

  if [ ! -f "$backend_dir/.env" ]; then
    echo -e "${RED}✗ Missing: $backend_dir/.env${NC}"
    echo -e "${RED}  Create root .env from .env.example, then run ./scripts/run.sh local${NC}\n"
    return 1
  fi

  if [ ! -f "$frontend_dir/.env" ]; then
    echo -e "${RED}✗ Missing: $frontend_dir/.env${NC}"
    echo -e "${RED}  Run ./scripts/run.sh local to sync frontend/backend env files${NC}\n"
    return 1
  fi

  if ! grep -q "^PUBLIC_URL=" "$backend_dir/.env"; then
    echo -e "${RED}✗ PUBLIC_URL not set in $backend_dir/.env${NC}"
    return 1
  fi

  if ! grep -q "^REACT_APP_API_URL=" "$frontend_dir/.env"; then
    echo -e "${RED}✗ REACT_APP_API_URL not set in $frontend_dir/.env${NC}"
    return 1
  fi
}

# ── Process helpers ─────────────────────────────────────────────────────────
stop_backend() {
  local pids
  pids=$(pgrep -f "app\.py" 2>/dev/null || true)
  if [ -n "$pids" ]; then
    echo -e "${BLUE}  Stopping backend processes...${NC}"
    pkill -9 -f "app\.py" 2>/dev/null || true
    sleep 1
    echo -e "${GREEN}  ✓ Backend stopped${NC}"
  fi
}

stop_frontend_dev() {
  local pids
  pids=$(pgrep -f "npm.*start" 2>/dev/null || true)
  if [ -n "$pids" ]; then
    echo -e "${BLUE}  Stopping React dev server...${NC}"
    pkill -9 -f "npm.*start" 2>/dev/null || true
    sleep 1
    echo -e "${GREEN}  ✓ React dev server stopped${NC}"
  fi
}

stop_nginx() {
  local pids
  pids=$(pgrep -f "nginx" 2>/dev/null || true)
  if [ -n "$pids" ]; then
    echo -e "${BLUE}  Stopping nginx...${NC}"
    sudo nginx -s quit 2>/dev/null || sudo systemctl stop nginx 2>/dev/null || true
    sleep 1
    echo -e "${GREEN}  ✓ Nginx stopped${NC}"
  fi
}
