#!/bin/bash
###############################################################################
# Stop Script — stops all services started by start.sh
#
# For PRODUCTION: Stops Flask backend and nginx
# For DEVELOPMENT: Stops Flask backend and npm dev server
#
# Usage: ./scripts/stop.sh
###############################################################################

set -euo pipefail

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
PROJECT_ROOT="$( cd "$SCRIPT_DIR/.." && pwd )"
# shellcheck source=scripts/lib/common.sh
source "$SCRIPT_DIR/lib/common.sh"

PID_FILE="$PROJECT_ROOT/.pids"
LOG_DIR="$PROJECT_ROOT/.logs"
BACKEND_DIR="$PROJECT_ROOT/backend"

print_banner "Enable Agents - Stopping Services"

# Detect environment from backend .env (default: development)
ENVIRONMENT="development"
if [ -f "$BACKEND_DIR/.env" ]; then
  ENVIRONMENT=$(grep "^ENVIRONMENT=" "$BACKEND_DIR/.env" | cut -d'=' -f2 || echo "development")
fi

# Stop tracked PIDs from the PID file
STOPPED=0
if [ -f "$PID_FILE" ]; then
  echo -e "${BLUE}Stopping tracked processes...${NC}"
  while IFS= read -r PID; do
    [ -z "$PID" ] && continue
    if kill -0 "$PID" 2>/dev/null; then
      echo -e "${BLUE}  Stopping PID $PID...${NC}"
      kill -TERM "$PID" 2>/dev/null || true
      for i in {1..50}; do
        if ! kill -0 "$PID" 2>/dev/null; then
          echo -e "${GREEN}  ✓ PID $PID stopped${NC}"
          ((STOPPED++))
          break
        fi
        sleep 0.1
      done
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

echo -e "${BLUE}Cleaning up any remaining services...${NC}"
stop_backend
stop_frontend_dev
[ "$ENVIRONMENT" = "production" ] && stop_nginx

print_success "All services stopped!"

if [ -d "$LOG_DIR" ]; then
  echo -e "${BLUE}Log files:${NC}"
  echo -e "  $LOG_DIR/python.log"
  [ "$ENVIRONMENT" != "production" ] && echo -e "  $LOG_DIR/react.log"
  echo
fi

echo -e "${BLUE}To restart:${NC}"
echo -e "  ${YELLOW}./scripts/start.sh${NC}\n"
