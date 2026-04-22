#!/bin/bash

###############################################################################
# Clean Start Script
# 
# This script performs a complete cleanup and fresh start of the application:
# 1. Stops all running services
# 2. Cleans up old build files and environment caches
# 3. Optionally updates environment variables
# 4. Restarts all services from scratch
#
# Usage: ./clean_start.sh [IP_ADDRESS]
# 
# Examples:
#   ./clean_start.sh                    # Uses existing .env files
#   ./clean_start.sh 34.70.101.143      # Updates .env with new IP
#   ./clean_start.sh localhost          # Use localhost
###############################################################################

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Get the directory where the script is located
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
PROJECT_ROOT="$SCRIPT_DIR"

# API URL parameter (optional)
API_URL="${1:-}"

echo -e "${YELLOW}========================================${NC}"
echo -e "${YELLOW}Enable Agents - Clean Start${NC}"
echo -e "${YELLOW}========================================${NC}\n"

# Step 1: Stop everything
echo -e "${BLUE}Step 1: Stopping all services...${NC}"
if [ -f "$PROJECT_ROOT/stop.sh" ]; then
    bash "$PROJECT_ROOT/stop.sh" 2>/dev/null || true
    sleep 2
else
    echo -e "${YELLOW}Warning: stop.sh not found${NC}"
fi

# Step 2: Clean up old build files and caches
echo -e "\n${BLUE}Step 2: Cleaning up old build files and caches...${NC}"

echo "  Removing React build..."
rm -rf "$PROJECT_ROOT/agent-app/build" 2>/dev/null || true
rm -rf "$PROJECT_ROOT/agent-app/.env.production.local" 2>/dev/null || true
rm -rf "$PROJECT_ROOT/agent-app/node_modules/.cache" 2>/dev/null || true
echo -e "${GREEN}  ✓ React build cleaned${NC}"

echo "  Removing Python cache..."
find "$PROJECT_ROOT/tools" -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
find "$PROJECT_ROOT/tools" -type f -name "*.pyc" -delete 2>/dev/null || true
echo -e "${GREEN}  ✓ Python cache cleaned${NC}"

# Step 3: Update environment files if API_URL provided
if [ ! -z "$API_URL" ]; then
    echo -e "\n${BLUE}Step 3: Updating environment variables to: $API_URL${NC}"
    
    # Update agent-app/.env
    cat > "$PROJECT_ROOT/agent-app/.env" << EOF
REACT_APP_API_URL=http://$API_URL:5000
EOF
    echo -e "${GREEN}  ✓ Updated agent-app/.env${NC}"
    
    # Update tools/.env
    cat > "$PROJECT_ROOT/tools/.env" << EOF
ENVIRONMENT=production
PUBLIC_URL=http://$API_URL:5000
REACT_APP_API_URL=http://$API_URL:5000
GOOGLE_REDIRECT_URI=http://$API_URL:5000/auth/google/callback
OAUTHLIB_INSECURE_TRANSPORT=1
EOF
    echo -e "${GREEN}  ✓ Updated tools/.env${NC}"
else
    echo -e "\n${BLUE}Step 3: Skipping environment update (using existing .env files)${NC}"
    echo -e "${YELLOW}  Current agent-app/.env:${NC}"
    cat "$PROJECT_ROOT/agent-app/.env" 2>/dev/null || echo "    (not found)"
fi

# Step 4: Start services
echo -e "\n${BLUE}Step 4: Starting services...${NC}"
bash "$PROJECT_ROOT/start.sh"

echo -e "\n${GREEN}========================================${NC}"
echo -e "${GREEN}✓ Clean start complete!${NC}"
echo -e "${GREEN}========================================${NC}\n"
