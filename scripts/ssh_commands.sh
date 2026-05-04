#!/bin/bash
###############################################################################
# SSH Commands — deploy static frontend build to a remote nginx server.
#
# Usage: ./scripts/ssh_commands.sh <user> <host> [remote_dir]
# Example: ./scripts/ssh_commands.sh rhishi 34.70.101.143
#          ./scripts/ssh_commands.sh rhishi 34.70.101.143 ~/projects/enable_agents
###############################################################################

REMOTE_USER="${1:?Usage: $0 <user> <host> [remote_dir]}"
REMOTE_HOST="${2:?Usage: $0 <user> <host> [remote_dir]}"
REMOTE_DIR="${3:-~/enable_agents}"

ssh -o ConnectTimeout=15 -o StrictHostKeyChecking=no "$REMOTE_USER@$REMOTE_HOST" <<REMOTE
set -e
REMOTE_DIR="$REMOTE_DIR"

echo "Building frontend if needed..."
ls -la "\$REMOTE_DIR/frontend/build" 2>/dev/null || (cd "\$REMOTE_DIR/frontend" && npm run build 2>&1 | tail -20)

echo "Deploying to nginx..."
sudo mkdir -p /usr/share/nginx/html
sudo cp -r "\$REMOTE_DIR/frontend/build/"* /usr/share/nginx/html/ 2>/dev/null || echo "Copy failed"
sudo ls -la /usr/share/nginx/html/ | head -15

echo "Restarting nginx..."
sudo systemctl restart nginx

echo "Verifying..."
curl -s http://localhost/ | head -20
curl -s http://localhost/api/ | head -20
REMOTE
