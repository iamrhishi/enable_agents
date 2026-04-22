#!/bin/bash

###############################################################################
# HTTPS Setup Script
# 
# Automates SSL certificate installation and nginx configuration
#
# Usage: ./setup_https.sh <domain> <email>
# Example: ./setup_https.sh agents.enableyou.co rhishi@enableyou.co
###############################################################################

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Parameters
DOMAIN="${1:-}"
EMAIL="${2:-}"
PROJECT_ROOT="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

# Validation
if [ -z "$DOMAIN" ] || [ -z "$EMAIL" ]; then
    echo -e "${RED}Usage: ./setup_https.sh <domain> <email>${NC}"
    echo -e "${RED}Example: ./setup_https.sh agents.enableyou.co rhishi@enableyou.co${NC}"
    exit 1
fi

echo -e "${YELLOW}========================================${NC}"
echo -e "${YELLOW}HTTPS Setup for $DOMAIN${NC}"
echo -e "${YELLOW}========================================${NC}\n"

# Step 1: Stop services
echo -e "${BLUE}Step 1: Stopping services...${NC}"
cd "$PROJECT_ROOT"
./stop.sh 2>/dev/null || true
sleep 3

# Step 2: Install certbot and nginx
echo -e "\n${BLUE}Step 2: Installing certbot and nginx...${NC}"
sudo apt update -qq
sudo apt install -y -qq certbot python3-certbot-nginx nginx > /dev/null 2>&1
echo -e "${GREEN}✓ Installed${NC}"

# Step 3: Get SSL certificate
echo -e "\n${BLUE}Step 3: Obtaining SSL certificate from Let's Encrypt...${NC}"
sudo certbot certonly --standalone -d "$DOMAIN" \
  --email "$EMAIL" \
  --agree-tos \
  --non-interactive \
  --quiet

if [ ! -f "/etc/letsencrypt/live/$DOMAIN/fullchain.pem" ]; then
    echo -e "${RED}✗ Certificate not found at /etc/letsencrypt/live/$DOMAIN/fullchain.pem${NC}"
    exit 1
fi
echo -e "${GREEN}✓ Certificate obtained${NC}"

# Step 4: Configure nginx
echo -e "\n${BLUE}Step 4: Configuring nginx reverse proxy...${NC}"

sudo tee /etc/nginx/sites-available/$DOMAIN > /dev/null << EOF
upstream backend {
    server localhost:5000;
}

upstream frontend {
    server localhost:3000;
}

# Redirect HTTP to HTTPS
server {
    listen 80;
    server_name $DOMAIN;
    return 301 https://\$server_name\$request_uri;
}

# HTTPS server
server {
    listen 443 ssl http2;
    server_name $DOMAIN;

    ssl_certificate /etc/letsencrypt/live/$DOMAIN/fullchain.pem;
    ssl_certificate_key /etc/letsencrypt/live/$DOMAIN/privkey.pem;

    ssl_protocols TLSv1.2 TLSv1.3;
    ssl_ciphers HIGH:!aNULL:!MD5;
    ssl_prefer_server_ciphers on;

    # Frontend (React on port 3000)
    location / {
        proxy_pass http://frontend;
        proxy_http_version 1.1;
        proxy_set_header Upgrade \$http_upgrade;
        proxy_set_header Connection 'upgrade';
        proxy_set_header Host \$host;
        proxy_cache_bypass \$http_upgrade;
    }

    # Backend API routes
    location /auth/ {
        proxy_pass http://backend;
        proxy_http_version 1.1;
        proxy_set_header Host \$host;
        proxy_set_header X-Real-IP \$remote_addr;
        proxy_set_header X-Forwarded-For \$proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto \$scheme;
    }

    location /login {
        proxy_pass http://backend;
        proxy_http_version 1.1;
        proxy_set_header Host \$host;
        proxy_set_header X-Real-IP \$remote_addr;
        proxy_set_header X-Forwarded-For \$proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto \$scheme;
    }

    location /register {
        proxy_pass http://backend;
        proxy_http_version 1.1;
        proxy_set_header Host \$host;
        proxy_set_header X-Real-IP \$remote_addr;
        proxy_set_header X-Forwarded-For \$proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto \$scheme;
    }

    location ~ ^/([a-zA-Z0-9_]+)$ {
        proxy_pass http://backend;
        proxy_http_version 1.1;
        proxy_set_header Host \$host;
        proxy_set_header X-Real-IP \$remote_addr;
        proxy_set_header X-Forwarded-For \$proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto \$scheme;
    }
}
EOF

# Enable the site
sudo ln -sf /etc/nginx/sites-available/$DOMAIN /etc/nginx/sites-enabled/
sudo rm -f /etc/nginx/sites-enabled/default

# Test nginx
sudo nginx -t > /dev/null 2>&1
if [ $? -ne 0 ]; then
    echo -e "${RED}✗ Nginx configuration test failed${NC}"
    exit 1
fi
echo -e "${GREEN}✓ Nginx configured${NC}"

# Step 5: Update .env files
echo -e "\n${BLUE}Step 5: Updating environment files...${NC}"

# Update agent-app/.env
cat > "$PROJECT_ROOT/agent-app/.env" << EOF
REACT_APP_API_URL=https://$DOMAIN
EOF
echo -e "${GREEN}  ✓ Updated agent-app/.env${NC}"

# Update tools/.env - preserve existing variables
if [ -f "$PROJECT_ROOT/tools/.env" ]; then
    # Extract non-URL variables
    grep -v "^REACT_APP_API_URL\|^PUBLIC_URL\|^GOOGLE_REDIRECT_URI\|^OAUTHLIB_INSECURE_TRANSPORT" \
        "$PROJECT_ROOT/tools/.env" > "$PROJECT_ROOT/tools/.env.tmp" || true
    
    # Prepend new URL variables
    cat > "$PROJECT_ROOT/tools/.env" << EOF
ENVIRONMENT=production
PUBLIC_URL=https://$DOMAIN
REACT_APP_API_URL=https://$DOMAIN
GOOGLE_REDIRECT_URI=https://$DOMAIN/auth/google/callback
OAUTHLIB_INSECURE_TRANSPORT=0
EOF
    
    # Append existing variables
    cat "$PROJECT_ROOT/tools/.env.tmp" >> "$PROJECT_ROOT/tools/.env" 2>/dev/null || true
    rm -f "$PROJECT_ROOT/tools/.env.tmp"
fi
echo -e "${GREEN}  ✓ Updated tools/.env${NC}"

# Step 6: Start nginx
echo -e "\n${BLUE}Step 6: Starting nginx...${NC}"
sudo systemctl start nginx
sudo systemctl enable nginx
echo -e "${GREEN}✓ Nginx started${NC}"

# Step 7: Set up auto-renewal
echo -e "\n${BLUE}Step 7: Setting up certificate auto-renewal...${NC}"
sudo systemctl enable certbot.timer
sudo systemctl start certbot.timer
echo -e "${GREEN}✓ Auto-renewal configured${NC}"

# Step 8: Start services
echo -e "\n${BLUE}Step 8: Starting application services...${NC}"
cd "$PROJECT_ROOT"
./start.sh

echo -e "\n${GREEN}========================================${NC}"
echo -e "${GREEN}✓ HTTPS Setup Complete!${NC}"
echo -e "${GREEN}========================================${NC}\n"

echo -e "${BLUE}Your application is now available at:${NC}"
echo -e "  ${YELLOW}https://$DOMAIN${NC}\n"

echo -e "${BLUE}Certificate details:${NC}"
sudo certbot certificates -d "$DOMAIN"

echo -e "\n${BLUE}Test your setup:${NC}"
echo -e "  ${YELLOW}curl https://$DOMAIN${NC}"
