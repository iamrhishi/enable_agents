#!/bin/bash

###############################################################################
# HTTPS Setup Script
# 
# Automates SSL certificate installation and nginx configuration
#
# Usage: ./scripts/setup_https.sh <domain> <email>
# Example: ./scripts/setup_https.sh agents.enableyou.co rhishi@enableyou.co
###############################################################################

set -e

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
PROJECT_ROOT="$( cd "$SCRIPT_DIR/.." && pwd )"
# shellcheck source=scripts/lib/common.sh
source "$SCRIPT_DIR/lib/common.sh"

DOMAIN="${1:-}"
EMAIL="${2:-}"

# Validation
if [ -z "$DOMAIN" ] || [ -z "$EMAIL" ]; then
    echo -e "${RED}Usage: ./scripts/setup_https.sh <domain> <email>${NC}"
    echo -e "${RED}Example: ./scripts/setup_https.sh agents.enableyou.co rhishi@enableyou.co${NC}"
    exit 1
fi

print_banner "HTTPS Setup for $DOMAIN"

# Step 1: Stop services
echo -e "${BLUE}Step 1: Stopping services...${NC}"
bash "$SCRIPT_DIR/stop.sh" 2>/dev/null || true
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

# Update frontend/.env
cat > "$PROJECT_ROOT/frontend/.env" << EOF
REACT_APP_API_URL=https://$DOMAIN
EOF
echo -e "${GREEN}  ✓ Updated frontend/.env${NC}"

# Update backend/.env - preserve existing variables
if [ -f "$PROJECT_ROOT/backend/.env" ]; then
    grep -v "^REACT_APP_API_URL\|^PUBLIC_URL\|^GOOGLE_REDIRECT_URI\|^OAUTHLIB_INSECURE_TRANSPORT" \
        "$PROJECT_ROOT/backend/.env" > "$PROJECT_ROOT/backend/.env.tmp" || true
    
    cat > "$PROJECT_ROOT/backend/.env" << EOF
ENVIRONMENT=production
PUBLIC_URL=https://$DOMAIN
REACT_APP_API_URL=https://$DOMAIN
GOOGLE_REDIRECT_URI=https://$DOMAIN/auth/google/callback
OAUTHLIB_INSECURE_TRANSPORT=0
EOF
    
    cat "$PROJECT_ROOT/backend/.env.tmp" >> "$PROJECT_ROOT/backend/.env" 2>/dev/null || true
    rm -f "$PROJECT_ROOT/backend/.env.tmp"
fi
echo -e "${GREEN}  ✓ Updated backend/.env${NC}"

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
bash "$SCRIPT_DIR/start.sh"

print_success "HTTPS Setup Complete!"

echo -e "${BLUE}Your application is now available at:${NC}"
echo -e "  ${YELLOW}https://$DOMAIN${NC}\n"

echo -e "${BLUE}Certificate details:${NC}"
sudo certbot certificates -d "$DOMAIN"

echo -e "\n${BLUE}Test your setup:${NC}"
echo -e "  ${YELLOW}curl https://$DOMAIN${NC}"
