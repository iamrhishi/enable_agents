# Remote Instance Setup Guide

## Quick Start

After pulling the code on your remote instance, follow these steps:

### 1. Configure Backend Environment

Create `.env` file in the `tools` directory:

```bash
cd ~/enable_agents/tools
cat > .env << 'EOF'
# Google OAuth Configuration
GOOGLE_CLIENT_ID=your_client_id_here
GOOGLE_CLIENT_SECRET=your_client_secret_here
GOOGLE_REDIRECT_URI=http://your-remote-ip:5000/auth/google/callback

# Other configurations as needed
EOF
```

**Get your credentials from:** https://console.cloud.google.com/

### 2. Configure Frontend Environment

Create `.env` file in the `agent-app` directory:

```bash
cd ~/enable_agents/agent-app
cat > .env << 'EOF'
# Replace with your actual remote server IP or domain
REACT_APP_API_URL=http://your-remote-ip:5000
EOF
```

### 3. Setup Database

```bash
cd ~/enable_agents
./setup_db.sh
```

### 4. Start Services

```bash
./start.sh
```

### 5. Access the Application

- **Frontend:** `http://your-remote-ip:3000`
- **Backend:** `http://your-remote-ip:5000`

---

## Environment Variable Guide

### Frontend (.env in agent-app/)

```bash
# Backend API URL (critical for remote deployment)
REACT_APP_API_URL=http://your-server-ip:5000

# Examples:
# REACT_APP_API_URL=http://192.168.1.100:5000
# REACT_APP_API_URL=http://example.com:5000
# REACT_APP_API_URL=https://api.example.com
```

### Backend (.env in tools/)

```bash
# Google OAuth
GOOGLE_CLIENT_ID=your_client_id
GOOGLE_CLIENT_SECRET=your_client_secret
GOOGLE_REDIRECT_URI=http://your-remote-ip:5000/auth/google/callback

# Database (if using remote database)
DATABASE_URL=your_database_url

# Other API keys and credentials
OPENAI_API_KEY=your_key
TWILIO_ACCOUNT_SID=your_sid
TWILIO_AUTH_TOKEN=your_token
```

---

## Troubleshooting

### "Failed to initiate Google Login" on Remote

**Problem:** Frontend can't connect to backend
**Solution:** 
1. Check `.env` file in `agent-app/` has correct `REACT_APP_API_URL`
2. Ensure backend is running: `./start.sh`
3. Check firewall allows port 5000

### Verify Connectivity

```bash
# From remote instance, test backend is reachable
curl http://localhost:5000/health

# From your local machine, test remote backend
curl http://your-remote-ip:5000/health
```

### Check Service Status

```bash
# Check if services are running
ps aux | grep -E "npm|python"

# View logs
tail -f .logs/react.log
tail -f .logs/python.log
```

---

## Docker Deployment (Alternative)

For easier deployment, you can use Docker:

```bash
# Build and run with Docker (if Dockerfile exists)
docker-compose up
```

---

## Important Notes

1. **API URLs must match:** Frontend must point to the actual backend server IP/domain
2. **Google OAuth redirect URI:** Must match your app's actual deployment URL
3. **Firewall:** Ensure ports 3000 (frontend) and 5000 (backend) are accessible
4. **Environment variables:** Are loaded from `.env` files, not committed to git

