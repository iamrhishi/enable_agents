# Deployment Configuration Guide

## Overview

This project uses a centralized environment configuration system that supports both local development and remote production deployment. All environment-specific URLs and credentials are configured via a single `.env` file.

## Quick Start

### 1. Development Setup (Localhost)

```bash
# Copy the example configuration
cp .env.example .env

# Edit .env for local development
# Set these values:
ENVIRONMENT=development
REACT_APP_API_URL=http://localhost:5000
PUBLIC_URL=http://localhost:5000
GOOGLE_REDIRECT_URI=http://localhost:5000/auth/google/callback
LINKEDIN_REDIRECT_URI=http://localhost:5000/auth/linkedin/callback
OAUTHLIB_INSECURE_TRANSPORT=1
```

### 2. Production Setup (Remote GCP)

```bash
# On remote instance: Copy example and customize
cp .env.example .env

# Edit .env for production
# Set these values:
ENVIRONMENT=production
REACT_APP_API_URL=http://agents.enableyou.co:5000
PUBLIC_URL=http://agents.enableyou.co:5000
GOOGLE_REDIRECT_URI=http://agents.enableyou.co:5000/auth/google/callback
LINKEDIN_REDIRECT_URI=http://agents.enableyou.co:5000/auth/linkedin/callback
OAUTHLIB_INSECURE_TRANSPORT=1  # Keep as 1 if using HTTP (not HTTPS)
```

## Configuration Files

### Backend Configuration

**File**: `tools/.env`

```env
# Environment
ENVIRONMENT=development|production

# URLs (must match where instance is accessible)
PUBLIC_URL=http://localhost:5000  # or http://agents.enableyou.co:5000

# Google OAuth
GOOGLE_CLIENT_ID=...
GOOGLE_CLIENT_SECRET=...
GOOGLE_REDIRECT_URI=http://localhost:5000/auth/google/callback

# Other credentials (optional)
LINKEDIN_CLIENT_ID=...
LINKEDIN_CLIENT_SECRET=...
OPENAI_API_KEY=...
```

**Python Module**: `tools/config.py`

This module validates all environment variables on startup and provides a single `Config` class:

```python
from config import Config

print(Config.PUBLIC_URL)  # http://localhost:5000
print(Config.ENVIRONMENT)  # development
print(Config.GOOGLE_REDIRECT_URI)  # auto-constructed
```

### Frontend Configuration

**File**: `agent-app/.env`

```env
REACT_APP_API_URL=http://localhost:5000
```

**Important**: React bakes environment variables into the build at compile time. After changing `.env`, you MUST rebuild:

```bash
cd agent-app
npm run build
```

**Module**: `agent-app/src/config/environment.config.js`

```javascript
import { API_URL, EnvironmentConfig } from './config/environment.config';

// Use in components
const authUrl = EnvironmentConfig.getAuthUrl('google');
```

## OAuth Configuration (Google)

### Register OAuth Credentials

1. Go to [Google Cloud Console](https://console.cloud.google.com/)
2. Create or select your project
3. Enable "Google+ API"
4. Go to "Credentials" > "Create Credentials" > "OAuth Client ID"
5. Application type: "Web Application"
6. Authorized JavaScript origins:
   - `http://localhost:5000` (for dev)
   - `http://agents.enableyou.co` (for production)
7. Authorized redirect URIs:
   - `http://localhost:5000/auth/google/callback`
   - `http://agents.enableyou.co:5000/auth/google/callback`

### Update .env

```env
GOOGLE_CLIENT_ID=<your-client-id>
GOOGLE_CLIENT_SECRET=<your-client-secret>
GOOGLE_REDIRECT_URI=http://localhost:5000/auth/google/callback  # for dev
# OR
GOOGLE_REDIRECT_URI=http://agents.enableyou.co:5000/auth/google/callback  # for prod
```

## Deployment Scripts

### Local Development

```bash
./start.sh
```

This script will:
1. Validate `.env` file exists
2. Kill existing processes on ports 5000/3000
3. Activate Python venv
4. Run Flask backend on port 5000
5. Run React frontend on port 3000

### Remote Deployment (GCP)

1. SSH into instance:
```bash
ssh rhishi@34.70.101.143
```

2. Navigate to project:
```bash
cd ~/enable_agents
```

3. Configure environment:
```bash
cp .env.example .env
# Edit .env with remote URLs
nano .env
```

4. Start services:
```bash
./start.sh
```

Services should now be running:
- Backend: http://agents.enableyou.co:5000
- Frontend: http://agents.enableyou.co (via Nginx)

### Docker Deployment

```bash
docker-compose build --build-arg ENVIRONMENT=production
docker-compose up
```

## Troubleshooting

### "REACT_APP_API_URL not set"

Solution: Ensure `.env` exists in `agent-app/` and rebuild:
```bash
cd agent-app
npm run build
```

### "PUBLIC_URL environment variable not set"

Solution: Ensure `.env` exists in `tools/` with `PUBLIC_URL` set:
```bash
cat tools/.env | grep PUBLIC_URL
```

### OAuth Callback Mismatch

Error: `redirect_uri_mismatch`

Solution: 
1. Check your `.env` file matches registered OAuth URLs exactly
2. Verify `GOOGLE_REDIRECT_URI` matches what's in Google Console
3. Clear browser cookies and try again

### Backend not accessible

Check:
```bash
# Is Flask running on port 5000?
lsof -i :5000

# Are logs showing errors?
tail -f logs/python.log

# Is nginx proxying correctly?
curl http://localhost/auth/google/start
```

## Configuration Checklist

Before deployment, ensure:

- [ ] `.env` file created in both `tools/` and `agent-app/`
- [ ] `ENVIRONMENT` set to `development` or `production`
- [ ] `PUBLIC_URL` matches domain/IP where instance is accessible
- [ ] `REACT_APP_API_URL` matches `PUBLIC_URL`
- [ ] `GOOGLE_REDIRECT_URI` registered in Google OAuth Console
- [ ] `GOOGLE_CLIENT_ID` and `GOOGLE_CLIENT_SECRET` set
- [ ] `OPENAI_API_KEY` set
- [ ] React build created after `.env` changes: `npm run build`
- [ ] Start script runs without errors: `./start.sh`
- [ ] Backend accessible: `curl http://<PUBLIC_URL>:5000/auth/google/start`
- [ ] Frontend loads: Browser to `http://<PUBLIC_URL>`

## Environment Variables Reference

| Variable | Used By | Example (Dev) | Example (Prod) |
|----------|---------|---------------|----------------|
| `ENVIRONMENT` | Backend | `development` | `production` |
| `PUBLIC_URL` | Backend | `http://localhost:5000` | `http://agents.enableyou.co:5000` |
| `REACT_APP_API_URL` | Frontend | `http://localhost:5000` | `http://agents.enableyou.co:5000` |
| `GOOGLE_CLIENT_ID` | Backend | [from console] | [from console] |
| `GOOGLE_CLIENT_SECRET` | Backend | [from console] | [from console] |
| `GOOGLE_REDIRECT_URI` | Backend | `http://localhost:5000/auth/google/callback` | `http://agents.enableyou.co:5000/auth/google/callback` |
| `OAUTHLIB_INSECURE_TRANSPORT` | Backend | `1` | `1` (for HTTP) or `0` (for HTTPS) |
| `OPENAI_API_KEY` | Backend | [from OpenAI] | [from OpenAI] |

## Support

If issues persist:
1. Check logs: `tail -f logs/python.log` and `logs/nginx.log`
2. Verify URLs match between `.env` files and OAuth configuration
3. Ensure `.env` files are NOT tracked in git (check `.gitignore`)
