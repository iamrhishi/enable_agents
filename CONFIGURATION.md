# Environment Configuration Best Practices

## How Configuration Works

This project uses a **centralized environment variable system** that supports both local development and remote production deployment.

### The Configuration Flow

```
┌─────────────────────────────────────────────────────────────────┐
│                      .env files (root)                           │
│  ┌──────────────────────────────────────────────────────────┐   │
│  │ ENVIRONMENT=development                                  │   │
│  │ PUBLIC_URL=http://localhost:5000                        │   │
│  │ REACT_APP_API_URL=http://localhost:5000                │   │
│  │ GOOGLE_CLIENT_ID=...                                    │   │
│  │ GOOGLE_CLIENT_SECRET=...                                │   │
│  │ OPENAI_API_KEY=...                                      │   │
│  └──────────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────┘
          │                          │
          ▼                          ▼
    ┌──────────────────┐    ┌──────────────────┐
    │   tools/.env     │    │ agent-app/.env   │
    │   (Backend)      │    │   (Frontend)     │
    │ - Symlink or     │    │ - React build    │
    │   copy of root   │    │   uses this      │
    └──────────────────┘    └──────────────────┘
          │                          │
          ▼                          ▼
    ┌──────────────────┐    ┌──────────────────┐
    │  tools/config.py │    │ apiConfig.js     │
    │  - Validates     │    │ - Imports from   │
    │  - Provides      │    │   process.env    │
    │    Config class  │    │ - Throws if not  │
    │  - Prints status │    │   set            │
    └──────────────────┘    └──────────────────┘
          │                          │
          ▼                          ▼
    ┌──────────────────┐    ┌──────────────────┐
    │  Flask app.py    │    │  React App.js    │
    │  - Uses Config   │    │ - Uses API_CONFIG│
    │  - All OAuth     │    │ - Makes calls to │
    │    URLs from     │    │   API endpoints  │
    │    config        │    │                  │
    └──────────────────┘    └──────────────────┘
```

## Setup for Different Environments

### Development (Localhost)

1. **Copy the development configuration:**
   ```bash
   cp .env.development .env
   ```

2. **Edit `.env` for your local setup:**
   ```bash
   nano .env
   # or
   vim .env
   ```

3. **Required changes:**
   - Add your Google OAuth credentials
   - Add your OpenAI API key
   - Keep URLs as `http://localhost:5000`

4. **Start services:**
   ```bash
   ./start.sh
   ```

### Production (Remote GCP)

1. **SSH into remote instance:**
   ```bash
   ssh rhishi@34.70.101.143
   cd ~/enable_agents
   ```

2. **Copy production configuration:**
   ```bash
   cp .env.production .env
   ```

3. **Edit `.env` for remote deployment:**
   ```bash
   nano .env
   # Key changes:
   # - ENVIRONMENT=production
   # - PUBLIC_URL=http://agents.enableyou.co:5000
   # - REACT_APP_API_URL=http://agents.enableyou.co:5000
   # - All OAuth URLs use agents.enableyou.co
   ```

4. **Start services:**
   ```bash
   ./start.sh
   ```

5. **Verify endpoints:**
   ```bash
   curl http://agents.enableyou.co:5000/auth/google/start
   curl http://agents.enableyou.co/  # Frontend via nginx
   ```

## Critical Configuration Files

### Backend: `tools/.env`

Controls backend behavior and OAuth credentials:

```env
ENVIRONMENT=production
PUBLIC_URL=http://agents.enableyou.co:5000
GOOGLE_CLIENT_ID=your_id
GOOGLE_CLIENT_SECRET=your_secret
GOOGLE_REDIRECT_URI=http://agents.enableyou.co:5000/auth/google/callback
OAUTHLIB_INSECURE_TRANSPORT=1
OPENAI_API_KEY=your_key
```

### Frontend: `agent-app/.env`

Used during React build process ONLY:

```env
REACT_APP_API_URL=http://agents.enableyou.co:5000
```

**CRITICAL**: Changing this file requires rebuild:
```bash
cd agent-app
npm run build
sudo cp -r build/* /usr/share/nginx/html/
```

### Backend Config Module: `tools/config.py`

Validates environment and provides single source of truth:

```python
from config import Config

# Access validated configuration
print(Config.PUBLIC_URL)  # Validated, no defaults
print(Config.ENVIRONMENT)  # development or production
print(Config.GOOGLE_REDIRECT_URI)  # Auto-constructed from PUBLIC_URL

# On startup, prints:
# BACKEND CONFIGURATION
# Environment: production
# Public URL: http://agents.enableyou.co:5000
# ...
```

## Deployment Checklist

Before starting services, verify:

### 1. Configuration Files Exist

```bash
# Check both .env files are present and readable
ls -la tools/.env
ls -la agent-app/.env
```

### 2. URLs Match Across All Files

**Locally verify:**
```bash
# These should all match:
grep REACT_APP_API_URL agent-app/.env
grep PUBLIC_URL tools/.env
grep GOOGLE_REDIRECT_URI tools/.env
```

**Example for localhost:**
```
REACT_APP_API_URL=http://localhost:5000
PUBLIC_URL=http://localhost:5000
GOOGLE_REDIRECT_URI=http://localhost:5000/auth/google/callback
```

**Example for remote:**
```
REACT_APP_API_URL=http://agents.enableyou.co:5000
PUBLIC_URL=http://agents.enableyou.co:5000
GOOGLE_REDIRECT_URI=http://agents.enableyou.co:5000/auth/google/callback
```

### 3. Credentials Set

```bash
# Verify these are set (not showing defaults)
grep "^GOOGLE_CLIENT_ID=" tools/.env | grep -v "your_"
grep "^OPENAI_API_KEY=" tools/.env | grep -v "your_"
```

### 4. OAuth Callbacks Registered

In [Google Cloud Console](https://console.cloud.google.com/):
1. Go to Credentials
2. Click your OAuth Client ID
3. Verify these URLs are listed under "Authorized redirect URIs":
   - `http://localhost:5000/auth/google/callback` (if localhost)
   - `http://agents.enableyou.co:5000/auth/google/callback` (if remote)

### 5. React Build Updated (After Config Changes)

**IMPORTANT**: React builds environment variables into static files:

```bash
# After changing agent-app/.env, rebuild:
cd agent-app
npm run build

# For remote deployment, copy to nginx:
sudo cp -r build/* /usr/share/nginx/html/

# Verify nginx is serving updated files:
curl -I http://agents.enableyou.co/
```

## Troubleshooting Configuration Issues

### Issue: "REACT_APP_API_URL not set"

**Cause**: `agent-app/.env` missing or not rebuilt

**Fix**:
```bash
# 1. Verify .env exists
cat agent-app/.env | grep REACT_APP_API_URL

# 2. Rebuild React app
cd agent-app
npm run build

# 3. For remote, copy to nginx
sudo cp -r build/* /usr/share/nginx/html/
```

### Issue: "PUBLIC_URL environment variable not set"

**Cause**: `tools/.env` missing or corrupted

**Fix**:
```bash
# 1. Verify .env exists
cat tools/.env | grep PUBLIC_URL

# 2. Check format is correct (no spaces around =)
grep "^PUBLIC_URL=" tools/.env

# 3. Restart backend
./stop.sh
./start.sh
```

### Issue: OAuth Callback Mismatch Error

**Cause**: URLs don't match between `.env` and Google Console

**Fix**:
```bash
# 1. Check what's configured locally
cat tools/.env | grep "GOOGLE_REDIRECT_URI\|PUBLIC_URL"

# 2. Verify matches Google Console:
#    https://console.cloud.google.com/
#    Credentials → Your OAuth Client ID → Authorized redirect URIs

# 3. Update if needed and restart
./stop.sh
./start.sh
```

### Issue: Frontend calling wrong backend URL

**Cause**: React build stale after `agent-app/.env` change

**Fix**:
```bash
# 1. Verify .env value
cat agent-app/.env

# 2. Check browser Network tab sees right URL
#    (Should show agent-app/.env value, not old value)

# 3. Force rebuild
cd agent-app
npm run build

# 4. Clear browser cache
#    Ctrl+Shift+Delete (or Cmd+Shift+Delete on Mac)
#    Clear "Cached images and files"

# 5. Restart for good measure
cd ..
./stop.sh
./start.sh
```

## Configuration Security

### DO's ✓

- ✓ Keep `.env` files **OUT of git** (.gitignore exists)
- ✓ Use **same credentials** for OAuth across environments
- ✓ Store **secrets securely** (never in code)
- ✓ **Document** required variables (see .env.example)
- ✓ **Validate** variables on startup (config.py does this)

### DON'Ts ✗

- ✗ Commit `.env` files to git
- ✗ Hardcode URLs in code (use config modules)
- ✗ Mix localhost and remote URLs
- ✗ Have different OAuth credentials per environment
- ✗ Forget to rebuild React after `.env` changes

## File Organization Summary

```
enable_agents/
├── .env                          # Root config (shared reference)
├── .env.development             # Template for localhost setup
├── .env.production              # Template for remote setup
├── .gitignore                   # .env files excluded ✓
│
├── tools/
│   ├── .env                     # Backend config (from root/.env)
│   ├── config.py                # Backend validation & Config class
│   └── app.py                   # Uses Config class
│
├── agent-app/
│   ├── .env                     # Frontend config (from root/.env)
│   ├── src/config/
│   │   ├── apiConfig.js         # Uses process.env.REACT_APP_API_URL
│   │   └── environment.config.js# Environment utilities
│   └── package.json             # Build uses .env vars
│
├── start.sh                     # Validates .env before starting
├── DEPLOYMENT_CONFIG.md         # Complete setup guide
├── CONFIGURATION.md             # This file (best practices)
└── DEPLOYMENT.md                # (Existing deployment docs)
```

## Next Steps

1. **Copy appropriate .env template:**
   - Localhost: `cp .env.development .env`
   - Remote: `cp .env.production .env` (on remote)

2. **Customize for your environment:**
   - Update OAuth credentials
   - Update URLs to match your setup
   - Add any missing secrets

3. **Start services:**
   - `./start.sh`
   - Watch for any configuration errors
   - Check logs for detailed status

4. **Verify endpoints:**
   - Backend: `curl http://<PUBLIC_URL>:5000/auth/google/start`
   - Frontend: Browser to `http://<PUBLIC_URL>`

5. **Test OAuth flow:**
   - Click "Login with Google"
   - Should redirect to Google
   - Should return without errors
