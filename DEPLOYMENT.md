# Enable Agents - Deployment & Management Scripts

This directory contains three utility scripts for managing the Enable Agents application on a remote server.

## Scripts Overview

### 1. `setup_db.sh` - Database Setup
Sets up the database on a remote server after pulling code from git.

**When to use:**
- After initial deployment
- When updating database schema
- Before running the application for the first time

**Usage:**
```bash
./setup_db.sh
```

**What it does:**
1. Creates a Python virtual environment (if needed)
2. Installs Python dependencies from `tools/requirements.txt`
3. Initializes the database schema
4. Displays success message

---

### 2. `start.sh` - Start Services
Starts both the React frontend and Python backend in the background.

**Usage:**
```bash
./start.sh
```

**What it does:**
1. Activates the Python virtual environment
2. Installs React dependencies (if needed)
3. Starts the React app (`npm start`) on port 3000
4. Starts the Python backend (`python app.py`) on port 5000
5. Creates log files in `.logs/` directory
6. Saves process IDs for the stop script

**Services:**
- **Frontend**: http://localhost:3000
- **Backend**: http://localhost:5000

**Log files:**
- React: `.logs/react.log`
- Python: `.logs/python.log`

---

### 3. `stop.sh` - Stop Services
Stops both the React frontend and Python backend services.

**Usage:**
```bash
./stop.sh
```

**What it does:**
1. Reads saved process IDs from `.pids` file
2. Gracefully terminates both services
3. Force kills if necessary
4. Cleans up PID file

---

## Quick Start - Remote Deployment

After pulling code from git on your remote server:

```bash
# 1. Set up database
./setup_db.sh

# 2. Start services
./start.sh

# 3. Check logs
tail -f .logs/react.log
tail -f .logs/python.log

# 4. When done, stop services
./stop.sh
```

---

## Monitoring

### View Live Logs

React frontend logs:
```bash
tail -f .logs/react.log
```

Python backend logs:
```bash
tail -f .logs/python.log
```

### Check Running Processes

```bash
ps aux | grep -E "npm|python"
```

---

## Troubleshooting

### Services fail to start

1. **Check logs:**
   ```bash
   cat .logs/react.log
   cat .logs/python.log
   ```

2. **Verify ports are available:**
   ```bash
   lsof -i :3000  # Check React port
   lsof -i :5000  # Check Python port
   ```

3. **Kill existing processes:**
   ```bash
   ./stop.sh
   # OR manually
   pkill -f "npm start"
   pkill -f "python app.py"
   ```

### Database setup fails

1. Ensure Python 3 is installed: `python3 --version`
2. Check database permissions
3. View detailed error: `cat .logs/python.log`

### Virtual environment issues

Remove and recreate:
```bash
rm -rf venv/
./setup_db.sh
```

---

## Files Created

- `.logs/react.log` - React frontend logs
- `.logs/python.log` - Python backend logs
- `.pids` - Stores process IDs (deleted on stop)
- `.logs/` - Log directory

---

## Notes

- The `start.sh` script will automatically install dependencies if needed
- Both services run in background - use `.stop.sh` to terminate
- Logs are rotated/recreated each time services start
- The Python virtual environment is only created on first run
- Services require Python 3.8+ and Node.js 14+

