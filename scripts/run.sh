#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
BACKEND_DIR="$PROJECT_ROOT/backend"
FRONTEND_DIR="$PROJECT_ROOT/frontend"
VENV_DIR="$PROJECT_ROOT/venv"
COMPOSE="docker compose -f $PROJECT_ROOT/docker-compose.yml"
INSTALL_SCRIPT="$SCRIPT_DIR/install-prerequisites.sh"
OS="$(uname -s)"

usage() {
  cat <<'EOF'
Usage:
  ./scripts/run.sh dev          # Docker: mysql + redis + backend (hot-reload) + frontend (npm dev)
  ./scripts/run.sh prod         # Docker: mysql + redis + backend (gunicorn) + frontend (nginx)
  ./scripts/run.sh stop         # Stop all Docker services
  ./scripts/run.sh test         # Run all tests (no Docker needed — uses a local venv)
  ./scripts/run.sh test docker  # Run tests inside the running dev container
  ./scripts/run.sh local        # Non-Docker: venv + npm start (fallback / CI)
  ./scripts/run.sh local-stop   # Stop non-Docker local services

First-time Docker (Linux — installs Engine + Compose; may prompt for sudo):
  ./scripts/install-prerequisites.sh
EOF
}

DEV_PORTS=(8000 3000 3306 6379 5555 8081)

# Free one TCP port: lsof (macOS/Linux), else fuser (Linux psmisc), else skip.
kill_port_listeners() {
  local port="$1"
  local pids=""
  if command -v lsof >/dev/null 2>&1; then
    pids=$(lsof -ti :"$port" 2>/dev/null || true)
    if [ -n "$pids" ]; then
      echo "  Killing process(es) on port $port: $pids"
      # shellcheck disable=SC2086
      kill -9 $pids 2>/dev/null || true
    fi
    return 0
  fi
  if command -v fuser >/dev/null 2>&1; then
    if fuser "$port/tcp" >/dev/null 2>&1; then
      echo "  Freeing port $port (fuser)..."
      fuser -k "$port/tcp" 2>/dev/null || true
    fi
    return 0
  fi
  echo "  Warning: cannot free port $port — install lsof or psmisc (fuser)." >&2
}

free_ports() {
  echo "Freeing required ports..."
  for port in "${DEV_PORTS[@]}"; do
    kill_port_listeners "$port"
  done
  sleep 1
}

ensure_docker_installed() {
  if command -v docker >/dev/null 2>&1 && docker compose version >/dev/null 2>&1; then
    return 0
  fi
  echo "Docker CLI or Docker Compose v2 plugin not found."
  if [ ! -f "$INSTALL_SCRIPT" ]; then
    echo "Missing $INSTALL_SCRIPT"
    exit 1
  fi
  echo "Running one-time setup: $INSTALL_SCRIPT"
  bash "$INSTALL_SCRIPT" || {
    echo "Install script failed. Fix the errors above or install Docker manually:"
    echo "  https://docs.docker.com/engine/install/"
    exit 1
  }
  if ! command -v docker >/dev/null 2>&1 || ! docker compose version >/dev/null 2>&1; then
    echo "Docker still not available. On Linux, you may need to log out and back in after being added to the 'docker' group, then run:"
    echo "  newgrp docker"
    exit 1
  fi
}

ensure_docker_daemon() {
  if docker info >/dev/null 2>&1; then
    return 0
  fi

  case "$OS" in
    Darwin)
      echo "Docker daemon not responding. Starting Docker Desktop..."
      open -a Docker 2>/dev/null || true
      ;;
    Linux)
      echo "Docker daemon not running. Trying to start it (may ask for sudo)..."
      if command -v systemctl >/dev/null 2>&1; then
        sudo systemctl start docker 2>/dev/null || true
      fi
      if ! docker info >/dev/null 2>&1 && command -v service >/dev/null 2>&1; then
        sudo service docker start 2>/dev/null || true
      fi
      ;;
    *)
      echo "Start the Docker service manually, then retry."
      exit 1
      ;;
  esac

  local waited=0
  while ! docker info >/dev/null 2>&1; do
    printf "."
    sleep 2
    waited=$((waited + 2))
    if [ "$waited" -ge 120 ]; then
      echo ""
      echo "Docker did not become ready in time."
      case "$OS" in
        Darwin)
          echo "Open Docker Desktop and wait until it finishes starting, then retry."
          ;;
        Linux)
          echo "Try: sudo systemctl status docker"
          echo "Ensure your user can run docker (group 'docker'), or use: newgrp docker"
          ;;
      esac
      exit 1
    fi
  done
  echo " Docker is ready."
}

ensure_docker() {
  ensure_docker_installed
  ensure_docker_daemon
}

check_env_docker() {
  if [ ! -f "$PROJECT_ROOT/.env.docker" ]; then
    echo "Missing .env.docker — copy from .env.docker.example and fill in values."
    exit 1
  fi
}

check_env_local() {
  if [ ! -f "$PROJECT_ROOT/.env" ]; then
    echo "Missing .env — copy from .env.example and fill in values."
    exit 1
  fi
  cp "$PROJECT_ROOT/.env" "$BACKEND_DIR/.env"
  cat > "$FRONTEND_DIR/.env" <<EOF
REACT_APP_API_URL=http://localhost:5000
EOF
}

ensure_venv() {
  if [ ! -d "$VENV_DIR" ]; then
    echo "Creating Python virtual environment..."
    python3 -m venv "$VENV_DIR"
  fi
  # shellcheck disable=SC1091
  source "$VENV_DIR/bin/activate"
  pip install -q --upgrade pip
  pip install -q flask flask-sqlalchemy flask-migrate flask-cors werkzeug \
    celery redis pytest python-dotenv requests
}

setup_local() {
  check_env_local
  ensure_venv
  pip install -q -r "$BACKEND_DIR/requirements.txt"
  if [ ! -d "$FRONTEND_DIR/node_modules" ]; then
    (cd "$FRONTEND_DIR" && npm install)
  fi
}

run_tests_local() {
  echo "Setting up test environment..."
  ensure_venv
  echo ""
  echo "Running tests..."
  "$VENV_DIR/bin/python" -m pytest "$PROJECT_ROOT/tests/" -v "$@"
}

run_tests_docker() {
  echo "Running tests inside the dev container..."
  $COMPOSE exec backend-dev pytest tests/ -v "$@"
}

case "${1:-}" in
  dev)
    check_env_docker
    ensure_docker
    free_ports
    $COMPOSE --profile dev up --build -d
    echo ""
    echo "Dev stack starting:"
    echo "  Frontend:       http://localhost:3000"
    echo "  Backend:        http://localhost:8000"
    echo "  Flower (tasks): http://localhost:5555"
    echo "  Redis UI:       http://localhost:8081"
    echo "  MySQL:          localhost:3306"
    ;;
  prod)
    check_env_docker
    ensure_docker
    free_ports
    $COMPOSE --profile prod up --build -d
    echo ""
    echo "Prod stack starting:"
    echo "  Frontend: http://localhost"
    echo "  Backend:  http://localhost:8000"
    echo "  MySQL:    localhost:3306"
    ;;
  stop)
    $COMPOSE down
    ;;
  test)
    shift
    if [ "${1:-}" = "docker" ]; then
      shift
      run_tests_docker "$@"
    else
      run_tests_local "$@"
    fi
    ;;
  local)
    setup_local
    "$PROJECT_ROOT/scripts/start.sh"
    ;;
  local-stop)
    "$PROJECT_ROOT/scripts/stop.sh"
    ;;
  *)
    usage
    exit 1
    ;;
esac
