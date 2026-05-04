#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
BACKEND_DIR="$PROJECT_ROOT/backend"
FRONTEND_DIR="$PROJECT_ROOT/frontend"
VENV_DIR="$PROJECT_ROOT/venv"
COMPOSE="docker compose -f $PROJECT_ROOT/docker-compose.yml"

usage() {
  cat <<'EOF'
Usage:
  ./scripts/run.sh dev          # Docker: mysql + redis + backend (hot-reload) + frontend (npm dev)
  ./scripts/run.sh prod         # Docker: mysql + redis + backend (gunicorn) + frontend (nginx)
  ./scripts/run.sh stop         # Stop all Docker services
  ./scripts/run.sh test         # Run all tests (no Docker needed — uses a local venv)
  ./scripts/run.sh test docker  # Run all tests inside the running dev container
  ./scripts/run.sh local        # Non-Docker: venv + npm start (fallback / CI)
  ./scripts/run.sh local-stop   # Stop non-Docker local services
EOF
}

DEV_PORTS=(8000 3000 3306 6379 5555 8081)

ensure_docker() {
  if docker info >/dev/null 2>&1; then
    return 0
  fi
  echo "Docker is not running. Starting Docker Desktop..."
  open -a Docker 2>/dev/null || true
  local waited=0
  while ! docker info >/dev/null 2>&1; do
    printf "."
    sleep 2
    waited=$((waited + 2))
    if [ "$waited" -ge 60 ]; then
      echo ""
      echo "Docker did not start within 60s. Please start Docker Desktop manually and retry."
      exit 1
    fi
  done
  echo " Docker is ready."
}

free_ports() {
  echo "Freeing required ports..."
  for port in "${DEV_PORTS[@]}"; do
    pids=$(lsof -ti :"$port" 2>/dev/null || true)
    if [ -n "$pids" ]; then
      echo "  Killing process(es) on port $port: $pids"
      kill -9 $pids 2>/dev/null || true
    fi
  done
  sleep 1
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
  # Install only the test-time dependencies (lighter than full backend stack)
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
