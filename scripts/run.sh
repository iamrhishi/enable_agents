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

# Port configuration (single source of truth)
PORT_BACKEND=8000
PORT_BACKEND_OAUTH=5000
PORT_FRONTEND=3000
PORT_MYSQL=3306
PORT_REDIS=6379
PORT_FLOWER=5555
PORT_REDIS_UI=8081

usage() {
  cat <<'EOF'
Usage:
  ./run.sh dev                    # Docker: mysql + redis + backend (hot-reload) + frontend (npm dev)
  ./run.sh prod                   # Docker: mysql + redis + backend (gunicorn) + frontend (nginx)
  ./run.sh remote                 # Docker: production with nginx reverse proxy (GCP/AWS)
  ./run.sh remote-ssl             # Setup Let's Encrypt SSL (requires DOMAIN in .env.docker)
  ./run.sh stop                   # Stop all Docker services
  ./run.sh test                   # Run all tests (no Docker needed — uses a local venv)
  ./run.sh test docker            # Run tests inside the running dev container
  ./run.sh local                  # Non-Docker: venv + npm start (fallback / CI)
  ./run.sh local-stop             # Stop non-Docker local services
  ./run.sh logs [service]         # View logs (all or specific service)

Remote Deployment (GCP/AWS):
  1. Edit .env.docker: set DEPLOY_MODE=remote, SERVER_IP, DOMAIN (optional)
  2. Run: ./run.sh remote
  3. (Optional) Setup SSL: ./run.sh remote-ssl

First-time Docker (Linux — installs Engine + Compose; may prompt for sudo):
  ./scripts/install-prerequisites.sh
EOF
}

DEV_PORTS=($PORT_BACKEND $PORT_BACKEND_OAUTH $PORT_FRONTEND $PORT_MYSQL $PORT_REDIS $PORT_FLOWER $PORT_REDIS_UI)

# Host listeners for published container ports are often docker-proxy / Desktop forwarders.
# kill -9 on those breaks the Docker engine until Docker Desktop is restarted.
is_docker_port_forwarder() {
  local pid="$1"
  local args comm
  args=$(ps -p "$pid" -o args= 2>/dev/null || echo "")
  comm=$(ps -p "$pid" -o comm= 2>/dev/null || echo "")
  case "$comm" in
    *docker-proxy*) return 0 ;;
  esac
  local args_lower
  args_lower=$(printf '%s' "$args" | tr '[:upper:]' '[:lower:]')
  case "$args_lower" in
    *docker-proxy*) return 0 ;;
    *vpnkit*) return 0 ;;
    *com.docker*) return 0 ;;
    */applications/docker.app/*) return 0 ;;
    *orbstack*) return 0 ;;
  esac
  return 1
}

compose_down_clean() {
  echo "Stopping any running project containers (safe port release)..."
  # Both profiles so all services are known; removes published ports without killing Docker internals.
  $COMPOSE --profile dev --profile prod down --remove-orphans 2>/dev/null || true
  sleep 2
}

# Free one TCP port: lsof (macOS/Linux), else fuser (Linux psmisc), else skip.
kill_port_listeners() {
  local port="$1"
  local pids=""
  local pid
  if command -v lsof >/dev/null 2>&1; then
    pids=$(lsof -ti :"$port" 2>/dev/null || true)
    if [ -n "$pids" ]; then
      for pid in $pids; do
        if is_docker_port_forwarder "$pid"; then
          echo "  Skipping Docker-managed listener on port $port (pid $pid) — use ./run.sh stop or compose down to release." >&2
          continue
        fi
        echo "  Killing process(es) on port $port: $pid"
        kill -9 "$pid" 2>/dev/null || true
      done
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
      # Socket can appear before the engine accepts API/pulls; avoid a false "ready".
      sleep 5
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
          echo "Open Docker Desktop and wait until the whale icon is idle, then run:"
          echo "  ./run.sh dev"
          ;;
        Linux)
          echo "Try: sudo systemctl status docker"
          echo "Ensure your user can run docker (group 'docker'), or use: newgrp docker"
          ;;
      esac
      exit 1
    fi
  done
  # Brief stability window: Desktop sometimes reports ready before pulls work.
  local stable=0
  while [ "$stable" -lt 3 ]; do
    if docker info >/dev/null 2>&1; then
      stable=$((stable + 1))
    else
      stable=0
    fi
    sleep 2
  done
  echo " Docker is ready."
}

# Re-check before compose (free_ports / Desktop restarts can race the daemon).
wait_for_docker_engine() {
  local max_s="${1:-90}"
  local s=0
  while [ "$s" -lt "$max_s" ]; do
    if docker info >/dev/null 2>&1; then
      return 0
    fi
    sleep 1
    s=$((s + 1))
  done
  echo "Docker daemon is not responding (waited ${max_s}s). Open Docker Desktop, wait until it is idle, then retry:" >&2
  echo "  docker info" >&2
  echo "  ./run.sh dev" >&2
  exit 1
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

# Generic HTTP readiness check.
wait_for_http() {
  local url="${1:?URL required}"
  local max_s="${2:-60}"
  local s=0
  while [ "$s" -lt "$max_s" ]; do
    if curl -fsS "$url" >/dev/null 2>&1; then
      echo "  ✓ $url is ready"
      return 0
    fi
    sleep 2
    s=$((s + 2))
  done
  echo "  ✗ $url not ready after ${max_s}s"
  return 1
}

# After compose up, confirm the API accepts connections (avoids "connection refused" in the browser).
wait_for_backend_http() {
  local url="${1:-http://127.0.0.1:8000/health}"
  local max_s="${2:-180}"
  local s=0
  echo "Waiting for backend HTTP (${url})..."
  while [ "$s" -lt "$max_s" ]; do
    if curl -fsS "$url" >/dev/null 2>&1; then
      echo "Backend is reachable on port ${PORT_BACKEND}."
      return 0
    fi
    sleep 2
    s=$((s + 2))
    printf "."
  done
  echo ""
  echo "Backend never became reachable at ${url} within ${max_s}s."
  echo "Inspect logs:"
  echo "  $COMPOSE logs --tail=80 backend-dev"
  return 1
}

# Run database migrations inside backend container.
run_migrations() {
  local container="${1:-backend-dev}"
  echo "Running database migrations..."
  $COMPOSE exec -T "$container" flask db upgrade 2>&1 || {
    echo "Warning: Migrations failed or no migrations to run."
  }
}

# Load environment variables from .env.docker
load_env() {
  if [ -f "$PROJECT_ROOT/.env.docker" ]; then
    set -a
    source "$PROJECT_ROOT/.env.docker"
    set +a
  fi
}

# Setup nginx config based on DOMAIN/SERVER_IP
setup_nginx_config() {
  local nginx_dir="$PROJECT_ROOT/deploy/nginx"
  mkdir -p "$nginx_dir"

  load_env

  if [ -n "${DOMAIN:-}" ] && [ -d "/etc/letsencrypt/live/${DOMAIN}" ]; then
    echo "Using SSL nginx config for $DOMAIN"
    export DOMAIN SERVER_IP
    envsubst '${DOMAIN} ${SERVER_IP}' < "$nginx_dir/nginx.conf" > "$nginx_dir/active.conf"
  else
    echo "Using HTTP-only nginx config"
    cp "$nginx_dir/nginx-http.conf" "$nginx_dir/active.conf"
  fi
}

# Update PUBLIC_URL and OAuth redirect URIs based on deployment mode
update_public_url() {
  load_env
  local public_url

  if [ -n "${DOMAIN:-}" ]; then
    if [ -d "/etc/letsencrypt/live/${DOMAIN}" ]; then
      public_url="https://${DOMAIN}"
    else
      public_url="http://${DOMAIN}"
    fi
  elif [ -n "${SERVER_IP:-}" ]; then
    public_url="http://${SERVER_IP}"
  else
    echo "Warning: Neither DOMAIN nor SERVER_IP set in .env.docker"
    return 1
  fi

  echo "Setting PUBLIC_URL to: $public_url"

  # Update .env.docker with correct PUBLIC_URL
  if grep -q "^PUBLIC_URL=" "$PROJECT_ROOT/.env.docker"; then
    sed -i.bak "s|^PUBLIC_URL=.*|PUBLIC_URL=${public_url}|" "$PROJECT_ROOT/.env.docker"
    sed -i.bak "s|^GOOGLE_REDIRECT_URI=.*|GOOGLE_REDIRECT_URI=${public_url}/auth/google/callback|" "$PROJECT_ROOT/.env.docker"
    sed -i.bak "s|^REACT_APP_API_URL=.*|REACT_APP_API_URL=${public_url}|" "$PROJECT_ROOT/.env.docker"
    rm -f "$PROJECT_ROOT/.env.docker.bak"
  fi
}

case "${1:-}" in
  dev)
    check_env_docker
    ensure_docker
    compose_down_clean
    free_ports
    wait_for_docker_engine 90
    $COMPOSE --profile dev up --build -d
    echo ""
    wait_for_backend_http "http://127.0.0.1:${PORT_BACKEND}/health" 180 || {
      echo ""
      echo "Fix backend startup, then run: ./run.sh dev  (or ./scripts/run.sh dev)"
      exit 1
    }
    run_migrations "backend-dev"
    echo ""
    echo "Waiting for services to become ready..."
    sleep 3
    wait_for_http "http://127.0.0.1:${PORT_FRONTEND}" 120 || {
      echo "Frontend not ready. Check logs: $COMPOSE logs frontend-dev"
    }
    wait_for_http "http://127.0.0.1:${PORT_BACKEND_OAUTH}/health" 30 || {
      echo "Backend OAuth port (${PORT_BACKEND_OAUTH}) not ready."
    }
    echo ""
    echo "Dev stack ready:"
    echo "  Frontend:       http://localhost:${PORT_FRONTEND}"
    echo "  Backend:        http://localhost:${PORT_BACKEND}"
    echo "  Backend OAuth:  http://localhost:${PORT_BACKEND_OAUTH}"
    echo "  Flower (tasks): http://localhost:${PORT_FLOWER}"
    echo "  Redis UI:       http://localhost:${PORT_REDIS_UI}"
    echo "  MySQL:          localhost:${PORT_MYSQL}"
    ;;
  prod)
    check_env_docker
    ensure_docker
    compose_down_clean
    free_ports
    wait_for_docker_engine 90
    $COMPOSE --profile prod up --build -d
    wait_for_backend_http "http://127.0.0.1:${PORT_BACKEND}/health" 180 || {
      echo ""
      echo "Fix backend startup, then run: ./run.sh prod"
      exit 1
    }
    run_migrations "backend"
    echo ""
    echo "Waiting for services to become ready..."
    sleep 3
    wait_for_http "http://127.0.0.1:80" 60 || {
      echo "Frontend not ready. Check logs: $COMPOSE logs frontend"
    }
    echo ""
    echo "Prod stack ready:"
    echo "  Frontend: http://localhost"
    echo "  Backend:  http://localhost:${PORT_BACKEND}"
    echo "  MySQL:    localhost:${PORT_MYSQL}"
    ;;
  remote)
    check_env_docker
    ensure_docker
    load_env

    # Validate remote config
    if [ -z "${SERVER_IP:-}" ] && [ -z "${DOMAIN:-}" ]; then
      echo "Error: Set SERVER_IP or DOMAIN in .env.docker for remote deployment"
      exit 1
    fi

    update_public_url
    setup_nginx_config

    compose_down_clean
    wait_for_docker_engine 90

    echo "Starting remote deployment..."
    $COMPOSE --profile remote up --build -d

    wait_for_backend_http "http://127.0.0.1:${PORT_BACKEND}/health" 180 || {
      echo ""
      echo "Fix backend startup, then run: ./run.sh remote"
      exit 1
    }
    run_migrations "backend-remote"

    echo ""
    echo "Waiting for services to become ready..."
    sleep 5
    wait_for_http "http://127.0.0.1:80" 60 || {
      echo "Nginx/Frontend not ready. Check logs: $COMPOSE logs nginx frontend-remote"
    }

    load_env
    echo ""
    echo "Remote stack ready:"
    if [ -n "${DOMAIN:-}" ]; then
      echo "  Frontend: https://${DOMAIN} (after SSL) or http://${DOMAIN}"
    fi
    if [ -n "${SERVER_IP:-}" ]; then
      echo "  Frontend: http://${SERVER_IP}"
    fi
    echo ""
    echo "Next: Update Google OAuth redirect URI to: ${PUBLIC_URL}/auth/google/callback"
    ;;
  remote-ssl)
    load_env

    if [ -z "${DOMAIN:-}" ]; then
      echo "Error: DOMAIN not set in .env.docker"
      exit 1
    fi
    if [ -z "${ADMIN_EMAIL:-}" ]; then
      echo "Error: ADMIN_EMAIL not set in .env.docker (required for Let's Encrypt)"
      exit 1
    fi

    echo "Setting up SSL for ${DOMAIN}..."

    # Ensure HTTP nginx is running for ACME challenge
    cp "$PROJECT_ROOT/deploy/nginx/nginx-http.conf" "$PROJECT_ROOT/deploy/nginx/active.conf"
    $COMPOSE --profile remote up -d nginx
    sleep 5

    # Run certbot
    echo "Requesting certificate from Let's Encrypt..."
    $COMPOSE run --rm certbot certonly \
      --webroot \
      --webroot-path=/var/www/certbot \
      --email "${ADMIN_EMAIL}" \
      --agree-tos \
      --no-eff-email \
      -d "${DOMAIN}"

    # Switch to SSL config and reload
    update_public_url
    setup_nginx_config
    $COMPOSE --profile remote up -d --build

    echo ""
    echo "SSL setup complete!"
    echo "  Site available at: https://${DOMAIN}"
    echo ""
    echo "Update Google OAuth redirect URI to: https://${DOMAIN}/auth/google/callback"
    ;;
  stop)
    if ! docker info >/dev/null 2>&1; then
      echo "Docker daemon is not running — nothing to stop via Compose (or start Docker Desktop and run stop again)."
      exit 0
    fi
    $COMPOSE --profile dev --profile prod --profile remote down --remove-orphans
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
  logs)
    shift
    $COMPOSE logs --tail=100 "$@"
    ;;
  logs-f)
    shift
    $COMPOSE logs -f "$@"
    ;;
  status)
    $COMPOSE ps
    ;;
  *)
    usage
    exit 1
    ;;
esac
