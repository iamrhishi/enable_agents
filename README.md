# Enable Agents

A full-stack multi-agent platform — Flask backend, React frontend, MySQL, Redis.

## Quick start

```bash
./scripts/install-prerequisites.sh    # first time: Docker Engine + Compose (Linux: sudo; macOS: Homebrew cask or instructions)
cp .env.docker.example .env.docker   # fill in API keys
./run.sh dev                          # start dev stack (same as ./scripts/run.sh dev)
```

## All run commands

| Command | What it does |
|---|---|
| `./scripts/install-prerequisites.sh` | One-time: Docker + Compose v2 (`docker compose`). Linux uses Docker’s official installer; macOS uses Homebrew cask or a download link |
| `./run.sh dev` | Same as `./scripts/run.sh dev` — Docker dev stack from repo root |
| `./scripts/run.sh dev` | Docker: mysql + redis + backend (hot-reload) + frontend (npm dev) |
| `./scripts/run.sh prod` | Docker: mysql + redis + backend (gunicorn) + frontend (nginx) |
| `./scripts/run.sh stop` | Stop all Docker services |
| `./scripts/run.sh test` | Run all tests (no Docker needed — auto-creates venv) |
| `./scripts/run.sh test docker` | Run tests inside the running dev container |
| `./scripts/run.sh local` | Non-Docker fallback: venv + npm start |
| `./scripts/run.sh local-stop` | Stop non-Docker services |

## Dev URLs (after `./scripts/run.sh dev`)

| Service | URL |
|---|---|
| Frontend | http://localhost:3000 |
| Backend API | http://localhost:8000 |
| Flower (task monitor) | http://localhost:5555 |
| Redis UI | http://localhost:8081 |
| MySQL | localhost:3306 |

**API URL for the React app:** Docker dev serves the API on **port 8000**. If you run `npm start` on the host (not in the frontend container), `frontend/src/config/apiConfig.js` defaults to `http://localhost:8000` unless `REACT_APP_API_URL` is set. For non-Docker `./scripts/run.sh local`, `frontend/.env` is generated with **port 5000** to match the local Flask backend from `start.sh`. Override anytime with `REACT_APP_API_URL` (then restart `npm start`).

If the browser shows **`ERR_CONNECTION_REFUSED`** on port **8000**, the backend is not up: start **Docker Desktop**, run **`./run.sh dev`**, and wait — the script polls **`/health`** for up to ~3 minutes. If it still fails, run `docker compose -f docker-compose.yml --profile dev logs --tail=100 backend-dev` (often MySQL/env issues).

## Folder structure

```
run.sh             Wrapper — calls scripts/run.sh (use ./run.sh dev from repo root)
backend/
  agents/          One folder per agent (manifest.json, routes.py, service.py, models.py, tasks.py)
  blueprints/      Core route groups (auth, health, prompts, favorites)
  core/            Shared infrastructure (database, celery, logging, models)
frontend/
  src/agents/      Agent UI components
  src/core/        Shared UI (Header, Login, Register)
scripts/
  install-prerequisites.sh  Docker + Compose (Linux/macOS one-time setup)
  run.sh           Main entry point (dev / prod / test / local)
  start.sh         Start services (non-Docker)
  stop.sh          Stop services (non-Docker)
  deploy.sh        Remote deployment over SSH
  setup_https.sh   SSL certificate + nginx setup
docs/              context.md (source of truth) · todo.md
tests/
  integration/     Per-agent and per-blueprint integration tests
  sanity/          Structural sanity checks
docker-compose.yml Single file — dev and prod profiles
```

## Branching

| Branch | Purpose |
|---|---|
| `develop` | Main integration branch — branch off here for new features |
| `harsh-code` | Stable reference branch |

## Documentation

- Architecture and decisions: `docs/context.md`
- Work backlog: `docs/todo.md`
