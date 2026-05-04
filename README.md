# Enable Agents

A full-stack multi-agent platform — Flask backend, React frontend, MySQL, Redis.

## Quick start

```bash
cp .env.docker.example .env.docker   # fill in API keys
./scripts/run.sh dev                  # start dev stack (hot-reload)
```

## All run commands

| Command | What it does |
|---|---|
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
| Backend API | http://localhost:5000 |
| Flower (task monitor) | http://localhost:5555 |
| Redis UI | http://localhost:8081 |
| MySQL | localhost:3306 |

## Folder structure

```
backend/
  agents/          One folder per agent (manifest.json, routes.py, service.py, models.py, tasks.py)
  blueprints/      Core route groups (auth, health, prompts, favorites)
  core/            Shared infrastructure (database, celery, logging, models)
frontend/
  src/agents/      Agent UI components
  src/core/        Shared UI (Header, Login, Register)
scripts/
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
