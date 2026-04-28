# Enable Agents — Context (Source of Truth)

## Product purpose
Enable Agents is a multi-agent platform for business workflows. It combines a Flask API backend with a React frontend, MySQL database, Redis cache/queue, and a plug-and-play agent architecture.

## Repository structure

```
backend/           Flask API, agents, core infrastructure
  agents/          One folder per agent (manifest.json, routes.py, service.py, models.py)
  core/            Shared code (database, auth, users, storage, logging, celery)
  data/            Runtime data (gitignored)
frontend/          React application
  src/agents/      Agent UI components (mirrors backend/agents/)
  src/core/        Shared UI (Header, Login, Register)
scripts/           run.sh · start.sh · stop.sh · setup_db.sh
docs/              context.md · todo.md
tests/             sanity/ + future unit tests
docker-compose.yml Single file, dev and prod profiles
```

## Runtime modes

### Docker (recommended)
```bash
./scripts/run.sh dev    # hot-reload, dev profile
./scripts/run.sh prod   # built images, gunicorn + nginx, prod profile
./scripts/run.sh stop   # tear down
```

Docker Compose profiles:
- `dev`: backend-dev (flask --reload, volume mount), frontend-dev (npm start, volume mount), celery-worker-dev
- `prod`: backend (gunicorn 4 workers), frontend (nginx, built), celery-worker
- `mysql` and `redis` have no profile — always start in both modes

### Non-Docker (fallback / CI)
```bash
./scripts/run.sh local       # venv + npm start
./scripts/run.sh local-stop
```

## Environment files

| File | Purpose | Committed |
|---|---|---|
| `.env.example` | Template for local non-Docker use | yes |
| `.env` | Local secrets | no |
| `.env.docker.example` | Template for Docker | yes |
| `.env.docker` | Docker runtime secrets | no |

`backend/.env` is synced from root `.env` by `./scripts/run.sh local`.
`frontend/.env` is generated with `REACT_APP_API_URL=http://localhost:5000`.

## Shared agent context (context lake)

Agents communicate through a two-tier shared store (`backend/core/context.py`), never by importing each other's models directly.

| Tier | Backend | Lifetime | Use |
|---|---|---|---|
| Fast | Redis | TTL (default 2h, env `CONTEXT_TTL_SECONDS`) | Session state, in-flight results |
| Persistent | MySQL `agent_context` table | Forever | Company profiles, user prefs, research results |

Usage in agent service code:

```python
from core.context import ContextStore

ctx = ContextStore()
ctx.set(user_id, agent_id="market_research", key="company_profile", value={...})
profile = ctx.get(user_id, "company_profile")   # Redis-first, MySQL fallback
snapshot = ctx.snapshot(user_id)                # all keys for this user
ctx.clear(user_id)                              # logout / full reset
```

Each agent's `manifest.json` declares its data contract:

```json
{
  "provides": { "company_profile": { "industry": "string" } },
  "consumes": ["user_profile"]
}
```

- **`provides`** — keys this agent writes to the shared context
- **`consumes`** — keys this agent reads from other agents

The registry validates this at startup and warns if a `consumes` key has no provider.  
`GET /api/v1/agents/context-graph` returns the full provides/consumes dependency graph.

---

## Plug-and-play agent architecture

Each agent lives in `backend/agents/<name>/` and carries:
- `manifest.json` — id, name, description, enabled flag, routes_prefix, required env vars, input schema, provides/consumes context contracts
- `routes.py` — Flask Blueprint
- `service.py` — business logic
- `models.py` — SQLAlchemy models
- `tasks.py` — Celery async tasks (optional)

`backend/agents/registry.py` scans manifests at startup, registers only enabled agents, and exposes `GET /api/v1/agents`.

To add a new agent: drop a folder into `backend/agents/`, create the manifest, restart backend.
To disable: set `"enabled": false` in the manifest (or toggle via `PATCH /api/v1/agents/<id>`).
To port to another repo: copy the agent folder and import its Blueprint.

## Database migrations

Flask-Migrate (Alembic). Each agent defines its SQLAlchemy models in `models.py`; the shared `db` instance in `backend/core/database.py` discovers them at import time.

```bash
flask db migrate -m "description"   # auto-generate migration
flask db upgrade                     # apply pending migrations
flask db downgrade                   # roll back one step
```

Docker dev starts with `flask db upgrade` automatically.

## Standards
- All shell scripts live in `scripts/`
- Only two doc files: `docs/context.md` and `docs/todo.md`
- Minimal root `README.md`
- API routes prefixed `/api/v1/`
