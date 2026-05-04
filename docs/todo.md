# Enable Agents — Work Backlog

## P0 — Critical (done)

- [x] Rename `tools/` → `backend/`, `agent-app/` → `frontend/`
- [x] Rewrite `docker-compose.yml` with `dev` and `prod` profiles (mysql + redis always-on)
- [x] Add Celery + Redis services to compose
- [x] Simplify `scripts/run.sh` to wrap `docker compose --profile`
- [x] Create `backend/core/` — database.py (SQLAlchemy + Flask-Migrate), celery_app.py, logging_config.py
- [x] Convert `app.py` from module-level app to `create_app()` factory
- [x] Add Flask-Migrate (Alembic) replacing `db.create_all()`
- [x] Create `backend/agents/registry.py` — plug-and-play agent loader
- [x] Create `content_marketing` agent package (manifest.json, routes.py, service.py, models.py, tasks.py)
- [x] Create `frontend/src/core/` (Header, Login, Register)
- [x] Create `frontend/src/agents/` and `agentRegistry.js`
- [x] Update `AgentsAssembly.js` to load from registry API
- [x] Update sanity tests (backend/ frontend/ folder checks, compose profiles, run.sh commands)
- [x] All sanity tests green (9/9)

## P1 — High priority

- [ ] Generate and apply first Alembic migration (`flask db migrate -m "initial"`)
- [ ] Fully migrate remaining raw sqlite3 tables in `app.py` to SQLAlchemy models in `backend/agents/`
- [ ] Split `app.py` routes into blueprints (auth, files, scraping, email, google, linkedin)
- [ ] Add `/api/v1/` prefix to all legacy routes
- [ ] Wire `register_agents()` in `app.py` — currently defined in `agents/registry.py` but never called; wiring it requires removing the duplicate monolith routes first
- [ ] Remove duplicate `search_google_businesses` route registered twice in `app.py` (~line 6261)
- [ ] Content Marketing: pick one persistence layer — currently both raw sqlite3 (app.py) and SQLAlchemy (agents/content_marketing/models.py) exist for the same feature; consolidate and delete the other
- [ ] Create a `backend/agents/email_outreach/` agent (manifest + routes + models + tasks)
- [ ] Create a `backend/agents/market_research/` agent
- [ ] Add `PATCH /api/v1/agents/<id>` persistent toggle (write back to manifest.json or DB)
- [ ] Add frontend admin UI for enabling/disabling agents
- [ ] Frontend: standardise on one HTTP client — currently most components use `fetch` but `AgentsAssembly.js` uses `axios`; pick one and create a shared `apiClient` wrapper
- [ ] `DataInsights.js`: extract shared `runRagOnFile(file, prompt)` helper — `handleGetInsights` and `handleBulkGetInsights` duplicate the upload→RAG flow

## P2 — Medium priority

- [ ] Rotate all API keys exposed in `.env` file (OPENAI_API_KEY, GOOGLE creds, Twilio etc.)
- [ ] Add structured logging to all existing routes
- [ ] Add Celery Beat for scheduled tasks (email report digests, data refresh)
- [ ] Add `flower` service to docker-compose for Celery monitoring
- [ ] Add `redis-commander` or similar for dev visibility
- [ ] Add integration tests per agent (happy path + error cases)
- [ ] Add Dockerfile entrypoint that runs `flask db upgrade` before gunicorn

## P3 — Nice to have

- [ ] Add connection pooling config for high-concurrency agents
- [ ] Add agent versioning (v field in manifest)
- [ ] Create a developer guide in `docs/agent-development.md`
- [ ] Add pre-commit hooks (black, ruff, eslint)
- [ ] Add GitHub Actions CI (lint + sanity tests + docker build)
