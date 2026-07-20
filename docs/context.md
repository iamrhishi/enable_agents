# Enable Agents — Context (Source of Truth)

**Last updated:** 2026-07-20

## Product purpose
Enable Agents is a multi-agent platform for business workflows. It combines a Flask API backend with a React frontend, PostgreSQL database (with pgvector for embeddings), Redis cache/queue, and a plug-and-play agent architecture.

## Repository structure

```
backend/
  agents/              One folder per agent
  core/
    connectors/        External data source integrations
    context.py         Shared context layer
    settings.py        User settings (encrypted)
    vector_store.py    pgvector embeddings
    database.py        SQLAlchemy + migrations
  data/                Runtime data (gitignored)
frontend/
  src/
    styles/
      tokens.css       Design tokens (SINGLE SOURCE OF TRUTH)
    agents/            Agent UI components
    core/              Shared UI (Header, Login)
    settings/          Settings page
scripts/               run.sh · start.sh · stop.sh
docs/                  context.md · todo.md
docker-compose.yml
```

---

## Design System

### Architecture

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                    tokens.css — Single Source of Truth                      │
│  ┌───────────────────────────────────────────────────────────────────────┐ │
│  │  Colors       Typography      Spacing       Shadows       Components  │ │
│  │  --color-*    --font-*        --space-*     --shadow-*    .btn-*     │ │
│  │  --gradient-* --text-*        --radius-*    --transition-* .input    │ │
│  └───────────────────────────────────────────────────────────────────────┘ │
│                                    │                                        │
│                                    ▼                                        │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │  index.css — imports tokens, sets base styles                       │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
│                                    │                                        │
│  ┌─────────────────────────────────┼─────────────────────────────────────┐ │
│  │         ▼           ▼           ▼           ▼           ▼             │ │
│  │  Settings.css  Login.css  Header.css  AgentsAssembly  agent-shell    │ │
│  │  (cards UI)    (uses vars) (uses vars) (uses vars)    (uses vars)    │ │
│  └───────────────────────────────────────────────────────────────────────┘ │
└─────────────────────────────────────────────────────────────────────────────┘
```

### Brand Palette

| Token | Value | Usage |
|-------|-------|-------|
| `--color-primary` | #1E3A5F | Headers, primary text |
| `--color-accent` | #C2410C | CTAs, active states, highlights |
| `--color-accent-dark` | #B45309 | Hover states |
| `--color-background` | #F1EAE4 | Page background |
| `--color-surface` | #FFFFFF | Cards, panels |
| `--color-border` | #D6C7B8 | Borders, dividers |

### Typography

| Token | Value | Usage |
|-------|-------|-------|
| `--font-display` | Fraunces | Headings h1-h3 |
| `--font-body` | IBM Plex Sans | Body text, UI |
| `--text-sm` | 0.875rem (14px) | Default body |
| `--text-base` | 1rem (16px) | Large body |

### Component Classes (from tokens.css)

```css
/* Buttons */
.btn              /* Base button styles */
.btn-primary      /* Accent gradient, white text */
.btn-secondary    /* Background color, bordered */
.btn-danger       /* Red background for destructive */
.btn-sm           /* Smaller padding */

/* Inputs */
.input            /* Standard input with max-width: 400px */
.input-full       /* Full width input */

/* Cards */
.card             /* White bg, bordered, rounded */
.card-elevated    /* With shadow */

/* Badges */
.badge-success    /* Green */
.badge-error      /* Red */
.badge-warning    /* Yellow */
```

### Rules

1. **Never hardcode colors** — Always use `var(--color-*)` or `var(--gradient-*)`
2. **Never hardcode fonts** — Always use `var(--font-display)` or `var(--font-body)`
3. **Use spacing scale** — `var(--space-2)` through `var(--space-12)`
4. **Use radius tokens** — `var(--radius-sm)` through `var(--radius-full)`
5. **Use shadow tokens** — `var(--shadow-sm)` through `var(--shadow-xl)`
6. **Constrain input widths** — `max-width: var(--input-max-width)` (400px)

---

## Connector Configuration Architecture

### Current State (IMPLEMENTED ✅)

```
┌─────────────────────────────────────────────────────────────────┐
│                    SETTINGS UI (Frontend)                        │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐              │
│  │ AI Provider │  │ Connectors  │  │  Proxies    │              │
│  │ OpenAI key  │  │ Google Auth │  │ Scraping    │              │
│  │ Anthropic   │  │ LinkedIn    │  │ Rate limits │              │
│  └──────┬──────┘  └──────┬──────┘  └──────┬──────┘              │
│         └────────────────┼────────────────┘                      │
│                          ▼                                       │
│              ┌───────────────────────┐                           │
│              │  POST /api/settings   │                           │
│              └───────────┬───────────┘                           │
└──────────────────────────┼───────────────────────────────────────┘
                           ▼
┌──────────────────────────────────────────────────────────────────┐
│                   SETTINGS STORAGE                                │
│  ┌─────────────────────────────────────────────────────────────┐ │
│  │  user_settings table (Fernet encrypted)                     │ │
│  │  ─────────────────────────────────────────────────────────  │ │
│  │  user_id │ category    │ key           │ value (encrypted)  │ │
│  │  user1   │ ai          │ openai_key    │ sk-xxx...         │ │
│  │  user1   │ connectors  │ google_tokens │ {access, refresh} │ │
│  │  user1   │ scraping    │ proxy_url     │ http://...        │ │
│  └─────────────────────────────────────────────────────────────┘ │
└──────────────────────────────────────────────────────────────────┘
                           ▼
┌──────────────────────────────────────────────────────────────────┐
│                   CONNECTOR USAGE                                 │
│                                                                   │
│  connector = get_connector("web_scraper", user_id=user_id)       │
│  # Connector reads user's proxy settings from user_settings      │
│  connector.connect()  # Uses user-specific config                │
└──────────────────────────────────────────────────────────────────┘
```

### Configuration Categories

| Category | Settings | Storage |
|----------|----------|---------|
| **AI Providers** | OpenAI key, Anthropic key, model preferences | Encrypted in DB |
| **OAuth Connectors** | Google, LinkedIn, Salesforce tokens | Encrypted in DB |
| **API Key Connectors** | HubSpot, SendGrid, Search API keys | Encrypted in DB |
| **Scraping Config** | Proxy URL, rate limits, user agent | Encrypted in DB |
| **System Defaults** | Fallback keys (admin only) | .env file |

---

## Data Flow Architecture

```
┌─────────────────────────────────────────────────────────────────────────┐
│                           CONNECTORS                                     │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐    │
│  │ WebScraper  │  │GoogleBusiness│  │  WebSearch  │  │  LinkedIn   │    │
│  └──────┬──────┘  └──────┬──────┘  └──────┬──────┘  └──────┬──────┘    │
│         │                │                │                │            │
│         │    ┌───────────┴───────────┐    │                │            │
│         │    │  User Settings Store  │◄───┴────────────────┘            │
│         │    └───────────────────────┘                                  │
│         │                │                                              │
│         └────────────────┼──────────────────────────────────────────────│
│                          ▼                                              │
│              ┌───────────────────────┐                                  │
│              │  ctx.store_from_source │                                 │
│              └───────────┬───────────┘                                  │
└──────────────────────────┼──────────────────────────────────────────────┘
                           ▼
┌──────────────────────────────────────────────────────────────────────────┐
│                         CONTEXT STORE                                     │
│  ┌─────────────┐    ┌─────────────┐    ┌─────────────┐                   │
│  │   Redis     │    │ PostgreSQL  │    │  pgvector   │                   │
│  └─────────────┘    └─────────────┘    └─────────────┘                   │
└──────────────────────────────────────────────────────────────────────────┘
```

---

## Implemented Connectors

| Connector | Type | Status | Config Required |
|-----------|------|--------|-----------------|
| **web_scraper** | Session | ✅ Done | Proxy URL (optional) |
| **google_business** | OAuth | ✅ Done | OAuth flow via UI |
| **web_search** | API Key | ✅ Done | Google/Bing API key |
| **linkedin** | OAuth | ✅ Done | OAuth flow via UI |
| **hubspot** | API Key | ❌ TODO | HubSpot API key |

---

## Context Store API

### Flexible "ask" Interface (reading)

```python
from core.context import ContextStore
ctx = ContextStore()

docs = ctx.ask(user_id, "documents")
context = ctx.ask(user_id, "document_context", query="pricing")
web = ctx.ask(user_id, "web_search")
external = ctx.ask(user_id, "external")
all_data = ctx.ask(user_id, "all")
```

### Store Patterns (writing)

```python
# Agent results
ctx.store_result(user_id, agent_id, result_type, result_id, data)

# External data (connectors use this automatically)
ctx.store_from_source(user_id, source, data, metadata)
```

---

## Agent Architecture

Each agent in `backend/agents/<name>/`:
- `manifest.json` — configuration
- `routes.py` — Flask Blueprint
- `service.py` — business logic
- `models.py` — SQLAlchemy models
- `tasks.py` — Celery tasks

Agents use:
- `get_connector()` — fetch external data
- `ContextStore.ask()` — read context
- `ContextStore.store_result()` — write results

---

## Runtime

```bash
./scripts/run.sh dev    # Development
./scripts/run.sh prod   # Production
./scripts/run.sh stop   # Stop
```

---

## Agent Dependency Graph

```
┌─────────────────────────────────────────────────────────────────────────┐
│                    AGENT DATA DEPENDENCIES                               │
│                                                                         │
│  market_research ──provides──▶ company_profile, prospect_list           │
│         │                                                               │
│         ▼                                                               │
│  content_marketing ──consumes──▶ company_profile, user_profile          │
│         │                                                               │
│         ▼                                                               │
│  email_outreach ──consumes──▶ company_profile, prospect_list,           │
│                               user_profile                              │
│                                                                         │
│  document_intelligence ──consumes──▶ user_profile                       │
│                                                                         │
│  ⚠️ GAP: user_profile consumed by 3 agents but NO agent provides it    │
└─────────────────────────────────────────────────────────────────────────┘
```

### Dependency Enforcement Status

| Component | Status | Location |
|-----------|--------|----------|
| Manifest declarations | ✅ Done | `backend/agents/*/manifest.json` |
| Registry loading | ✅ Done | `backend/agents/registry.py` |
| Runtime validation | ✅ Done | `backend/core/dependency_validator.py` |
| API endpoint | ✅ Done | `GET /api/dependencies/status/<agent_id>` |
| Frontend gating | ✅ Done | `frontend/src/components/AgentPrerequisiteGate.js` |
| Config file | ✅ Done | `backend/config/agent-dependencies.json` |

---

## Known Gaps (July 2026 Audit) — MOSTLY RESOLVED

| Issue | Status | Notes |
|-------|--------|-------|
| **Executive Assistant backend** | ✅ DONE | Routes, service, models at `backend/agents/executive_assistant/` |
| **Dependency enforcement** | ✅ DONE | Validator with warn/strict modes, frontend gate |
| **Workflow Templates** | ✅ DONE | Full CRUD, state machine, progress UI |
| **CI/CD Pipeline** | ✅ DONE | GitHub Actions, Playwright E2E tests |
| **Projects persistence** | ✅ DONE | SQLAlchemy models in `core/models.py` |
| **Teams persistence** | ✅ DONE | SQLAlchemy models in `core/models.py` |
| **No user_profile provider** | ⚠️ TODO | Settings fallback exists, dedicated provider needed |
| **RequirementsGathering.js** | ✅ FIXED | Restored from git commit 51846a4b |

---

## Workflow Templates System ✅ IMPLEMENTED

### Architecture

```
┌─────────────────────────────────────────────────────────────────────────┐
│                     WORKFLOW TEMPLATE SYSTEM                             │
│                                                                         │
│  Backend:                                                               │
│  ├── backend/models/workflow.py — SQLAlchemy models                     │
│  ├── backend/routes/workflows.py — CRUD + state transitions             │
│  └── backend/config/workflow-templates/*.json — System templates        │
│                                                                         │
│  Frontend:                                                              │
│  ├── frontend/src/workflows/WorkflowsPage.js — Template listing         │
│  ├── frontend/src/workflows/WorkflowRunner.js — Active workflow UI      │
│  └── frontend/src/workflows/WorkflowProgress.js — Progress tracker      │
│                                                                         │
│  API Endpoints:                                                         │
│  ├── GET/POST /api/workflows/templates — Template CRUD                  │
│  ├── GET/POST /api/workflows/instances — Instance management            │
│  ├── POST /api/workflows/instances/:id/start — Start workflow           │
│  ├── POST /api/workflows/instances/:id/complete-stage — Advance         │
│  └── POST /api/workflows/instances/:id/pause — Pause workflow           │
└─────────────────────────────────────────────────────────────────────────┘
```

### UX Features

- **Progress bar** with "X / Y stages completed"
- **Timeline view** showing completed (✓), current (number), pending stages
- **Current stage panel** with description, required inputs, and actions
- **Context sidebar** showing data collected from completed stages
- **State machine** with pending → running → paused → completed flow

### Current Limitations (Potential LangGraph Migration)

- **Linear progression only** — No conditional branching
- **Forward-only** — Cannot go back to previous stages
- **Manual agent invocation** — User must manually run agent, then mark complete
- **No agent output piping** — Stage outputs don't auto-feed into next agent

**LangGraph would add:** Conditional routing, parallel execution, checkpointing, automatic agent composition, interrupt/resume from any point.
