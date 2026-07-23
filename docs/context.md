# Enable Agents — Context (Source of Truth)

**Last updated:** 2026-07-22

## Product purpose
Enable Agents is a multi-agent platform for business workflows. It combines a Flask API backend with a React frontend, PostgreSQL database (with pgvector for embeddings), Redis cache/queue, and a plug-and-play agent architecture.

## Landing Page
- **Route:** `/dashboard` (default after login)
- **Purpose:** Hybrid landing page showing both workflows and agents
- **Sections:**
  - Hero with stats (active workflows, completed, available agents)
  - Start a Workflow (3 featured templates)
  - Quick Actions (6 AI agents)
  - Recent Activity (recent workflow executions)

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
7. **Business-friendly language only** — No technical jargon (entity extraction → Key Facts, knowledge graph → Visual Connections)

### Business-Friendly Language (Phase 1 Principle)

**Goal:** Platform must be accessible to business users without technical background.

**Avoid Technical Jargon:**
| ❌ Technical | ✅ Business-Friendly |
|-------------|---------------------|
| Entity extraction | Key Facts / Important Information |
| Knowledge graph | Visual Connections / Connections Map |
| RAG (Retrieval Augmented Generation) | Smart Search / Intelligent Search |
| Embeddings / Vectors | (Hide completely - internal only) |
| Confidence: 87% | Confidence: High (with color indicator) |
| Nodes • Edges | Items • Connections |
| Processing chunks | Analyzing your document... |
| Vector similarity | Finding related information... |

**Writing Guidelines:**
- Use active, clear language: "Find suppliers" not "Execute supplier discovery"
- Explain what happens, not how: "Analyze documents" not "Extract entities and build knowledge graph"
- Use progress language: "Analyzing..." not "Vectorizing chunks..."
- Show outcomes: "Found 15 key facts" not "Extracted 15 entities with 0.85 confidence"

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

### Supplier Qualification Pipeline ✅ IMPLEMENTED

The primary demo workflow with 6 stages - **stages named to match what agents actually do**:

| Stage | ID | Agent | Route | What It Does |
|-------|-----|-------|-------|--------------|
| 1. Supplier Discovery | `supplier_discovery` | requirements_gathering | /market-research | Search & find suppliers |
| 2. Document Analysis | `document_analysis` | market_research | /data-insights | Analyze supplier docs |
| 3. RFQ Outreach | `rfq_outreach` | email_outreach | /email-outreach | Send RFQ emails |
| 4. Response Analysis | `response_analysis` | sales_helper | /sales-helper | Track & rank responses |
| 5. Qualification Audit | `qualification_audit` | supply_chain | /supply-chain-agent | Audit suppliers |
| 6. Selection Tasks | `selection_tasks` | executive_assistant | /executive-assistant | Manage selection tasks |

#### ⚠️ CRITICAL: Agent Naming Confusion (July 2026 Audit)

**Issue**: Agent ID `market_research` routes to DataInsights (`/data-insights`), not Market Research agent.
- `requirements_gathering` → routes to `/market-research` ✅ CORRECT
- `market_research` → routes to `/data-insights` ❌ **CONFUSING NAME**

**Impact**: Developers/users confused about which agent does what.

**Fix Required**: Rename `market_research` agent ID → `data_insights` in:
- `WorkflowRunner.js` AGENT_CONFIG
- `backend/config/workflow-templates/supplier-qualification.json`

**Platform Status**: 7/10 — All agents support workflows properly, data persistence works, but naming must be fixed before production.

**WorkflowExecutionBanner** (`src/components/WorkflowExecutionBanner.js`):
- Minimal context indicator showing workflow name and stage
- "Back to Workflow" link for navigation
- Does NOT display data - agents show data in their own UI
- Compact design: 8px padding, inline layout, 16px icons

**Agent Config** (`src/workflows/WorkflowRunner.js`):
```javascript
const AGENT_CONFIG = {
  requirements_gathering: { route: '/market-research', label: 'Supplier Discovery', type: 'agent' },
  market_research: { route: '/data-insights', label: 'Document Analysis', type: 'agent' },  // ⚠️ RENAME TO data_insights
  email_outreach: { route: '/email-outreach', label: 'RFQ Outreach', type: 'agent' },
  sales_helper: { route: '/sales-helper', label: 'Response Analysis', type: 'agent' },
  supply_chain: { route: '/supply-chain-agent', label: 'Qualification Audit', type: 'agent' },
  executive_assistant: { route: '/executive-assistant', label: 'Selection Tasks', type: 'agent' },
};
```

### Workflow Data Flow & Auto-Save ✅ IMPLEMENTED

**useWorkflowContext Hook** (`src/hooks/useWorkflowContext.js`):
```javascript
const { isInWorkflow, saveStageData, stageData, workflowInstance } = useWorkflowContext();

// When agent completes work, save data:
await saveStageData({
  businesses_found: 15,
  top_results: [...],
  search_query: "..."
});
```

**Backend Auto-Save** (`POST /api/workflows/instances/:id/stages/:stageId/data`):
- Saves to `stageStates[stageId].data` in WorkflowInstance
- Updates workflow context for next stages
- PATCH merges data, POST replaces

**Agent Integration Pattern**:
```javascript
// All 6 workflow agents follow this pattern:

// 1. Import hook
import { useWorkflowContext } from '../hooks/useWorkflowContext';

// 2. Get workflow state
const { isInWorkflow, saveStageData, stageData } = useWorkflowContext();

// 3. Load history data (if viewing completed stage)
useEffect(() => {
  if (stageData) {
    // Load saved data into agent UI
    setResults(stageData.top_results || []);
  }
}, [stageData]);

// 4. Save actual computed results (NOT hardcoded)
const handleComplete = async () => {
  await saveStageData({
    // Use actual/computed values from agent execution
    businesses_found: results.length,
    confidence_score: calculateConfidence(results),
  });
};

// 5. Conditional UI for workflow context
{isInWorkflow && <WorkflowExecutionBanner />}
{!isInWorkflow && <BackButton />}
```

**Data Examples by Agent**:
- **RequirementsGathering**: `businesses_found`, `top_results`, `search_query`
- **DataInsights**: `document_analyzed`, `findings`, `confidence_score` (computed from sources)
- **EmailOutreachAgent**: `emails_sent`, `email_subject`, `email_body`, `recipients`
- **SalesHelperAgent**: `prospects_matched`, `vendor_ranking`, `match_scores`
- **SupplyChainAgent**: `audit_score`, `passed_audit`, `supplier_evaluated`, `category_scores`
- **ExecutiveAssistant**: `tasks_created`, `completion_rate` (computed from actual tasks)

---

## Workflow Context Visualization

### WorkflowContextCard Component

**Purpose:** Display data from previous workflow stages to show data lineage and context flow.

**Location:** `/frontend/src/components/WorkflowContextCard.js`

**Usage:**
```jsx
import { WorkflowContextCard } from '../components';

// In workflow-aware agent component
const { getContext, stageId } = useWorkflowContext();

// Render context card
{isInWorkflow && !isHistoryView && (
  <WorkflowContextCard context={getContext()} currentStageId={stageId} />
)}
```

**Features:**
- Stage-specific context rendering
- Shows relevant data from completed previous stages
- Blue gradient design matching platform theme
- Animated slide-in effect
- Displays business names, metrics, and key findings

**Implemented in:**
- RequirementsGathering (stage 1)
- DataInsights (stage 2) 
- EmailOutreachAgent (stage 3)
- SalesHelperAgent (stage 4)
- SupplyChainAgent (stage 5)
- ExecutiveAssistantPage (stage 6)

---

## Navigation & Routing

### Main Routes

| Route | Component | Purpose |
|-------|-----------|---------|
| `/` | Redirects to `/dashboard` | Root redirect |
| `/dashboard` | Dashboard | Landing page (workflows + agents) |
| `/agents` | AgentsAssembly | All agents catalog |
| `/workflows` | WorkflowsPage | Workflow templates & instances |
| `/workflows/:id` | WorkflowRunner | Execute workflow |
| `/market-research` | RequirementsGathering | Market research agent |
| `/email-outreach` | EmailOutreachAgent | Email campaigns |
| `/supply-chain-agent` | SupplyChainAgent | Supplier audits |
| `/executive-assistant` | ExecutiveAssistantPage | Task management |
| `/data-insights` | DataInsights | Document analysis |
| `/sales-helper` | SalesHelperAgent | Lead qualification |
| `/projects` | Projects | Project management |
| `/team` | Team | Team management |
| `/settings` | Settings | User settings |

### Header Navigation

User dropdown menu (top-right):
- Dashboard
- Agents
- Workflows  
- Projects
- Team
- Settings
- Sign Out

### BackButton Component

**Default behavior:** Returns to `/dashboard`

**Workflow-aware:** If `?workflow=<id>` param present, returns to workflow

**Usage:**
```jsx
<BackButton />                          // Goes to /dashboard
<BackButton to="/agents" />             // Custom destination
<BackButton to="back" />                // Browser back
```

---

## Recent Major Changes (2026-07-22)

### 1. Hybrid Dashboard Landing Page
- Created `/dashboard` as new landing page
- Shows both workflow templates and quick action agents
- Stats cards with theme colors
- Recent activity section
- Replaced `/agents-assembly` as default route

### 2. Workflow Data Flow Fix
- Added `WorkflowContextCard` component
- All workflow agents now display previous stage data
- Context shown at top of each stage
- Removed dummy data, using actual workflow context

### 3. Theme Consistency
- Updated Dashboard to use design tokens
- Primary: #1E3A5F (blue)
- Accent: #C2410C (orange)
- Background: #F1EAE4 (warm beige)
- Removed purple/bright gradients

### 4. Realistic Test Data
- Created `populate_realistic_workflow.py` script
- Populated workflow with automotive supplier data
- 5 realistic suppliers with complete details
- Data flows through all 6 stages
- Workflow ID: `065b4298-5b8e-4122-b325-b7cb798c7f41`

---

## Testing Workflows End-to-End

### Demo Workflow

**ID:** `065b4298-5b8e-4122-b325-b7cb798c7f41`  
**Name:** "Apex Manufacturing - CNC Housing Sourcing"  
**Owner:** rajeshdarak1991@gmail.com  
**Status:** Completed  
**Template:** Supplier Qualification Pipeline

**Data Flow:**
1. **Supplier Discovery** → 5 suppliers found (India, China)
2. **Document Analysis** → 12 documents analyzed, 94% confidence
3. **RFQ Outreach** → 5 emails sent to suppliers
4. **Response Analysis** → 4 quotes received, Shenzhen AutoTech top match (95%)
5. **Qualification Audit** → 3 audited, 2 passed, Shenzhen scored 92/100
6. **Selection Tasks** → Contract finalized: $99,000 (50k units @ $1.98/unit)

**URL:** http://localhost:3000/workflows/065b4298-5b8e-4122-b325-b7cb798c7f41

### Running the Populate Script

```bash
# Execute inside Docker container
docker compose exec backend-dev python3 /app/scripts/populate_realistic_workflow.py
```

**Script creates:**
- Realistic supplier data (Precision Circuits India, Shenzhen AutoTech, etc.)
- Complete stage-by-stage data
- Proper timestamps and completion statuses
- Contextual data flowing between stages
