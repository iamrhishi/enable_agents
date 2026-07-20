# Enable Agents — Work Backlog

## Phase 1 — Testable MVP ✅ COMPLETE

### All Working
- [x] Document upload/processing (PDF, DOCX, TXT, XLSX, CSV)
- [x] Document chat (RAG)
- [x] Connector architecture (web_scraper, google_business, web_search, linkedin)
- [x] Connector API endpoints (`/api/connectors/*`)
- [x] Context storage (auto-stored via connectors)
- [x] Settings UI — users configure API keys, OAuth, proxies
- [x] Settings API — store/retrieve user settings (encrypted)
- [x] Connectors use UserSettings — read config from user settings
- [x] Cross-agent data sharing — verified via ContextStore

---

## Design System ✅ IMPLEMENTED

### Completed
- [x] **tokens.css created** — Single source of truth for all design values
- [x] **index.css updated** — Imports tokens, sets base styles
- [x] **Settings.css fixed** — Uses brand colors, constrained input widths
- [x] **Settings.js updated** — Back button, connector cards UI (industry standard)
- [x] **Login.css updated** — Uses tokens
- [x] **Header.css updated** — Removed duplicate :root, uses tokens
- [x] **AgentsAssembly.css updated** — Removed duplicate :root, uses tokens
- [x] **agent-shell.css updated** — Uses tokens
- [x] **ContentMarketingAgent.css updated** — Uses tokens, removed duplicates

### Files Created/Modified
| File | Purpose |
|------|---------|
| `src/styles/tokens.css` | Design tokens (colors, typography, spacing, shadows, component classes) |
| `src/index.css` | Base styles using tokens |
| `src/settings/Settings.css` | Fixed to use tokens, brand palette, connector cards |
| `src/settings/Settings.js` | Back button, connector cards UI |
| `src/styles/Login.css` | Uses tokens |
| `src/styles/Header.css` | Uses tokens |
| `src/styles/AgentsAssembly.css` | Uses tokens |
| `src/styles/agent-shell.css` | Uses tokens |
| `src/styles/ContentMarketingAgent.css` | Uses tokens |

### Key Tokens
```css
--color-primary: #1E3A5F      /* Deep ink blue */
--color-accent: #C2410C       /* Burnt ember */
--color-background: #F1EAE4   /* Paper warm */
--color-border: #D6C7B8       /* Soft clay */
--font-display: 'Fraunces'    /* Headings */
--font-body: 'IBM Plex Sans'  /* Body text */
--input-max-width: 400px      /* Input constraint */
```

### Component Classes Available
- `.btn` `.btn-primary` `.btn-secondary` `.btn-danger` `.btn-sm`
- `.input` `.input-full`
- `.card` `.card-elevated`
- `.badge-success` `.badge-error` `.badge-warning`

### Industry standard — theme, color, typography (planning)

**Verdict:** Keep the **Enable palette** in `tokens.css` (ink blue + warm paper + ember accent + Fraunces/IBM Plex). It is valid for B2B SaaS if applied with discipline. Do **not** introduce a second theme (e.g. Settings blue banner, Executive Assistant purple/green, login photo treatment as a separate “brand”).

#### Color system (semantic tokens — industry pattern)

Enterprise products (Atlassian, Linear, Stripe Dashboard, Material 3) separate **brand** from **semantic** colors:

| Role | Your token(s) | Use for | Do not use for |
|------|---------------|---------|----------------|
| **Brand primary** | `--color-primary` | Headings, nav text, secondary buttons outline | Body paragraphs, large backgrounds |
| **Brand accent** | `--color-accent` | Primary CTA, active tab, focus ring, links | Success/error, decorative gradients everywhere |
| **Canvas** | `--color-background` | Page background (all authenticated pages) | Card interiors |
| **Surface** | `--color-surface` | Cards, modals, inputs, header bar | Full-page fill |
| **Border** | `--color-border` | Dividers, card outline | Text |
| **Text default** | `--color-text` | Body, labels | — |
| **Text muted** | `--color-text-muted` | Descriptions, metadata | Primary actions |
| **Success / Warning / Error** | `--color-success*` etc. | Status only | Brand CTAs, navigation |

**Rules (WCAG 2.1 AA):**
- Body text on `--color-surface` or `--color-background`: ≥ **4.5:1** contrast (`--color-text` on white/paper passes; `--color-text-subtle` only for captions ≥14px bold or ≥18px regular).
- Primary button: `--color-text-inverse` on `--color-accent` (verify accent orange meets 4.5:1 with white text — bump to `#9A3412` hover if needed).
- One accent hue only for interactive emphasis; status colors never compete with CTA orange.
- **No new hex in components** — extend `tokens.css` if a shade is missing.

**Settings / Login exceptions to fix:**
- Settings `settings-header` full bleed `--color-primary` → use **same white/paper header** as Agents Assembly + in-page title.
- Login full-bleed photography → OK for marketing; inside the card use **same** `--font-*`, `--color-*`, `.btn-primary` as app.

#### Typography (2-font system — industry standard)

| Token | Font | Use | Size scale |
|-------|------|-----|------------|
| `--font-display` | Fraunces | Page titles, section headings, modal titles | `--text-title` (24px), optional `--text-3xl` for marketing only |
| `--font-body` | IBM Plex Sans | UI, tables, buttons, inputs, chat | `--text-body` (16px) default; `--text-small` (14px) labels |
| `--font-mono` | IBM Plex Mono | JSON/debug, code snippets only | `--text-small` |

**Rules:**
- **Max 2 families** in product UI (already correct). No Arial/Inter overrides in agent CSS.
- **Type scale:** Prefer 3 sizes in app UI: 24 / 16 / 14 (your `--text-title`, `--text-body`, `--text-small`). Deprecate ad-hoc `0.95rem`, `1.08em` in modals.
- **Weight:** 600 for headings, 500 for labels, 400 for body; 700 only for primary CTA text.
- **Line height:** `--leading-normal` (1.5) body; `--leading-tight` (1.25) card titles.
- Load fonts once in `tokens.css` (already); `font-display: swap` on link tag.

#### Layout & density (enterprise SaaS norm)

- **8px spacing grid** — use only `--space-*` (4, 8, 12, 16, 24, 32).
- **Content max-width** — `--content-max-width` (1280px) for Settings, Assembly, Profile; full width only for data tables.
- **Header height** — 56–64px fixed; same on every page.
- **Border radius** — cards `--radius-lg`, buttons `--radius-md`, pills `--radius-full` (no mixed 8px/18px inline).

#### Component chrome (one vocabulary)

| Element | Standard class | Notes |
|---------|----------------|-------|
| Primary action | `.btn.btn-primary` | One per panel |
| Secondary | `.btn.btn-secondary` | Try, Back, Cancel |
| Destructive | `.btn.btn-danger` | Delete, Disconnect |
| Input | `.input` + `FormField` | max-width `--input-max-width` on settings forms |
| Card | `.card` / `Card` component | Same shadow, border, padding everywhere |
| Status | `StatusIndicator` icon + token color | Not text pills, not random greens |

#### What “consistent theme” means for Enable (checklist)

- [ ] Every route: `--color-background` page + white `--color-surface` header (not inverted blue strips)
- [ ] Every heading: `font-family: var(--font-display)` + `color: var(--color-text)`
- [ ] Every CTA: `.btn-primary` (accent), never purple/green one-offs (`ExecutiveAssistantPage.css`)
- [ ] Modals: `--color-surface`, `--shadow-xl`, token overlay — not `#f8fafc` / `#334155` pairs
- [ ] Gradients: only `--gradient-accent` on primary buttons; avoid gradient nav pills (Settings active nav)
- [ ] Dark mode: **out of scope** until v2; document light theme only

**References (patterns, not copying visuals):** [Material Design 3 — color roles](https://m3.material.io/styles/color/roles), [Atlassian Design — tokens](https://atlassian.design/foundations/tokens), [WCAG contrast](https://www.w3.org/WAI/WCAG21/Understanding/contrast-minimum.html).

---

## UX Polish — Remaining

### P1 — Use tokens in other files ✅ COMPLETE
- [x] Update Login.css to use tokens
- [x] Update Header.css to use tokens (remove duplicate :root)
- [x] Update AgentsAssembly.css to use tokens (remove duplicate :root)
- [x] Update agent-shell.css to use tokens
- [x] Update ContentMarketingAgent.css to use tokens

### P1 — Navigation Improvements ✅ COMPLETE
- [x] Created BackButton component (`src/components/BackButton.js`)
- [x] Added back buttons to all agent pages:
  - ContentMarketingAgent, RequirementsGathering, SalesHelperAgent
  - Chatbot, DataInsights, CommunityNetworkAgent
  - InvestAgent, SupplyChainAgent, CampaignDashboard
  - EventNetworkingAgent

### P2 — Consistency cleanup ✅ COMPLETE
- [x] Remove hardcoded font declarations (all files now use --font-body/--font-display)
- [x] Remove hardcoded color values (all files now use design tokens)
- [x] Added hover tokens: --color-success-bg-hover, --color-error-bg-hover

### Files Fixed in P2
| File | Changes |
|------|---------|
| `DataInsights.css` | Rewritten: 30+ hardcoded values → tokens |
| `Header_brand.css` | Rewritten: removed :root, 100+ token conversions |
| `SalesHelperAgent.css` | Rewritten: 14+ hardcoded values → tokens |
| `Chatbot.css` | Rewritten: 14+ hardcoded values → tokens |
| `tokens.css` | Added hover variants for status backgrounds |
| `Settings.css` | Fixed 2 remaining hardcoded hover colors |

---

## UX Redefinition — Industry Standards

**Goal:** Align all surfaces with enterprise SaaS patterns (information density, consistent components, icon-first status, predictable layout).  
**Audit date:** May 2026 — full frontend pass (all routes, Header modals, Settings, visibility rules).  
**Planning status:** ✅ **Closed** (product decisions locked below — May 2026)

### Product decisions — locked

| # | Decision | Implementation notes |
|---|----------|----------------------|
| 1 | **Google OAuth on Login and Register** | Add “Continue with Google” to `RegisterUser.js` (same flow as `Login.js` → `/auth/google/start`, callback to `/login?google_auth=success` or register-specific redirect). One account model: email from Google creates/updates `User`. |
| 2 | **Merge settings — single hub** | One authenticated destination: **`/settings`** (optional alias `/profile` → redirect to Account tab). Consolidate: API keys, connectors/OAuth, business context (ex–System modal tab 2), browser tools import (ex–Landscape), preferences. **No** separate blue-banner Settings chrome; **no** duplicate Connection modal in Header. Profile fields (name, email, avatar, Google link) = **Account** section inside Settings, not a dead dropdown toast. |
| 3 | **Real data vs demo data toggle** | **Live** \| **Demo** — **always shown to all users** (Header + Settings → Preferences). Default: **Live**. Persist `localStorage` first → user setting API later. **Live:** real APIs; stubs as **locked** cards. **Demo:** full catalog including stubs, demo badges, sample/seeded data. |
| 4 | **Stub agents in catalog** | **Live:** locked cards (visible, no Try). **Demo:** full access with demo labeling. Applies to Invest, Supply Chain, travel-agent, Requirements until shipped. |
| 5 | **Buy button** | **Keep visible.** On click: informative message (toast or small modal) — e.g. “Checkout is coming soon — we’re enabling billing for this module.” Replace `alert()` stub. No silent failure. Try remains gated by `ready` / route in Live mode. |

**Supplementary decisions (May 2026 — user confirmed):**

| Topic | Decision |
|-------|----------|
| **Live / Demo toggle visibility** | **Everyone, all the time** — always visible in Header (and mirrored in Settings → Preferences). Not dev-only. |
| **Default on first visit** | **Live** (user switches to Demo when exploring samples). |
| **Stubs in Live mode** | **Locked cards** — visible but not clickable; label “Not available yet” + no Try; cleaner than invisible for roadmap awareness. |
| **Header “system” icon** | **Keep** — slim modal: tools landscape + agent recommendations only (business context lives in Settings). |
| **Browser tools scan** | **Later than P0** — Settings section stub/“Coming soon” OK; optional seeded `tools_landscape.json` for Demo. |
| **Document Intelligence** | **Later** — no v1 catalog card; backend stays API-only until FE route planned. |
| **Requirements Gathering** | **Restore from git** first; minimal rewrite only if history unrecoverable. |
| **Prices on module cards** | **Keep** display; Buy explains checkout coming soon. |
| **After Google auth** | Redirect to **`/agents`**. |
| **Register auth methods** | **Email/password + Google** (both). |
| **Mobile v1** | **Desktop-first**; responsive pass in P3, not a P0 blocker. |
| **Demo mode data** | **Front-end seed** for stub modules + sample labels; optional server seed for tools landscape in Demo. |

### Visibility rule (product policy)

> **If it does not render real functionality, do not show it in the UI.**

| Rule | Examples found in codebase |
|------|---------------------------|
| No menu items that only toast "coming soon" | Header → Profile shows toast instead of a page |
| No Try navigation to broken routes in **Live** mode | `/travel-agent`, empty Requirements — block or explain |
| "In Progress" modules in **Live** mode | Try disabled with tooltip; visible in **Demo** mode with demo badge |
| Stub agents in **Live** mode | **Locked** card (“Not available yet”) — full card in **Demo** (decisions #3–4) |
| Buy on immature commerce | Informative toast/modal — not hidden (decision #5) |
| Hidden header icons stay removed from DOM | Connection, Landscape, Process (already commented out — keep hidden until shipped) |

**Account (auth exists):** Google OAuth on login **and** register. Account email, name, Google connection status, sign-out live under **Settings → Account** (decision #2), linked from Header dropdown.

---

### Full application audit (every route)

| Route | Component | Renders? | Global Header? | Uses shared `.card` / `Card`? | Priority issues |
|-------|-----------|----------|----------------|-------------------------------|-----------------|
| `/login` | `Login.js` | ✅ | No (auth layout) | Form card only | OK-ish; different visual language than app (full-bleed photo bg) |
| `/register` | `RegisterUser.js` | ✅ | No | Form card only | Same as login; long form without `FormField` |
| `/agents`, `/agents-assembly` | `AgentsAssembly.js` | ✅ | ✅ | Custom `module-card` (not shared) | Whitespace, inconsistent cards, text status badges, inline tab styles |
| `/settings` | `Settings.js` | ✅ | **No** — custom blue banner | `connector-card` only; AI/settings use list rows | **Theme break** vs rest of app (see below) |
| `/requirements` | `RequirementsGathering.js` | ❌ **EMPTY FILE (0 bytes)** | — | — | **Broken route** — white screen / build error |
| `/campaign-dashboard` | `CampaignDashboard.js` | ✅ | ✅ | Tables only | Inline styles; "Loading..." text; links to broken `/requirements` |
| `/datainsights` | `DataInsights.js` | ✅ | ✅ | `upload-card` in shell | Magic viewport heights; not using `Card` component |
| `/aichatbot` | `Chatbot.js` | ✅ | ✅ | Shell panels | Basic chat UI; no skeleton on load |
| `/community-network` | `CommunityNetworkAgent.js` | ✅ | ✅ | Custom HTML in chat | Inline profile cards in messages; left panel not card-based |
| `/sales-helper` | `SalesHelperAgent.js` | ✅ | ✅ | Custom sales cards | Heavy custom CSS; status text not icons |
| `/content-marketing` | `ContentMarketingAgent.js` | ✅ | ✅ | Mixed | Campaign form not card grid |
| `/event-networking-agent` | `EventNetworkingAgent.js` | ⚠️ Partial | **No Header** | Topic buttons only | Orphan page — no app chrome, only `BackButton` |
| `/invest-agent` | `InvestAgent.js` | ⚠️ Stub | ✅ | `parameter-card` | Disabled inputs + "defined soon" — **hide from catalog or gate** |
| `/supply-chain-agent` | `SupplyChainAgent.js` | ⚠️ Stub | ✅ | Placeholder div | "coming soon" copy — **hide from catalog or gate** |
| `/executive-assistant` | `ExecutiveAssistantPage.js` | ✅ | ✅ | Custom `.card-*` | 80+ non-token colors (P3 #1) |
| `/travel-agent` | — | ❌ **No route** | — | — | Linked from Agents Assembly Try/Buy — **remove or implement** |
| **Profile** | — (merged) | ❌ | — | — | **Settings → Account** tab; dropdown “Profile” → `/settings?tab=account` |

**Header-only surfaces (not routes):**

| Surface | File | Issues |
|---------|------|--------|
| **System Overview modal** | `Header.js` | Reuses `history-modal` class; tab 2 shows raw JSON in `<pre>`; tab 1 table not cards; hardcoded `#FFFFFF`, `#f8fafc`; no focus trap; no shared `Modal` component |
| **Application Landscape modal** | `Header.js` (DOM injection) | Imperative `document.createElement` — separate styling from React modals; Chrome history API often fails |
| User dropdown | `Header.js` | Profile = dead end; Settings works; Sign out OK |
| Connection modal | `Header.js` | Commented out (good) — dead code + `handleCreateConnection` still toasts "coming soon" |

---

### Global principles (apply everywhere)

| Principle | Current problem | Target pattern |
|-----------|-----------------|----------------|
| **Viewport utilization** | Large gaps between header, title, filters, and content; cards start below the fold | Compact page header (`--space-4` max below nav); sticky filter bar; content grid starts within first viewport on 1440×900 |
| **Layout grid** | Mixed inline styles, ad-hoc margins | Shared `PageLayout` wrapper: title row + toolbar row + main (single max-width, e.g. 1280px) |
| **Card system** | One-off card markup per page; Agents Assembly cards break when titles wrap | Single `ModuleCard` / `AgentCard` component: fixed regions (icon, title, status, actions) |
| **Status communication** | Text pills: "READY", "IN PROGRESS" | Icon + color + `aria-label` + tooltip (e.g. check-circle = ready, clock = in progress, lock = unavailable) |
| **Actions** | "Try" / "Buy" vary in size and placement | Primary CTA right, secondary left; same height (`--btn-height-md`) on every card |
| **Typography** | Long titles wrap and push badges out of alignment | Title: 2-line clamp + ellipsis; status icon top-right, never in flex flow with title |
| **Filters & tabs** | Inline styles in JSX; 32px+ margins between sections | Toolbar component: filters left, toggles right; tabs use design tokens only |
| **Empty / loading** | Text-only or missing | Skeleton cards in grid; `EmptyState` with one action |
| **Accessibility** | Status is color-only or uppercase text | Icons paired with `aria-label`; tooltips on hover/focus; don’t rely on color alone |

### Shared components to introduce

**Card unification (highest leverage):** `tokens.css` defines `.card` / `.card-elevated` but almost no page uses them. Introduce React wrappers and migrate all surfaces.

- [ ] **`Card`** (`components/Card.js`) — Wraps `.card`; props: `elevated`, `padding`, `onClick`, `footer` slot. Used by Settings, Agents Assembly, connectors, upload panels, System modal tool rows, profile sections
- [ ] **`CardGrid`** — Responsive `grid` + `gap` from tokens; replaces `.modules-container`, `.connector-cards`, parameter grids
- [ ] **`PageLayout`** — Optional app `Header` + title row + toolbar + content (max-width container)
- [ ] **`ModuleCard`** — Extends `Card`; props: `icon`, `title`, `status`, `price`, `onTry`, `onBuy`, `variant`; disables actions when `status !== 'ready'`
- [ ] **`Modal`** — Focus trap, ESC close, `aria-modal`, entrance animation; replace `history-modal` + inline overlays
- [ ] **`StatusIndicator`** — Icon + tooltip (replaces text pills and "Connected" / "Configured" strings where possible)
- [ ] **`FilterBar`** / **`TabList`** — Remove inline styles from `AgentsAssembly.js`

**Status icon spec (draft):**

| Status | Icon | Color token | Accessible name |
|--------|------|-------------|-----------------|
| Ready | ✓ / check-circle | `--color-success` | "Available now" |
| In progress | ◷ / clock | `--color-warning` | "In development" |
| Unavailable | ⊘ / lock | `--color-text-subtle` | "Not available" |

---

### Per-page UX backlog

#### Agents Assembly (`/agents`, `/agents-assembly`) — **P0**

**Files:** `AgentsAssembly.js`, `AgentsAssembly.css`

- [ ] **Reduce vertical whitespace** — Tighten gaps between Header → h2 → Agent Stage toggle → dropdowns → tabs → card grid (target: grid visible without scroll on laptop)
- [ ] **Unify module cards** — Extract `ModuleCard`; fixed header row (40×40 icon, title clamp, status icon top-right); footer row (Try secondary, Buy primary) always bottom-aligned via `min-height` + `margin-top: auto`
- [ ] **Replace text status badges** — Remove "Ready" / "In Progress" pills; use `StatusIndicator` with tooltip
- [ ] **Align business vs technical cards** — Same structure; only accent border/icon tint differs (not different badge positions or button styles)
- [ ] **Toolbar layout** — Agent Stage toggle + Industry + Process on one row; move tabs directly under toolbar (remove extra `marginBottom: 32px` inline styles)
- [ ] **Grid density** — Increase columns on wide screens (`minmax(200px, 1fr)` or 5–6 columns at 1440px); reduce card internal padding to `--space-3`
- [ ] **Remove inline styles** — Module tabs, toggle, chatbot section → CSS classes + tokens
- [ ] **Chatbot panel** — When "Agent Stage" active, use side drawer or collapsible panel instead of pushing entire grid down (preserve catalog above the fold)
- [ ] **Recommended modules row** — Same `ModuleCard` as main grid; "Recommended" as small label chip, not different card chrome
- [ ] **Detailed Report popup** — Migrate to shared `Modal`; report sections as `Card` grid; remove corporate-style one-offs / hardcoded blues in popup
- [ ] **Process Map popup** — Remove dead modal code OR expose via finished Process feature; no orphan `showProcessMap` UI
- [ ] **Buy action** — Replace `alert()` checkout stub with toast or real flow; align with visibility rule (hide Buy until real)

#### Cross-cutting (planning — not yet in page sections)

- [ ] **Protected routes** — Unauthenticated → `/login`; preserve OAuth return URL
- [ ] **Toast single pattern** — Standardize on `useToast` + `<Toast />` OR `showToast`; use for Buy “coming soon” message
- [ ] **`useDataMode()` hook** — `live` \| `demo`; Header toggle (always on) + Settings Preferences; default `live`; persist `localStorage`
- [ ] **Orphan cleanup** — `ExecutiveAssistantAgent.js`, `AvatarAgent.css`: delete or register route
- [ ] **Wire built primitives** — `FormField`, `EmptyState`, `SkeletonLoader` on Settings hub, Campaign, Assembly

#### Header (global)

**Files:** `Header.js`, `Header.css`, `Header_brand.css`

- [ ] **Compact header height** — Reduce padding so main content gains ~24–32px vertical space
- [ ] **Icon-only actions with labels** — `aria-label` + tooltip for System, User (not text label under icon on desktop)
- [ ] **Keep hidden until shipped** — Landscape, Process, Connection (already commented out)
- [ ] **Remove dead code paths** — `handleCreateConnection`, imperative history modal, unused connection modal JSX
- [ ] **User menu** — Profile → `/settings?tab=account`; Settings → `/settings`; Sign out (unchanged)

#### System Overview modal (Header → "system") — **P0**

**Files:** `Header.js`, `Header.css`

- [ ] **Use shared `Modal` + `TabList`** — Not `history-modal` reuse; consistent z-index, overlay token, close button
- [ ] **Tab 1 — Tools** — Render tools as `Card` grid (icon, name, category chip) or compact table with token colors; remove `#FFFFFF` / `#f8fafc` hardcodes in tab content
- [ ] **Tab 2 — Business context** — Move to Settings or Profile (persistent context); wizard-in-modal is hard to discover
- [ ] **Tab 3 — Recommendations** — Replace raw JSON `<pre>` with formatted `Card` list (agent name, reason, CTA to open module); handle API errors with `EmptyState`
- [ ] **Loading / empty** — Skeleton rows while `get_tools_landscape` / `recommend_agents` fetch; not blank "No tools found"
- [ ] **Accessibility** — Focus trap, ESC, `aria-labelledby` on title, return focus to System icon on close

#### Settings hub (`/settings`) — **P0** (merged per product decision #2)

**Files:** `Settings.js`, `Settings.css`; remove duplicate entry points; optional `GET /api/user/me`

**Tabs / sections (one page, one Header):**

| Tab | Absorbs | Content |
|-----|---------|---------|
| **Account** | Ex-Profile dropdown | Email, name, avatar, Google connect/disconnect, sign out |
| **AI & API** | Current AI settings | Keys, models, test connection |
| **Connectors** | Current connectors + ex-Header Connection | OAuth, API keys, cards grid |
| **Business context** | Ex-System modal tab 2 | Industry, role, product/service — persisted to API/ContextStore |
| **Tools landscape** | Ex-Landscape header (optional) | Import browser tools / view scanned tools |
| **Preferences** | Demo toggle (decision #3) | **Live / Demo** (duplicate control OK — same state as Header); default **Live**; other user prefs |

- [ ] **Use global `Header`** — Paper background; no inverted blue banner
- [ ] **Migrate all sections to `Card`** + `CardGrid`; `StatusIndicator` for connected state
- [ ] **Header dropdown** — “Profile” → `/settings?tab=account`; “Settings” → `/settings`
- [ ] **Remove** System modal business tab duplication once migrated (System modal → tools + recommendations only, or fold into Settings entirely)
- [ ] **Loading / validation** — SkeletonLoader; inline errors; toast on save

#### Login & Register (`/login`, `/register`)

**Files:** `Login.js`, `Login.css`, `RegisterUser.js`

- [ ] **Google OAuth on both** — Register gets same `handleGoogleLogin` / callback as Login (decision #1)
- [ ] **Centered card** — Same tokens, `.btn-primary`, OR divider + Google button on register
- [ ] **Tablet breakpoint** — 768px readable

#### Agent pages (shared shell)

**Routes:** `/requirements`, `/campaign-dashboard`, `/datainsights`, `/aichatbot`, `/community-network`, `/sales-helper`, `/content-marketing`, `/event-networking-agent`, `/invest-agent`, `/supply-chain-agent`

**Files:** `agent-shell.css`, each `*Agent.js` + `*.css`

- [ ] **Consistent shell** — Header + `BackButton` + page title + two-column layout (controls left, work area right); same padding as `PageLayout`
- [ ] **Upload cards** — Match `ModuleCard` elevation, border, hover; status on uploads = icon (processing / done / error)
- [ ] **Chat panels** — Fixed min/max height; don’t use `100vh` minus magic numbers; message list scrolls inside panel
- [ ] **Primary actions** — One obvious CTA per panel (Generate, Send, Analyze); secondary actions ghost/outline
- [ ] **Per-agent audit** (apply shell rules + page-specific notes):

| Agent | Renders? | Page-specific UX notes |
|-------|----------|------------------------|
| **Requirements Gathering** | ❌ **Broken** | **Restore component** (file is 0 bytes); was largest agent — CSS still exists. Until fixed: **remove from Agents Assembly Try routes** |
| Campaign Dashboard | ✅ | KPI/status as icons + numbers; replace inline table styles; skeleton table; fix Leads tab → `/requirements` when restored |
| Data Insights | ✅ | `upload-card` → shared `Card`; chart fills panel; skeleton on generate |
| AI Chatbot | ✅ | Composer pinned bottom; message list tokens only |
| Community Network | ✅ | Replace inline HTML profile blocks with `Card`; favorites as card grid |
| Sales Helper | ✅ | `sales-profile-card` → `Card`; lead temperature = `StatusIndicator` |
| Content Marketing | ✅ | Campaign wizard steps in `Card` stack; output preview `Card` + footer actions |
| Event Networking | ⚠️ | **Add `Header`**; topic buttons → `Card` grid; align with app shell |
| Invest Agent | ⚠️ Stub | **Live:** locked card; **Demo:** Try enabled + demo badge |
| Supply Chain | ⚠️ Stub | **Live:** locked card; **Demo:** Try enabled + demo badge |
| Executive Assistant | ✅ | Token migration; stakeholder rows as `Card`; see P3 #1 |

#### Agents Assembly — catalog integrity (visibility)

**Files:** `AgentsAssembly.js`, `agentRegistry.js`

- [ ] **Live / Demo toggle** — **Header, always visible, all users**; default Live; sync with Settings → Preferences (decision #3)
- [ ] **Route guard (Live mode)** — Try only if route exists + `ready`; tooltip if blocked
- [ ] **Stub modules (Live mode)** — **Locked cards** (visible, no Try): Invest, Supply Chain, travel-agent, Requirements until shipped (decision #4)
- [ ] **Demo mode** — Show stubs with `StatusIndicator` or badge “Demo”; dummy process map OK when labeled
- [ ] **Registry-driven status** — `fetchAgents()` + mode toggle; reduce hardcoded lists
- [ ] **Buy button** — Keep visible; toast/modal: checkout coming soon (decision #5); remove `alert()`

#### Executive Assistant (`/executive-assistant`)

**Files:** `ExecutiveAssistantPage.js`, `ExecutiveAssistantPage.css`

- [ ] **Full token migration** (P3 #1) — Remove purple/green one-off palette; match brand
- [ ] **Task/reminder list** — Row layout: icon status, title, time, actions; no oversized cards
- [ ] **WhatsApp / integration blocks** — Connection status via `StatusIndicator`

---

### UX rework priority (updated after full audit)

| Priority | Scope | Outcome |
|----------|--------|---------|
| **P0 — Blockers** | Restore `/requirements`; Settings hub (Account + merge); Google on register; Live/Demo toggle; shared `Card` + `Modal` | One settings surface; honest data modes |
| **P0 — Catalog** | Assembly density + `ModuleCard` + `StatusIndicator`; Live mode hides stubs; Buy → informative toast | Fixes screenshot; user-controlled demo |
| **P1** | `PageLayout` on all authenticated pages; Event Networking gets Header; agent-shell uses `Card` | One product chrome everywhere |
| **P2** | Login/Register visual alignment; Executive Assistant tokens; Campaign/DataInsights skeletons | Auth + heavy agents polished |
| **P3** | Domain-specific charts/tables | Incremental agent UX |

*Depends on:* Design tokens (done), Toast/Skeleton/EmptyState (exist — wire everywhere), **`Card` React component (not started)**.

---

### Planning phase — coverage checklist

**Scope of this audit:** All `App.js` routes, Header + dropdown + System modal, Settings, Agents Assembly (including catalog behavior). Cross-cutting UX lives in **UX Redefinition**; token/a11y/validation detail remains in **P3 — Enterprise UI/UX Overhaul** below (intentional split).

| Area | In plan? | Where |
|------|----------|--------|
| Every routed page (15 routes + `/` redirect) | ✅ | Full application audit table |
| Missing `/profile` + Google login requirement | ✅ | Profile P0 + visibility rule |
| Settings theme mismatch (no Header, blue banner) | ✅ | Settings P0 |
| System Overview modal | ✅ | System modal P0 |
| Shared `Card` / `Modal` / `StatusIndicator` | ✅ | Shared components |
| Catalog honesty (stubs, travel-agent, Buy/Try) | ✅ | Catalog integrity + visibility |
| Agent-specific notes (10 agents) | ✅ | Per-agent table |
| Header dead features (Landscape, Process, Connection) | ✅ | Header + Header-only surfaces |
| Executive Assistant page | ✅ | Executive Assistant section + P3 #1 |
| Login / Register | ✅ | Login & Register (light depth) |
| Accessibility, forms, responsive, loading | ⚠️ Partial | P3 #2–4, #7–8 (not repeated per page) |
| **Agents Assembly — Detailed Report popup** | ❌ Add | Large `detailed-report-modal`; corporate report styling; inline hardcoded colors — treat as `Modal` + `Card` migration |
| **Agents Assembly — Process Map popup** | ❌ Add | `modal-overlay` in page; Process header action hidden but code remains — remove or finish |
| **Agents Assembly — embedded BI chat** | ⚠️ Partial | Agent Stage toggle / chatbot panel (drawer plan only) |
| **Auth: no route guards** | ❌ Add | `/agents`, `/settings`, etc. reachable without login; planning should define protected routes + redirect |
| **Register + Google OAuth** | ✅ | Decision #1 — both login and register |
| **Settings merge (Profile, context, connectors)** | ✅ | Decision #2 — single `/settings` hub |
| **Live / Demo data toggle** | ✅ | Decision #3 — global preference |
| **Stub catalog (Live vs Demo)** | ✅ | Decision #4 |
| **Buy → informative message** | ✅ | Decision #5 |
| **Orphan code** | ❌ Add | `ExecutiveAssistantAgent.js` (not in `App.js`); `AvatarAgent.css` (no component); delete or wire |
| **Orphan CSS** | ⚠️ Partial | `RequirementsGathering.css` (~2k lines) while JS empty — restore page or archive CSS |
| **Document Intelligence UI** | ❌ Add | Backend agent exists; no frontend route — out of scope until product adds catalog entry |
| **Dual notification systems** | ❌ Add | `core/toast.js` (imperative) vs `components/Toast.js` + `useToast` — pick one pattern in plan |
| **Adopt existing components** | ⚠️ Partial | `BackButton`, `EmptyState`, `SkeletonLoader`, `FormField` built but not mandated per page |
| **Buy flow still uses `alert()`** | ❌ Add | `AgentsAssembly.js` `handleBuyModule` — noted in P3 #2 but not UX audit table |
| **Mobile / tablet** | ⚠️ Partial | P3 #8 global breakpoints; not per Settings modal / Assembly grid |
| **i18n / dark mode** | ➖ Out of scope | Not in current product plan |
| **sprint-plan-p3.md sync** | ⚠️ | Align sprint doc with UX Redefinition priorities when implementation starts |

### Hidden UI inventory — what exists & what data can fill it

| Hidden / dormant UI | Location | Still in code? | Real data available today? | Recommendation |
|---------------------|----------|----------------|----------------------------|----------------|
| **Connection** icon + modal | `Header.js` (commented) | Handlers remain: bulk PDF → `POST /upload`, DB fields, `handleTestConnection` | **Partial** — `/upload` works; DB test POST does not match backend (`GET /test-connection` is health-only). Connectors live in **Settings** | **Do not re-expose** — keep in Settings; delete dead Header handlers |
| **Landscape** (Application Landscape) | `Header.js` (commented) | Full imperative modal + `handleLandscapeClick` | **Yes, if Chrome closed** — `GET /chrome_history?user_id=` → saves `user_data/tools_landscape/tools_landscape.json` → feeds System tab 1 via `GET /get_tools_landscape` | Re-enable only as Settings action “Scan browser tools” with clear Chrome-close instructions; or seed JSON server-side for demos |
| **Process** icon | `Header.js` (commented) | `onProcessClick` still passed from `AgentsAssembly` | **No real API** — `generateProcessMapData()` is **dummy** steps; can use `selectedIndustry`, `chatState`, `enterprise_chat` answers if persisted | Wire to `enterprise_chat` / `ContextStore` output before showing; or keep hidden |
| **Process Map modal** | `AgentsAssembly.js` | Renders when `showProcessMap` true (no header button) | Same as above — dummy JSON | Same as Process icon |
| **System modal tab 1** (Tools) | `Header.js` (visible via **system**) | Active | **Only if** `tools_landscape.json` exists (Landscape scan or manual file). Repo has **no** default file → often “No tools found” | Populate via Landscape scan, manual import, or connector-derived tool list |
| **System modal tab 2** (Business context) | `Header.js` | Active — wizard in modal | **Local React state only** — not saved on Confirm | Persist to `ContextStore` / Profile / Settings; or merge with Agents Assembly `enterprise_chat` `chat_state` |
| **System modal tab 3** (Recommendations) | `Header.js` | Active — raw JSON `<pre>` | **Yes** — `POST /recommend_agents` (needs OpenAI). **Bug:** Header sends `tools_landscape`, `business_description`; API expects `tools`, `product_service` | Fix payload + render structured cards (same shape as Detailed Report popup) |
| **Profile** menu item | `Header.js` dropdown | Active — toast only | **Yes** — account data in **Settings → Account** (decision #2) | Link dropdown to `/settings?tab=account` |
| **Connection setup** toast | `handleCreateConnection` | Dead path | None | Remove |

**Visible slots that could show more data (not hidden):**

| UI | Data to use |
|----|-------------|
| Agents Assembly **Agent Stage** chat | Already uses `POST /enterprise_chat` — feeds recommendations + Detailed Report when chat completes |
| Agents Assembly **Detailed Report** popup | Uses `enterprise_chat` / search response — real when flow completes |
| **Settings → Connectors** | `GET /api/settings`, `POST /api/settings/test-connection`, OAuth `auth-url` — real |
| **Data Insights** left panel | Duplicates old Connection pattern (file upload, DB/API select) — agent-local, not header |

**Verdict:** UX/UI **planning phase closed** — product decisions locked above. Remaining ❌ rows in this table are **implementation backlog** (auth guards, modal migrations, sprint-plan sync), not open product questions.

---

## P3 — Enterprise UI/UX Overhaul

**Current Score: 43/100 — NOT ENTERPRISE READY**

| Category | Score | Status |
|----------|-------|--------|
| Design Token Consistency | 45/100 | POOR |
| Accessibility | 35/100 | POOR |
| Loading/Empty States | 40/100 | POOR |
| Form Validation UI | 25/100 | CRITICAL |
| Modal Quality | 50/100 | FAIR |
| Responsive Design | 60/100 | FAIR |
| Professional Polish | 40/100 | POOR |

---

### CRITICAL — Space Optimization Patterns (Apply Everywhere)

**Before implementing any form/settings UI, check:**

| Pattern | Apply To | Implementation |
|---------|----------|----------------|
| **2-column form grids** | Settings, Profile, forms with 4+ fields | `grid-template-columns: repeat(2, 1fr)` on desktop; 1fr on mobile |
| **Inline label+input** | Short fields (name, date) | Flex row with label as fixed width, input flexible |
| **Compact section headers** | All page headers | Reduce padding (`--space-3` not `--space-6`); icon + text same line |
| **Equal-width action buttons** | Save/Cancel, Edit/Delete pairs | `min-width: 140px; max-width: 200px; flex: 0 0 auto` |
| **Avoid full-width buttons** | Forms | Only for primary CTA on mobile |
| **Reduce vertical gaps** | Between form sections | `--space-4` between groups, not `--space-6` or `--space-8` |
| **Full-width only when needed** | Textareas, file uploads | Add `.full-width` class explicitly |
| **Compact cards** | Module grids, connector cards | `--space-3` internal padding, not `--space-5` |
| **Horizontal filters** | Filter dropdowns, toggles | Same row as tabs, right-aligned |

---

### CRITICAL — This Sprint

#### 1. ExecutiveAssistantPage.css — Complete Rewrite
- [ ] Replace 80+ hardcoded colors with design tokens
- [ ] Fix: #667eea, #764ba2, #48bb78, #25d366, etc. → tokens
- [ ] Update button variants (.btn-save, .btn-cancel, .btn-view, etc.)
- [ ] Fix gradient backgrounds to use token colors

#### 2. Form Validation UI — Create System
- [ ] Add error state styling for inputs (red border, error message)
- [ ] Add success state styling (green check, confirmation)
- [ ] Mark required fields visually (asterisk or label)
- [ ] Create inline validation component
- [ ] Replace all `alert()` calls with toast notifications

#### 3. Accessibility — ARIA & Focus
- [ ] Add aria-label to all icon buttons (Header icons, modal close)
- [ ] Add aria-live regions for dynamic content (chat, status updates)
- [ ] Implement focus trap in modals (Tab cycles within modal)
- [ ] Add keyboard shortcuts (ESC to close modal)
- [ ] Create visible focus ring token/class

#### 4. Loading States — Visual Feedback
- [ ] Create skeleton loader component
- [ ] Replace "Loading..." text with skeleton screens
- [ ] Add spinner/progress for file uploads
- [ ] Show "Saved!" confirmation on settings changes
- [ ] Add sent/delivered state for messages

---

### HIGH — Next Sprint

#### 5. Header Cleanup
- [ ] Hide Landscape icon (feature requires Chrome history API — broken)
- [ ] Hide Process icon (feature incomplete)
- [ ] Remove Connection modal (consolidate into Settings)
- [ ] Fix modal-overlay color (hardcoded rgba → token)
- [ ] Add hover/active states for header icons

#### 6. AgentsAssembly.css — Token Consistency
*See also: **UX Redefinition → Agents Assembly** for layout, cards, and status icons.*
- [ ] Line 41: Remove hardcoded #A84D08 from gradient
- [ ] Lines 52-73: Replace hardcoded yellow (#F59E0B) badges with tokens
- [ ] Line 117: .corporate-title color → var(--color-primary)
- [ ] Unify Business/Technical module card colors (covered by `ModuleCard` in UX Redefinition)
- [ ] Replace text status badges with `StatusIndicator` icons (UX Redefinition P0)

#### 7. Modal Polish
- [ ] Add entrance animation (scale + fade)
- [ ] Add aria-label to close buttons
- [ ] Fix System Overview modal styling
- [ ] Update Connection modal inputs (consistent padding)
- [ ] Add keyboard trap management

#### 8. Responsive Design
- [ ] Add 480px breakpoint (small phones)
- [ ] Add 768px tablet breakpoint to Login.css
- [ ] Add landscape orientation handling
- [ ] Test all agents on mobile viewport

---

### MEDIUM — Future Sprint

#### 9. Button/Input Standardization
- [ ] Document all button variants (.btn-primary, .btn-secondary, etc.)
- [ ] Consolidate duplicate button styles across pages
- [ ] Standardize all inputs to use .input class
- [ ] Create .input-error, .input-success variants

#### 10. Professional Polish
- [ ] Add micro-interactions (button scale on click)
- [ ] Add copy-to-clipboard visual feedback
- [ ] Create toast notification component
- [ ] Add empty state illustrations
- [ ] Standardize spacing (remove hardcoded px values)

#### 11. Color Contrast (WCAG AA)
- [ ] Audit text colors on light backgrounds
- [ ] Fix #718096 text on #f7fafc background
- [ ] Ensure 4.5:1 minimum contrast ratio

#### 12. Settings Page
- [ ] Unify color theme with main app
- [ ] Move Database/File/API connectors from Header
- [ ] Better connected vs not-connected visual
- [ ] Add form validation feedback

---

### Files Priority List

**CRITICAL:**
| File | Issues |
|------|--------|
| `ExecutiveAssistantPage.css` | 80+ hardcoded colors, worst offender |
| `Login.js` | Missing tablet responsive breakpoint |
| `Settings.js` | No form validation feedback |

**HIGH:**
| File | Issues |
|------|--------|
| `AgentsAssembly.css` | Mixed token/hardcode colors |
| `Header.css` | Modal accessibility, hardcoded overlay |
| `Header.js` | Icon accessibility, broken features |
| `RequirementsGathering.js` | Empty states, uses alert() |

**POSITIVE (Keep):**
- ✓ tokens.css — Strong foundation
- ✓ agent-shell.css — Good reusable patterns
- ✓ SalesHelperAgent.css — Follows token system
- ✓ Login.css — Good responsive example

---

## Workflow Templates ✅ IMPLEMENTED

**Status:** Done — Linear state machine working. LangGraph migration optional for advanced features.

### Backend Tasks
- [x] Create `backend/models/workflow.py` — WorkflowTemplate, WorkflowStage, WorkflowInstance
- [x] Create `backend/routes/workflows.py` — CRUD + state transitions
- [x] Create `backend/config/workflow-templates/` — JSON template definitions
- [x] Add workflow state machine logic

### Frontend Tasks
- [x] Create `frontend/src/workflows/WorkflowsPage.js` — Templates listing
- [x] Create `frontend/src/workflows/WorkflowRunner.js` — Active workflow UI
- [x] Create `frontend/src/workflows/WorkflowProgress.js` — Progress tracker
- [ ] Add Workflows section to landing page (navigation exists via /workflows route)

---

## Agent Dependency System ✅ IMPLEMENTED

**Status:** Done — Validator with warn/strict modes, API endpoints, frontend gate.

### Backend Tasks
- [x] Create `backend/config/agent-dependencies.json` — dependency config
- [x] Create `backend/core/dependency_validator.py` — validation middleware
- [x] Create `backend/routes/dependencies.py` — API endpoints
- [ ] Add `user_profile` provider agent (gap: 3 agents need it, settings fallback exists)

### Frontend Tasks
- [x] Create `frontend/src/components/AgentPrerequisiteGate.js` — UI gate
- [x] Show user what's needed before using an agent

---

## CI/CD + QA Automation

### QA Test Automation (from QA_Test_Checklist.xlsx audit)

| Category | Total | Automatable | Notes |
|----------|-------|-------------|-------|
| E2E tests | 52 | ~45 (87%) | Playwright/Cypress |
| Partially automatable | - | ~5 (10%) | Drag/drop, AI responses, async |
| Manual preferred | - | ~2 (3%) | External channel, AI quality |

### CI/CD Tasks ✅ IMPLEMENTED
- [x] Create `.github/workflows/ci.yml` — GitHub Actions pipeline
- [x] Add Playwright to `frontend/package.json`
- [x] Create `frontend/e2e/` — test directory
- [x] Create `frontend/playwright.config.js`
- [x] Add pytest job for backend
- [x] Add ESLint + Prettier job

### Priority E2E Tests ✅ IMPLEMENTED
- [x] Login/logout flow — `e2e/auth.spec.js`
- [x] Project creation — `e2e/projects.spec.js`
- [x] Agent navigation — `e2e/navigation.spec.js`
- [x] Settings save/load — `e2e/settings.spec.js`
- [ ] Document upload (needs backend test fixtures)
- [ ] Demo mode toggle (needs frontend implementation)

---

## Backend Critical Gaps (July 2026 Audit) — STATUS UPDATE

| Issue | Status | Notes |
|-------|--------|-------|
| **Executive Assistant no backend** | ✅ DONE | Full CRUD for tasks, reminders, stakeholders |
| **Projects in-memory** | ✅ DONE | Uses SQLAlchemy models in `core/models.py` |
| **Teams in-memory** | ✅ DONE | Uses SQLAlchemy models in `core/models.py` |
| **No user_profile provider** | ⚠️ TODO | Settings fallback exists in dependency config |
| **Dependency enforcement** | ✅ DONE | Validator with warn/strict modes + API |

### Executive Assistant Backend ✅ IMPLEMENTED
- [x] Create `backend/agents/executive_assistant/routes.py`
- [x] Create `backend/agents/executive_assistant/service.py`
- [x] Create `backend/agents/executive_assistant/models.py`
- [x] Add task management endpoints (CRUD)
- [x] Add reminder endpoints (CRUD, linked to tasks)
- [x] Add stakeholder endpoints (CRUD)
- [ ] Integrate with calendar/email connectors (future)

### Projects Persistence
- [ ] Create SQLAlchemy Project model
- [ ] Migrate `backend/routes/projects.py` from dict to DB
- [ ] Add project-agent association table

### Teams Persistence
- [ ] Create SQLAlchemy Team, TeamMember models
- [ ] Migrate `backend/routes/team.py` from dict to DB
- [ ] Add invitation system with email

---

## Frontend Critical Gaps (July 2026 Audit) — STATUS UPDATE

| Issue | Status | Notes |
|-------|--------|-------|
| **RequirementsGathering.js empty** | ✅ FIXED | Restored from git (2381 lines) |
| **Event Networking no Header** | ✅ DONE | Already has Header (line 1238) |
| **Orphan files removed** | ✅ DONE | Deleted ExecutiveAssistantAgent.js, AvatarAgent.css |

### RequirementsGathering Restore ✅ DONE
- [x] Check git history: `git log --all -- frontend/src/agents/RequirementsGathering.js`
- [x] Restore last working version (commit 51846a4b)
- [x] File restored with 2381 lines + existing CSS

---

## Backend P2 Tasks

- [ ] Image OCR for documents
- [ ] LLM entity extraction
- [ ] Connector health monitoring
- [ ] SSE notifications for processing status
- [ ] HubSpot connector

## Backend P3 Tasks

- [ ] More connectors (Salesforce, Twitter/X)
- [ ] Knowledge graph visualization
- [ ] Data lineage tracking
- [ ] Domain packs for industry entities

---

## API Reference

### Documents
| Method | Endpoint |
|--------|----------|
| POST | `/api/document-intelligence/upload` |
| GET | `/api/document-intelligence/documents` |
| POST | `/api/document-intelligence/chat` |

### Connectors
| Method | Endpoint |
|--------|----------|
| GET | `/api/connectors` |
| POST | `/api/connectors/:id/fetch` |
| GET | `/api/connectors/:id/auth-url` |

### Settings
| Method | Endpoint |
|--------|----------|
| GET | `/api/settings` |
| POST | `/api/settings` |
| DELETE | `/api/settings/:category/:key` |
| POST | `/api/settings/test-connection` |
