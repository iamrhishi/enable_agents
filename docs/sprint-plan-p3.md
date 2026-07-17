# P3 Sprint Plan — Enterprise UI/UX Overhaul

**Goal:** Bring Enterprise Readiness from 43/100 → 85/100
**Approach:** Foundation First → Then Apply Everywhere

---

## Sprint 1: Foundation — Token System Complete

**Focus:** Expand tokens.css with everything we need before touching other files

### 1.1 Add Missing Color Tokens
| Token | Value | Purpose |
|-------|-------|---------|
| `--color-focus-ring` | `rgba(194, 65, 12, 0.4)` | Consistent focus outlines |
| `--color-skeleton` | `#E8DDD5` | Skeleton loader background |
| `--color-skeleton-shine` | `#F1EAE4` | Skeleton shimmer highlight |

### 1.2 Add Input State Classes
```css
.input-error { border-color: var(--color-error); }
.input-success { border-color: var(--color-success); }
.input-error-message { color: var(--color-error); font-size: var(--text-sm); }
```

### 1.3 Add Focus Ring Utility
```css
.focus-ring:focus {
  outline: none;
  box-shadow: 0 0 0 3px var(--color-focus-ring);
}
```

### 1.4 Add Animation Tokens
```css
--animation-skeleton: skeleton-shimmer 1.5s infinite;
--animation-fade-in: fade-in 200ms ease;
--animation-scale-in: scale-in 200ms ease;
```

### 1.5 Add Z-Index for Toast
```css
--z-toast: 1100;
```

**Deliverable:** Complete token system ready for all components

---

## Sprint 2: Foundation — Core Components

**Focus:** Build reusable components that will be used everywhere

### 2.1 Toast Notification Component

**Files to create:**
- `src/components/Toast.js`
- `src/components/Toast.css`
- `src/hooks/useToast.js`

**Features:**
- Success, Error, Warning, Info variants
- Auto-dismiss after 3s
- Manual dismiss with X
- Stack multiple toasts
- Slide-in animation

### 2.2 Skeleton Loader Component

**Files to create:**
- `src/components/SkeletonLoader.js`
- `src/components/SkeletonLoader.css`

**Variants:**
- `<Skeleton.Text />` — Single line
- `<Skeleton.Paragraph />` — Multiple lines
- `<Skeleton.Card />` — Card placeholder
- `<Skeleton.Avatar />` — Circle placeholder
- `<Skeleton.Table rows={5} />` — Table rows

### 2.3 Empty State Component

**Files to create:**
- `src/components/EmptyState.js`
- `src/components/EmptyState.css`

**Props:**
- `icon` — Optional icon
- `title` — Main message
- `description` — Secondary text
- `action` — Optional CTA button

### 2.4 Spinner Component

**Files to create:**
- `src/components/Spinner.js`
- `src/components/Spinner.css`

**Variants:**
- `size="sm"` — 16px (inline)
- `size="md"` — 24px (button)
- `size="lg"` — 48px (page)

**Deliverable:** 4 reusable components ready to use

---

## Sprint 3: Foundation — Form System

**Focus:** Build form validation infrastructure

### 3.1 FormField Component

**Files to create:**
- `src/components/FormField.js`
- `src/components/FormField.css`

**Features:**
- Label with required indicator (*)
- Error message display
- Success checkmark
- Help text support

**Usage:**
```jsx
<FormField
  label="Email"
  required
  error="Invalid email format"
>
  <input type="email" className="input input-error" />
</FormField>
```

### 3.2 Validation Hook

**File to create:**
- `src/hooks/useValidation.js`

**Validators:**
- `required(value)` — Not empty
- `email(value)` — Valid email format
- `minLength(value, min)` — Minimum chars
- `maxLength(value, max)` — Maximum chars
- `pattern(value, regex)` — Custom regex

### 3.3 Focus Trap Hook

**File to create:**
- `src/hooks/useFocusTrap.js`

**Features:**
- Trap Tab within container
- Return focus on unmount
- ESC key to close

**Deliverable:** Complete form validation system

---

## Sprint 4: Foundation — Accessibility Utilities

**Focus:** Build accessibility infrastructure

### 4.1 ARIA Utilities

**File to create:**
- `src/utils/accessibility.js`

**Functions:**
- `announceToScreenReader(message)` — Live region announcer
- `generateId()` — Unique IDs for aria-labelledby
- `trapFocus(containerRef)` — Focus management

### 4.2 Keyboard Navigation Hook

**File to create:**
- `src/hooks/useKeyboard.js`

**Features:**
- ESC to close modals
- Arrow keys for lists
- Enter to select

### 4.3 Skip Link Component

**Files to create:**
- `src/components/SkipLink.js`
- `src/components/SkipLink.css`

**Deliverable:** Accessibility utilities ready

---

## Sprint 5: Apply — Quick Wins

**Focus:** Use foundation to fix obvious issues

### 5.1 Header Cleanup
- [ ] Hide Landscape icon (broken)
- [ ] Hide Process icon (incomplete)
- [ ] Remove Connection modal code
- [ ] Add aria-labels to remaining icons
- [ ] Apply focus-ring to icon buttons

### 5.2 Replace All alert() Calls
- [ ] RequirementsGathering.js → useToast
- [ ] SalesHelperAgent.js → useToast
- [ ] Settings.js → useToast

### 5.3 Add Loading States
- [ ] RequirementsGathering.js → SkeletonLoader
- [ ] SalesHelperAgent.js → SkeletonLoader
- [ ] Settings.js → Spinner on save

### 5.4 Add Empty States
- [ ] Saved leads table → EmptyState
- [ ] Chat history → EmptyState
- [ ] Requirements list → EmptyState

**Deliverable:** Visible UX improvements using new components

---

## Sprint 6: Apply — Design System Enforcement

**Focus:** Fix files not using tokens

### 6.1 ExecutiveAssistantPage.css — Full Rewrite
- [ ] Map all colors to tokens
- [ ] Map all spacing to tokens
- [ ] Map all typography to tokens
- [ ] Apply .btn classes
- [ ] Apply .input classes

### 6.2 AgentsAssembly.css — Cleanup
- [ ] Remove hardcoded #A84D08
- [ ] Remove hardcoded #F59E0B
- [ ] Fix .corporate-title color
- [ ] Unify module card colors

### 6.3 Header.css — Modal Polish
- [ ] Fix modal-overlay color
- [ ] Apply focus-ring to close button
- [ ] Add modal animations

### 6.4 Form Validation Application
- [ ] Login.js → FormField + useValidation
- [ ] Settings.js → FormField + useValidation

**Deliverable:** All files using design system

---

## Sprint 7: Apply — Accessibility Pass

**Focus:** Apply accessibility to entire app

### 7.1 ARIA Labels Pass
- [ ] All icon buttons have aria-label
- [ ] All form fields have labels
- [ ] All modals have aria-labelledby

### 7.2 Focus Management Pass
- [ ] All modals trap focus
- [ ] ESC closes all modals
- [ ] Focus returns after modal close

### 7.3 Keyboard Navigation Pass
- [ ] Tab order makes sense
- [ ] All interactive elements reachable
- [ ] Focus visible on all elements

**Deliverable:** WCAG AA compliant

---

## Sprint 8: Apply — Responsive & Polish

**Focus:** Mobile and final polish

### 8.1 Responsive Breakpoints
- [ ] Add 480px to all agent pages
- [ ] Add 768px tablet to Login.css
- [ ] Test on real mobile devices

### 8.2 Micro-interactions
- [ ] Button press animation (scale 0.98)
- [ ] Modal entrance animation
- [ ] Card hover lift with shadow

### 8.3 Color Contrast Audit
- [ ] Run automated contrast check
- [ ] Fix any failures

**Deliverable:** Mobile-ready, polished, enterprise-ready

---

## Files to Create (Foundation)

| File | Sprint | Purpose |
|------|--------|---------|
| `src/components/Toast.js` | 2 | Notification component |
| `src/components/Toast.css` | 2 | Toast styles |
| `src/hooks/useToast.js` | 2 | Toast hook |
| `src/components/SkeletonLoader.js` | 2 | Loading skeleton |
| `src/components/SkeletonLoader.css` | 2 | Skeleton styles |
| `src/components/EmptyState.js` | 2 | Empty state display |
| `src/components/EmptyState.css` | 2 | Empty state styles |
| `src/components/Spinner.js` | 2 | Loading spinner |
| `src/components/Spinner.css` | 2 | Spinner styles |
| `src/components/FormField.js` | 3 | Form field wrapper |
| `src/components/FormField.css` | 3 | Form field styles |
| `src/hooks/useValidation.js` | 3 | Form validation |
| `src/hooks/useFocusTrap.js` | 3 | Modal focus trap |
| `src/utils/accessibility.js` | 4 | A11y utilities |
| `src/hooks/useKeyboard.js` | 4 | Keyboard shortcuts |
| `src/components/SkipLink.js` | 4 | Skip to content |

---

## Timeline Summary

| Sprint | Focus | Type |
|--------|-------|------|
| 1 | Token System Complete | Foundation |
| 2 | Core Components (Toast, Skeleton, etc.) | Foundation |
| 3 | Form System (FormField, Validation) | Foundation |
| 4 | Accessibility Utilities | Foundation |
| 5 | Quick Wins (Header, alerts, loading) | Apply |
| 6 | Design System Enforcement | Apply |
| 7 | Accessibility Pass | Apply |
| 8 | Responsive & Polish | Apply |

**Sprints 1-4:** Build foundation (no user-visible changes)
**Sprints 5-8:** Apply foundation (visible improvements)

---

## Success Metrics

| Metric | Before | After |
|--------|--------|-------|
| Enterprise Readiness | 43/100 | 85/100 |
| Reusable Components | 2 | 10+ |
| Design Token Usage | 45% | 95% |
| Accessibility | 35/100 | 80/100 |
| Form Validation | 25/100 | 90/100 |
