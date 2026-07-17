/**
 * Accessibility Utilities
 *
 * Helper functions for building accessible UI components.
 */

// ═══════════════════════════════════════════════════════════════════════════════
// ID Generation
// ═══════════════════════════════════════════════════════════════════════════════

let idCounter = 0;

/**
 * Generate a unique ID for ARIA attributes
 * @param {string} prefix - Optional prefix for the ID
 * @returns {string} Unique ID
 */
export function generateId(prefix = 'id') {
  return `${prefix}-${++idCounter}`;
}

/**
 * Reset ID counter (useful for testing)
 */
export function resetIdCounter() {
  idCounter = 0;
}

// ═══════════════════════════════════════════════════════════════════════════════
// Screen Reader Announcements
// ═══════════════════════════════════════════════════════════════════════════════

let liveRegion = null;

/**
 * Get or create the live region element for screen reader announcements
 */
function getLiveRegion() {
  if (liveRegion) return liveRegion;

  liveRegion = document.createElement('div');
  liveRegion.setAttribute('role', 'status');
  liveRegion.setAttribute('aria-live', 'polite');
  liveRegion.setAttribute('aria-atomic', 'true');
  liveRegion.className = 'sr-only';
  liveRegion.style.cssText = `
    position: absolute;
    width: 1px;
    height: 1px;
    padding: 0;
    margin: -1px;
    overflow: hidden;
    clip: rect(0, 0, 0, 0);
    white-space: nowrap;
    border: 0;
  `;
  document.body.appendChild(liveRegion);

  return liveRegion;
}

/**
 * Announce a message to screen readers
 * @param {string} message - Message to announce
 * @param {string} priority - 'polite' or 'assertive'
 */
export function announce(message, priority = 'polite') {
  const region = getLiveRegion();
  region.setAttribute('aria-live', priority);

  // Clear and set message (forces re-announcement)
  region.textContent = '';
  requestAnimationFrame(() => {
    region.textContent = message;
  });
}

/**
 * Announce assertively (interrupts current speech)
 */
export function announceAssertive(message) {
  announce(message, 'assertive');
}

// ═══════════════════════════════════════════════════════════════════════════════
// Focus Management
// ═══════════════════════════════════════════════════════════════════════════════

const FOCUSABLE_SELECTORS = [
  'button:not([disabled])',
  'input:not([disabled])',
  'select:not([disabled])',
  'textarea:not([disabled])',
  'a[href]',
  '[tabindex]:not([tabindex="-1"])',
  '[contenteditable="true"]'
].join(', ');

/**
 * Get all focusable elements within a container
 * @param {HTMLElement} container - Container element
 * @returns {HTMLElement[]} Array of focusable elements
 */
export function getFocusableElements(container) {
  if (!container) return [];
  return Array.from(container.querySelectorAll(FOCUSABLE_SELECTORS))
    .filter((el) => el.offsetParent !== null); // Filter hidden elements
}

/**
 * Get the first focusable element in a container
 */
export function getFirstFocusable(container) {
  const elements = getFocusableElements(container);
  return elements[0] || null;
}

/**
 * Get the last focusable element in a container
 */
export function getLastFocusable(container) {
  const elements = getFocusableElements(container);
  return elements[elements.length - 1] || null;
}

/**
 * Focus the first focusable element in a container
 */
export function focusFirst(container) {
  const first = getFirstFocusable(container);
  if (first) {
    first.focus();
    return true;
  }
  return false;
}

// ═══════════════════════════════════════════════════════════════════════════════
// ARIA Helpers
// ═══════════════════════════════════════════════════════════════════════════════

/**
 * Create ARIA props for a button that controls expandable content
 */
export function getExpandableButtonProps(id, isExpanded) {
  return {
    'aria-expanded': isExpanded,
    'aria-controls': id
  };
}

/**
 * Create ARIA props for expandable content
 */
export function getExpandableContentProps(id, isExpanded) {
  return {
    id,
    'aria-hidden': !isExpanded
  };
}

/**
 * Create ARIA props for a tab
 */
export function getTabProps(id, panelId, isSelected) {
  return {
    id,
    role: 'tab',
    'aria-selected': isSelected,
    'aria-controls': panelId,
    tabIndex: isSelected ? 0 : -1
  };
}

/**
 * Create ARIA props for a tab panel
 */
export function getTabPanelProps(id, tabId, isSelected) {
  return {
    id,
    role: 'tabpanel',
    'aria-labelledby': tabId,
    hidden: !isSelected,
    tabIndex: 0
  };
}

/**
 * Create ARIA props for a modal dialog
 */
export function getDialogProps(labelId, descriptionId) {
  return {
    role: 'dialog',
    'aria-modal': true,
    'aria-labelledby': labelId,
    'aria-describedby': descriptionId
  };
}

/**
 * Create ARIA props for a listbox option
 */
export function getOptionProps(id, isSelected, isDisabled = false) {
  return {
    id,
    role: 'option',
    'aria-selected': isSelected,
    'aria-disabled': isDisabled
  };
}

// ═══════════════════════════════════════════════════════════════════════════════
// Color Contrast
// ═══════════════════════════════════════════════════════════════════════════════

/**
 * Calculate relative luminance of a color
 * @param {number} r - Red (0-255)
 * @param {number} g - Green (0-255)
 * @param {number} b - Blue (0-255)
 */
function getLuminance(r, g, b) {
  const [rs, gs, bs] = [r, g, b].map((c) => {
    c = c / 255;
    return c <= 0.03928 ? c / 12.92 : Math.pow((c + 0.055) / 1.055, 2.4);
  });
  return 0.2126 * rs + 0.7152 * gs + 0.0722 * bs;
}

/**
 * Parse hex color to RGB
 */
function hexToRgb(hex) {
  const result = /^#?([a-f\d]{2})([a-f\d]{2})([a-f\d]{2})$/i.exec(hex);
  return result ? {
    r: parseInt(result[1], 16),
    g: parseInt(result[2], 16),
    b: parseInt(result[3], 16)
  } : null;
}

/**
 * Calculate contrast ratio between two colors
 * @param {string} color1 - Hex color
 * @param {string} color2 - Hex color
 * @returns {number} Contrast ratio (1-21)
 */
export function getContrastRatio(color1, color2) {
  const rgb1 = hexToRgb(color1);
  const rgb2 = hexToRgb(color2);

  if (!rgb1 || !rgb2) return 0;

  const l1 = getLuminance(rgb1.r, rgb1.g, rgb1.b);
  const l2 = getLuminance(rgb2.r, rgb2.g, rgb2.b);

  const lighter = Math.max(l1, l2);
  const darker = Math.min(l1, l2);

  return (lighter + 0.05) / (darker + 0.05);
}

/**
 * Check if contrast meets WCAG AA for normal text (4.5:1)
 */
export function meetsContrastAA(foreground, background) {
  return getContrastRatio(foreground, background) >= 4.5;
}

/**
 * Check if contrast meets WCAG AAA for normal text (7:1)
 */
export function meetsContrastAAA(foreground, background) {
  return getContrastRatio(foreground, background) >= 7;
}

// ═══════════════════════════════════════════════════════════════════════════════
// Reduced Motion
// ═══════════════════════════════════════════════════════════════════════════════

/**
 * Check if user prefers reduced motion
 */
export function prefersReducedMotion() {
  return window.matchMedia('(prefers-reduced-motion: reduce)').matches;
}

/**
 * Get animation duration based on user preference
 * @param {number} duration - Default duration in ms
 * @returns {number} Duration (0 if reduced motion preferred)
 */
export function getAnimationDuration(duration) {
  return prefersReducedMotion() ? 0 : duration;
}
