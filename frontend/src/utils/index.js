/**
 * Utils Library Index
 *
 * Usage:
 *   import { announce, generateId, getFocusableElements } from '../utils';
 */

export {
  // Mode storage (demo/live data segregation)
  isDemoMode,
  getAgentData,
  setAgentData,
  updateAgentData,
  clearAgentData,
  useModeStorage,
  AGENT_KEYS,
} from './modeStorage';

export {
  // ID generation
  generateId,
  resetIdCounter,

  // Screen reader announcements
  announce,
  announceAssertive,

  // Focus management
  getFocusableElements,
  getFirstFocusable,
  getLastFocusable,
  focusFirst,

  // ARIA helpers
  getExpandableButtonProps,
  getExpandableContentProps,
  getTabProps,
  getTabPanelProps,
  getDialogProps,
  getOptionProps,

  // Color contrast
  getContrastRatio,
  meetsContrastAA,
  meetsContrastAAA,

  // Motion preferences
  prefersReducedMotion,
  getAnimationDuration
} from './accessibility';

export {
  // Demo mode API helpers
  isDemoMode as isDemoModeApi,
  demoFetch,
  demoAiFetch,
  showDemoWarning,
} from './demoApi';
