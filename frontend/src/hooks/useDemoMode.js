import { useCallback } from 'react';
import { useMode } from '../contexts';

/**
 * Reactive Live/Demo mode — synced via ModeContext.
 * No polling needed - uses React Context with event-based updates.
 */
export function useDemoMode() {
  const { isDemoMode, setMode } = useMode();

  const switchToDemo = useCallback(() => {
    setMode(true);
  }, [setMode]);

  const switchToLive = useCallback(() => {
    setMode(false);
  }, [setMode]);

  return { isDemoMode, isLiveMode: !isDemoMode, switchToDemo, switchToLive };
}

export default useDemoMode;
