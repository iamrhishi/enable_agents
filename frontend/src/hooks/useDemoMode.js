import { useState, useEffect, useCallback } from 'react';

/**
 * Reactive Live/Demo mode — synced with Header toggle via localStorage.
 */
export function useDemoMode() {
  const readMode = () => localStorage.getItem('enableAgentsMode') !== 'live';

  const [isDemoMode, setIsDemoMode] = useState(readMode);

  useEffect(() => {
    const sync = () => setIsDemoMode(readMode());
    window.addEventListener('storage', sync);
    const interval = setInterval(sync, 1000);
    return () => {
      window.removeEventListener('storage', sync);
      clearInterval(interval);
    };
  }, []);

  const switchToDemo = useCallback(() => {
    localStorage.setItem('enableAgentsMode', 'demo');
    setIsDemoMode(true);
    window.dispatchEvent(new Event('storage'));
  }, []);

  const switchToLive = useCallback(() => {
    localStorage.setItem('enableAgentsMode', 'live');
    setIsDemoMode(false);
    window.dispatchEvent(new Event('storage'));
  }, []);

  return { isDemoMode, isLiveMode: !isDemoMode, switchToDemo, switchToLive };
}

export default useDemoMode;
