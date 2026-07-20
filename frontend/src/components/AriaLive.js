import React, { useEffect, useState, useCallback } from 'react';
import './AriaLive.css';

/**
 * AriaLive Component
 *
 * Announces dynamic content changes to screen readers.
 *
 * Usage:
 *   <AriaLive message="Form saved successfully" />
 *   <AriaLive message="Loading..." politeness="polite" />
 *   <AriaLive message="Error: Invalid input" politeness="assertive" />
 */
export function AriaLive({
  message,
  politeness = 'polite', // 'polite' | 'assertive' | 'off'
  atomic = true,
  clearAfter = 0, // ms to clear message (0 = don't clear)
}) {
  const [announcement, setAnnouncement] = useState(message);

  useEffect(() => {
    setAnnouncement(message);

    if (clearAfter > 0 && message) {
      const timer = setTimeout(() => setAnnouncement(''), clearAfter);
      return () => clearTimeout(timer);
    }
  }, [message, clearAfter]);

  return (
    <div
      role="status"
      aria-live={politeness}
      aria-atomic={atomic}
      className="visually-hidden aria-live"
    >
      {announcement}
    </div>
  );
}

/**
 * useAnnounce Hook
 *
 * Programmatically announce messages to screen readers.
 *
 * Usage:
 *   const announce = useAnnounce();
 *   announce('Item added to cart');
 *   announce('Error occurred', 'assertive');
 */
export function useAnnounce() {
  const [, setMessage] = useState({ text: '', key: 0 });
  const announce = useCallback((text, politeness = 'polite') => {
    // Update with unique key to force re-render
    setMessage(prev => ({ text, key: prev.key + 1 }));

    // Also update the global announcer if it exists
    const announcer = document.getElementById(`aria-live-${politeness}`);
    if (announcer) {
      announcer.textContent = '';
      // Small delay to ensure screen readers pick up the change
      setTimeout(() => {
        announcer.textContent = text;
      }, 100);
    }
  }, []);

  return announce;
}

/**
 * AriaLiveRegion Component
 *
 * Global live region container. Place once at app root.
 * Works with useAnnounce hook for programmatic announcements.
 *
 * Usage:
 *   // In App.js
 *   <AriaLiveRegion />
 */
export function AriaLiveRegion() {
  return (
    <>
      <div
        id="aria-live-polite"
        role="status"
        aria-live="polite"
        aria-atomic="true"
        className="visually-hidden"
      />
      <div
        id="aria-live-assertive"
        role="alert"
        aria-live="assertive"
        aria-atomic="true"
        className="visually-hidden"
      />
    </>
  );
}

export default AriaLive;
