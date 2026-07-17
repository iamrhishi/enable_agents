import { useEffect, useRef, useCallback } from 'react';

/**
 * useFocusTrap Hook
 *
 * Traps focus within a container (for modals, dialogs, dropdowns).
 * - Tab cycles through focusable elements
 * - Shift+Tab cycles backwards
 * - ESC key calls onEscape callback
 * - Returns focus to trigger element on unmount
 *
 * Usage:
 *   const modalRef = useFocusTrap({
 *     isActive: isModalOpen,
 *     onEscape: closeModal
 *   });
 *
 *   <div ref={modalRef} role="dialog" aria-modal="true">
 *     ...modal content...
 *   </div>
 */

const FOCUSABLE_SELECTORS = [
  'button:not([disabled])',
  'input:not([disabled])',
  'select:not([disabled])',
  'textarea:not([disabled])',
  'a[href]',
  '[tabindex]:not([tabindex="-1"])',
  '[contenteditable="true"]'
].join(', ');

function useFocusTrap({ isActive = true, onEscape, autoFocus = true }) {
  const containerRef = useRef(null);
  const previousActiveElement = useRef(null);

  // Get all focusable elements within the container
  const getFocusableElements = useCallback(() => {
    if (!containerRef.current) return [];
    return Array.from(containerRef.current.querySelectorAll(FOCUSABLE_SELECTORS))
      .filter((el) => {
        // Filter out hidden elements
        return el.offsetParent !== null;
      });
  }, []);

  // Handle keydown events
  const handleKeyDown = useCallback((e) => {
    if (!isActive || !containerRef.current) return;

    // ESC key
    if (e.key === 'Escape' && onEscape) {
      e.preventDefault();
      onEscape();
      return;
    }

    // Tab key - trap focus
    if (e.key === 'Tab') {
      const focusableElements = getFocusableElements();
      if (focusableElements.length === 0) {
        e.preventDefault();
        return;
      }

      const firstElement = focusableElements[0];
      const lastElement = focusableElements[focusableElements.length - 1];

      // Shift + Tab (backwards)
      if (e.shiftKey) {
        if (document.activeElement === firstElement) {
          e.preventDefault();
          lastElement.focus();
        }
      } else {
        // Tab (forwards)
        if (document.activeElement === lastElement) {
          e.preventDefault();
          firstElement.focus();
        }
      }
    }
  }, [isActive, onEscape, getFocusableElements]);

  // Set up focus trap
  useEffect(() => {
    if (!isActive) return;

    // Store the currently focused element
    previousActiveElement.current = document.activeElement;

    // Add keydown listener
    document.addEventListener('keydown', handleKeyDown);

    // Auto-focus first focusable element
    if (autoFocus) {
      const focusableElements = getFocusableElements();
      if (focusableElements.length > 0) {
        // Small delay to ensure DOM is ready
        requestAnimationFrame(() => {
          focusableElements[0].focus();
        });
      }
    }

    // Cleanup
    return () => {
      document.removeEventListener('keydown', handleKeyDown);

      // Return focus to previous element
      if (previousActiveElement.current && typeof previousActiveElement.current.focus === 'function') {
        previousActiveElement.current.focus();
      }
    };
  }, [isActive, autoFocus, handleKeyDown, getFocusableElements]);

  return containerRef;
}

/**
 * useKeyboardShortcut Hook
 *
 * Listens for keyboard shortcuts.
 *
 * Usage:
 *   useKeyboardShortcut('Escape', handleClose);
 *   useKeyboardShortcut('Enter', handleSubmit, { ctrl: true });
 */
function useKeyboardShortcut(key, callback, { ctrl = false, shift = false, alt = false, meta = false } = {}) {
  useEffect(() => {
    const handleKeyDown = (e) => {
      const matchesKey = e.key === key || e.code === key;
      const matchesModifiers =
        e.ctrlKey === ctrl &&
        e.shiftKey === shift &&
        e.altKey === alt &&
        e.metaKey === meta;

      if (matchesKey && matchesModifiers) {
        e.preventDefault();
        callback(e);
      }
    };

    document.addEventListener('keydown', handleKeyDown);
    return () => document.removeEventListener('keydown', handleKeyDown);
  }, [key, callback, ctrl, shift, alt, meta]);
}

export { useFocusTrap, useKeyboardShortcut };
export default useFocusTrap;
