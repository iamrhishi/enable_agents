import React, { useEffect, useRef, useCallback } from 'react';
import './Modal.css';

/**
 * Modal Component
 *
 * Accessible, focus-trapped modal dialog.
 *
 * @param {boolean} open - Whether modal is visible
 * @param {function} onClose - Close handler (called on overlay click, Escape key, close button)
 * @param {string} title - Modal title (rendered in header)
 * @param {string} size - 'sm' | 'md' | 'lg' | 'xl' | 'full' (default: 'md')
 * @param {boolean} showClose - Show close button (default: true)
 * @param {React.ReactNode} footer - Optional footer content
 * @param {React.ReactNode} children - Modal body content
 */
export function Modal({
  open,
  onClose,
  title,
  size = 'md',
  showClose = true,
  footer,
  className = '',
  children,
  ...props
}) {
  const modalRef = useRef(null);
  const previousFocus = useRef(null);

  // Store previous focus and focus modal on open
  useEffect(() => {
    if (open) {
      previousFocus.current = document.activeElement;
      // Focus first focusable element or modal itself
      setTimeout(() => {
        const focusable = modalRef.current?.querySelector(
          'button, [href], input, select, textarea, [tabindex]:not([tabindex="-1"])'
        );
        (focusable || modalRef.current)?.focus();
      }, 0);
    } else if (previousFocus.current) {
      previousFocus.current.focus();
    }
  }, [open]);

  // Focus trap
  const handleKeyDown = useCallback((e) => {
    if (!open) return;

    if (e.key === 'Escape') {
      e.preventDefault();
      onClose?.();
      return;
    }

    if (e.key === 'Tab') {
      const focusable = modalRef.current?.querySelectorAll(
        'button, [href], input, select, textarea, [tabindex]:not([tabindex="-1"])'
      );
      if (!focusable?.length) return;

      const first = focusable[0];
      const last = focusable[focusable.length - 1];

      if (e.shiftKey && document.activeElement === first) {
        e.preventDefault();
        last.focus();
      } else if (!e.shiftKey && document.activeElement === last) {
        e.preventDefault();
        first.focus();
      }
    }
  }, [open, onClose]);

  useEffect(() => {
    document.addEventListener('keydown', handleKeyDown);
    return () => document.removeEventListener('keydown', handleKeyDown);
  }, [handleKeyDown]);

  // Prevent body scroll when open
  useEffect(() => {
    if (open) {
      document.body.style.overflow = 'hidden';
    } else {
      document.body.style.overflow = '';
    }
    return () => { document.body.style.overflow = ''; };
  }, [open]);

  if (!open) return null;

  return (
    <div
      className="modal-overlay"
      onClick={(e) => e.target === e.currentTarget && onClose?.()}
      role="presentation"
    >
      <div
        ref={modalRef}
        className={`modal modal--${size} ${className}`}
        role="dialog"
        aria-modal="true"
        aria-labelledby={title ? 'modal-title' : undefined}
        tabIndex={-1}
        {...props}
      >
        {(title || showClose) && (
          <div className="modal-header">
            {title && (
              <h2 id="modal-title" className="modal-title">
                {title}
              </h2>
            )}
            {showClose && (
              <button
                type="button"
                className="modal-close"
                onClick={onClose}
                aria-label="Close modal"
              >
                <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
                  <line x1="18" y1="6" x2="6" y2="18" />
                  <line x1="6" y1="6" x2="18" y2="18" />
                </svg>
              </button>
            )}
          </div>
        )}
        <div className="modal-body">
          {children}
        </div>
        {footer && (
          <div className="modal-footer">
            {footer}
          </div>
        )}
      </div>
    </div>
  );
}

/**
 * ModalTabs Component
 *
 * Tab navigation for modal content.
 *
 * @param {Array} tabs - Array of { id, label, content }
 * @param {string} activeTab - Currently active tab ID
 * @param {function} onTabChange - Tab change handler
 */
export function ModalTabs({
  tabs,
  activeTab,
  onTabChange,
  className = '',
}) {
  return (
    <div className={`modal-tabs ${className}`}>
      <div className="modal-tabs-nav" role="tablist">
        {tabs.map((tab) => (
          <button
            key={tab.id}
            role="tab"
            aria-selected={activeTab === tab.id}
            aria-controls={`tabpanel-${tab.id}`}
            className={`modal-tab ${activeTab === tab.id ? 'modal-tab--active' : ''}`}
            onClick={() => onTabChange(tab.id)}
          >
            {tab.label}
          </button>
        ))}
      </div>
      <div className="modal-tabs-content">
        {tabs.map((tab) => (
          <div
            key={tab.id}
            id={`tabpanel-${tab.id}`}
            role="tabpanel"
            aria-labelledby={tab.id}
            hidden={activeTab !== tab.id}
            className="modal-tabpanel"
          >
            {tab.content}
          </div>
        ))}
      </div>
    </div>
  );
}

export default Modal;
