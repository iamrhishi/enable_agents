import React from 'react';
import './Spinner.css';

/**
 * Spinner Component
 *
 * Usage:
 *   <Spinner />
 *   <Spinner size="sm" />
 *   <Spinner size="lg" color="accent" />
 *   <Spinner.Button>Saving...</Spinner.Button>
 */

function Spinner({ size = 'md', color = 'primary', className = '' }) {
  return (
    <div
      className={`spinner spinner-${size} spinner-${color} ${className}`}
      role="status"
      aria-label="Loading"
    >
      <svg viewBox="0 0 24 24" fill="none">
        <circle
          className="spinner-track"
          cx="12"
          cy="12"
          r="10"
          strokeWidth="3"
        />
        <circle
          className="spinner-fill"
          cx="12"
          cy="12"
          r="10"
          strokeWidth="3"
          strokeLinecap="round"
        />
      </svg>
    </div>
  );
}

// Spinner with text (for buttons)
function SpinnerButton({ children, size = 'sm' }) {
  return (
    <span className="spinner-button">
      <Spinner size={size} color="inherit" />
      {children && <span className="spinner-button-text">{children}</span>}
    </span>
  );
}

// Full page loading overlay
function SpinnerOverlay({ message = 'Loading...' }) {
  return (
    <div className="spinner-overlay">
      <div className="spinner-overlay-content">
        <Spinner size="lg" />
        {message && <p className="spinner-overlay-message">{message}</p>}
      </div>
    </div>
  );
}

// Inline loading indicator
function SpinnerInline({ text = 'Loading' }) {
  return (
    <span className="spinner-inline">
      <Spinner size="sm" />
      <span>{text}</span>
    </span>
  );
}

Spinner.Button = SpinnerButton;
Spinner.Overlay = SpinnerOverlay;
Spinner.Inline = SpinnerInline;

export default Spinner;
