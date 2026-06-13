import React, { forwardRef } from 'react';
import './Select.css';

/**
 * Select Component
 *
 * Centralized dropdown/select component for consistent styling.
 *
 * Usage:
 *   <Select>
 *     <option value="">Choose...</option>
 *     <option value="1">Option 1</option>
 *   </Select>
 *
 *   <Select variant="filled" size="sm" />
 */

const Select = forwardRef(({
  variant = 'pill',      // 'outlined' | 'filled' | 'pill'
  size = 'md',           // 'sm' | 'md' | 'lg'
  error = false,
  disabled = false,
  fullWidth = false,
  className = '',
  children,
  ...props
}, ref) => {
  const classes = [
    'select',
    `select--${variant}`,
    `select--${size}`,
    error && 'select--error',
    disabled && 'select--disabled',
    fullWidth && 'select--full-width',
    className
  ].filter(Boolean).join(' ');

  return (
    <div className="select-wrapper">
      <select
        ref={ref}
        className={classes}
        disabled={disabled}
        {...props}
      >
        {children}
      </select>
      <span className="select-chevron">
        <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
          <path d="M6 9l6 6 6-6" />
        </svg>
      </span>
    </div>
  );
});

Select.displayName = 'Select';

export default Select;
