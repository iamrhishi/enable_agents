import React, { forwardRef } from 'react';
import './Input.css';

/**
 * Input Component
 *
 * Centralized input component for consistent styling across the app.
 *
 * Usage:
 *   <Input placeholder="Enter email" />
 *   <Input variant="filled" size="lg" />
 *   <Input disabled error />
 */

const Input = forwardRef(({
  type = 'text',
  variant = 'outlined',  // 'outlined' | 'filled'
  size = 'md',           // 'sm' | 'md' | 'lg'
  error = false,
  success = false,
  disabled = false,
  fullWidth = false,
  className = '',
  icon,
  iconPosition = 'left',
  ...props
}, ref) => {
  const classes = [
    'input',
    `input--${variant}`,
    `input--${size}`,
    error && 'input--error',
    success && 'input--success',
    disabled && 'input--disabled',
    fullWidth && 'input--full-width',
    icon && `input--has-icon-${iconPosition}`,
    className
  ].filter(Boolean).join(' ');

  return (
    <div className="input-wrapper">
      {icon && iconPosition === 'left' && (
        <span className="input-icon input-icon--left">{icon}</span>
      )}
      <input
        ref={ref}
        type={type}
        className={classes}
        disabled={disabled}
        {...props}
      />
      {icon && iconPosition === 'right' && (
        <span className="input-icon input-icon--right">{icon}</span>
      )}
    </div>
  );
});

Input.displayName = 'Input';

/**
 * Textarea variant
 */
export const Textarea = forwardRef(({
  variant = 'outlined',
  size = 'md',
  error = false,
  success = false,
  disabled = false,
  fullWidth = true,
  rows = 3,
  className = '',
  ...props
}, ref) => {
  const classes = [
    'input',
    'input--textarea',
    `input--${variant}`,
    `input--${size}`,
    error && 'input--error',
    success && 'input--success',
    disabled && 'input--disabled',
    fullWidth && 'input--full-width',
    className
  ].filter(Boolean).join(' ');

  return (
    <textarea
      ref={ref}
      className={classes}
      disabled={disabled}
      rows={rows}
      {...props}
    />
  );
});

Textarea.displayName = 'Textarea';

export default Input;
