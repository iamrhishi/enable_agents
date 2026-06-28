import { useState, useCallback } from 'react';

/**
 * useToast Hook
 *
 * Manages toast notifications state.
 *
 * Usage:
 *   const { toasts, showToast, removeToast } = useToast();
 *
 *   // Show a toast
 *   showToast('Settings saved!', 'success');
 *   showToast('Something went wrong', 'error');
 *   showToast('Please check your input', 'warning');
 *   showToast('New update available', 'info');
 *
 *   // Render container in your component
 *   <ToastContainer toasts={toasts} removeToast={removeToast} />
 */

let toastId = 0;

function useToast() {
  const [toasts, setToasts] = useState([]);

  const showToast = useCallback((message, type = 'info', duration = 3000) => {
    const id = ++toastId;

    setToasts((prev) => [
      ...prev,
      { id, message, type, duration }
    ]);

    return id;
  }, []);

  const removeToast = useCallback((id) => {
    setToasts((prev) => prev.filter((toast) => toast.id !== id));
  }, []);

  // Convenience methods
  const success = useCallback((message, duration) => {
    return showToast(message, 'success', duration);
  }, [showToast]);

  const error = useCallback((message, duration) => {
    return showToast(message, 'error', duration);
  }, [showToast]);

  const warning = useCallback((message, duration) => {
    return showToast(message, 'warning', duration);
  }, [showToast]);

  const info = useCallback((message, duration) => {
    return showToast(message, 'info', duration);
  }, [showToast]);

  return {
    toasts,
    showToast,
    removeToast,
    // Convenience methods
    toast: {
      success,
      error,
      warning,
      info
    }
  };
}

export default useToast;
