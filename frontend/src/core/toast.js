/**
 * Platform toast notifications — replaces browser alert() / confirm() dialogs.
 *
 * Usage (anywhere in the app, no context/provider needed):
 *   import { showToast } from '../core/toast';
 *   showToast('Saved!');                          // default: info, 3 s
 *   showToast('Login failed', 'error');
 *   showToast('Email sent', 'success');
 *   showToast('Check your settings', 'warning');
 */

const ICONS = {
  success: '✓',
  error: '✕',
  warning: '⚠',
  info: 'ℹ',
};

let _container = null;

function getContainer() {
  if (_container && document.body.contains(_container)) return _container;
  _container = document.createElement('div');
  _container.id = 'ea-toast-root';
  document.body.appendChild(_container);

  const style = document.createElement('style');
  style.textContent = `
    #ea-toast-root {
      position: fixed;
      top: 20px;
      right: 20px;
      z-index: 99999;
      display: flex;
      flex-direction: column;
      gap: 10px;
      pointer-events: none;
    }
    .ea-toast {
      display: flex;
      align-items: flex-start;
      gap: 10px;
      min-width: 280px;
      max-width: 420px;
      padding: 14px 18px;
      border-radius: 10px;
      font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
      font-size: 14px;
      line-height: 1.4;
      color: #fff;
      box-shadow: 0 4px 16px rgba(0,0,0,0.18);
      pointer-events: all;
      cursor: pointer;
      animation: ea-toast-in 0.25s ease forwards;
      position: relative;
      overflow: hidden;
    }
    .ea-toast.ea-toast-out {
      animation: ea-toast-out 0.3s ease forwards;
    }
    .ea-toast--success { background: #16a34a; }
    .ea-toast--error   { background: #dc2626; }
    .ea-toast--warning { background: #d97706; }
    .ea-toast--info    { background: #2563eb; }
    .ea-toast__icon {
      font-size: 16px;
      flex-shrink: 0;
      margin-top: 1px;
    }
    .ea-toast__msg { flex: 1; word-break: break-word; }
    .ea-toast__progress {
      position: absolute;
      bottom: 0; left: 0;
      height: 3px;
      background: rgba(255,255,255,0.4);
      animation: ea-progress linear forwards;
    }
    @keyframes ea-toast-in {
      from { opacity: 0; transform: translateX(30px); }
      to   { opacity: 1; transform: translateX(0); }
    }
    @keyframes ea-toast-out {
      from { opacity: 1; transform: translateX(0); }
      to   { opacity: 0; transform: translateX(30px); }
    }
    @keyframes ea-progress {
      from { width: 100%; }
      to   { width: 0%; }
    }
  `;
  document.head.appendChild(style);
  return _container;
}

/**
 * Show a platform toast notification.
 * @param {string} message  - Text to display
 * @param {'success'|'error'|'warning'|'info'} type - Visual variant
 * @param {number} duration - Auto-dismiss delay in ms (default 4000)
 */
export function showToast(message, type = 'info', duration = 4000) {
  const container = getContainer();

  const toast = document.createElement('div');
  toast.className = `ea-toast ea-toast--${type}`;
  toast.title = 'Click to dismiss';
  toast.innerHTML = `
    <span class="ea-toast__icon">${ICONS[type] ?? ICONS.info}</span>
    <span class="ea-toast__msg">${message}</span>
    <span class="ea-toast__progress" style="animation-duration:${duration}ms"></span>
  `;

  const dismiss = () => {
    toast.classList.add('ea-toast-out');
    toast.addEventListener('animationend', () => toast.remove(), { once: true });
  };

  toast.addEventListener('click', dismiss);
  container.appendChild(toast);
  setTimeout(dismiss, duration);
}

export default showToast;
