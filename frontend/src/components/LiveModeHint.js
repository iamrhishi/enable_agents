import React, { useState } from 'react';
import { useSearchParams } from 'react-router-dom';
import { useDemoMode } from '../hooks/useDemoMode';
import './LiveModeHint.css';

const DISMISS_KEY = 'enableAgentsLiveHintDismissed';

function LiveModeHint({ message, requireProject = false }) {
  const { isDemoMode, switchToDemo } = useDemoMode();
  const [searchParams] = useSearchParams();
  const [dismissed, setDismissed] = useState(
    () => localStorage.getItem(DISMISS_KEY) === 'true'
  );

  if (isDemoMode || dismissed) return null;
  if (requireProject && !searchParams.get('project')) return null;

  const handleDismiss = () => {
    localStorage.setItem(DISMISS_KEY, 'true');
    setDismissed(true);
  };

  return (
    <div className="live-mode-hint" role="status">
      <p>
        {message || 'You\'re in Live mode — connect your data or switch to Demo to explore sample workflows.'}
      </p>
      <div className="live-mode-hint-actions">
        <button type="button" className="btn btn-primary btn-sm" onClick={switchToDemo}>
          Try Demo mode
        </button>
        <button type="button" className="btn btn-ghost btn-sm" onClick={handleDismiss}>
          Dismiss
        </button>
      </div>
    </div>
  );
}

export default LiveModeHint;
