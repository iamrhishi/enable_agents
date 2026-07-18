import React, { useState } from 'react';
import { useSearchParams } from 'react-router-dom';
import { useMode } from '../contexts';
import './LiveModeHint.css';

const DISMISS_KEY = 'enableAgentsLiveHintDismissed';

function LiveModeHint({ message, requireProject = false }) {
  const { isDemoMode, setMode } = useMode();
  const [searchParams] = useSearchParams();
  const [dismissed, setDismissed] = useState(
    () => localStorage.getItem(DISMISS_KEY) === 'true'
  );

  // Show only in Live mode without a project
  if (isDemoMode || dismissed) return null;

  // If requireProject is true, only show when no project is selected
  const hasProject = searchParams.get('project');
  if (requireProject && hasProject) return null;
  if (!requireProject) return null; // Only show when requireProject is set

  const handleDismiss = () => {
    localStorage.setItem(DISMISS_KEY, 'true');
    setDismissed(true);
  };

  const switchToDemo = () => {
    setMode(true); // true = demo mode
  };

  return (
    <div className="live-mode-hint-wrapper">
      <div className="live-mode-hint" role="status">
        <p>
          {message || 'Select a project or try Demo mode for sample data'}
        </p>
        <div className="live-mode-hint-actions">
          <button type="button" className="btn btn-primary btn-sm" onClick={switchToDemo}>
            Demo
          </button>
          <button type="button" className="btn btn-ghost btn-sm" onClick={handleDismiss}>
            ✕
          </button>
        </div>
      </div>
    </div>
  );
}

export default LiveModeHint;
