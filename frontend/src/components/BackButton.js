import React from 'react';
import { useNavigate, useSearchParams } from 'react-router-dom';
import './BackButton.css';

/**
 * Reusable back button component for agent pages.
 *
 * Automatically detects workflow context and navigates back to workflow.
 *
 * Usage: <BackButton /> or <BackButton to="/custom-path" />
 */
function BackButton({ to = '/dashboard', label = 'Back to Dashboard' }) {
  const navigate = useNavigate();
  const [searchParams] = useSearchParams();

  // Check if coming from workflow
  const workflowId = searchParams.get('workflow');
  const isFromWorkflow = Boolean(workflowId);

  const actualTo = isFromWorkflow ? `/workflows/${workflowId}` : to;
  const actualLabel = isFromWorkflow ? 'Back to Workflow' : label;

  const handleClick = () => {
    if (to === 'back') {
      navigate(-1);
    } else {
      navigate(actualTo);
    }
  };

  return (
    <button className="back-to-hub-button" onClick={handleClick} title={actualLabel}>
      <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
        <path d="M19 12H5M12 19l-7-7 7-7"/>
      </svg>
      <span>{actualLabel}</span>
    </button>
  );
}

export default BackButton;
