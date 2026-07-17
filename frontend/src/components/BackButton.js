import React from 'react';
import { useNavigate } from 'react-router-dom';
import './BackButton.css';

/**
 * Reusable back button component for agent pages.
 *
 * Usage: <BackButton /> or <BackButton to="/custom-path" />
 */
function BackButton({ to = '/agents-assembly', label = 'Back to Agents' }) {
  const navigate = useNavigate();

  const handleClick = () => {
    if (to === 'back') {
      navigate(-1);
    } else {
      navigate(to);
    }
  };

  return (
    <button className="back-to-hub-button" onClick={handleClick} title={label}>
      <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
        <path d="M19 12H5M12 19l-7-7 7-7"/>
      </svg>
      <span>{label}</span>
    </button>
  );
}

export default BackButton;
