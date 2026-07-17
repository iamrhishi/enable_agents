import React from 'react';
import { useDemoMode } from '../hooks/useDemoMode';
import './DemoModeBadge.css';

function DemoModeBadge() {
  const { isDemoMode } = useDemoMode();
  if (!isDemoMode) return null;
  return <span className="demo-mode-badge">Demo</span>;
}

export default DemoModeBadge;
