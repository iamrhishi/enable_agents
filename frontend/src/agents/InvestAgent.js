import React from 'react';
import { AgentPlaceholderShell } from '../components';

const parameterNames = Array.from({ length: 11 }, (_, i) => `Parameter ${i + 1}`);

function InvestAgent() {
  return (
    <AgentPlaceholderShell
      title="Invest Agent"
      subtitle="Financial instrument assessment and portfolio analysis."
      outcomes={[
        { iconSrc: '/assets/icons/bar-chart.png', title: 'Instrument scoring', description: 'Evaluate financial instruments against your criteria.' },
        { iconSrc: '/assets/icons/reports.png', title: 'Risk reports', description: 'Generate assessment summaries and comparisons.' },
        { iconSrc: '/assets/icons/save-money.png', title: 'Portfolio fit', description: 'Match investments to your strategy and constraints.' },
      ]}
    >
      <div className="agent-placeholder-card">
        <span className="agent-placeholder-badge">Coming soon</span>
        <h2>Invest Agent Dashboard</h2>
        <p>Parameter-driven financial instrument assessment is under development. Check back for scoring models and portfolio tools.</p>
        <div className="parameters-grid">
          {parameterNames.map((param, idx) => (
            <div className="parameter-card" key={idx}>
              <div className="parameter-label">{param}</div>
              <input className="parameter-input" type="text" placeholder="Enter value..." disabled />
            </div>
          ))}
        </div>
      </div>
    </AgentPlaceholderShell>
  );
}

export default InvestAgent;
