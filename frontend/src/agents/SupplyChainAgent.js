import React from 'react';
import { AgentPlaceholderShell } from '../components';

function SupplyChainAgent() {
  return (
    <AgentPlaceholderShell
      title="Supply Chain"
      subtitle="Visualize supply chain impact and event-driven disruption analysis."
      outcomes={[
        { iconSrc: '/assets/icons/supply-chain-management.png', title: 'Supply lines', description: 'Map suppliers, routes, and dependencies.' },
        { iconSrc: '/assets/icons/alerts.png', title: 'Event impact', description: 'Simulate how disruptions affect your network.' },
        { iconSrc: '/assets/icons/monitoring.png', title: 'Live monitoring', description: 'Track risk signals across your chain.' },
      ]}
    >
      <div className="agent-placeholder-card">
        <span className="agent-placeholder-badge">Coming soon</span>
        <h2>Supply Chain Dashboard</h2>
        <p>Interactive visualizations and event impact analysis are under development. This preview shows the planned workspace layout.</p>
        <div className="supply-chain-dashboard-placeholder">
          <p>Network graph and scenario modeling will appear here.</p>
        </div>
      </div>
    </AgentPlaceholderShell>
  );
}

export default SupplyChainAgent;
