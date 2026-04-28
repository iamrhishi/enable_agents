import React from 'react';
import Header from '../core/Header';
import '../styles/AgentsAssembly.css';

function SupplyChainAgent() {
  return (
    <>
      <Header />
      <div className="agent-dashboard supply-chain-agent-dashboard">
        <h2 className="dashboard-title">Supply Chain Agent Dashboard</h2>
        <div className="supply-chain-visual-section">
          <h3>Visualize Supply Chain Impact</h3>
          <div className="supply-chain-dashboard-placeholder">
            <p>This dashboard will visually show the impact of different events on your supply lines.</p>
            <p>(Interactive visualizations and event impact analysis coming soon.)</p>
          </div>
        </div>
      </div>
    </>
  );
}

export default SupplyChainAgent;
