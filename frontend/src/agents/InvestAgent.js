import React from 'react';
import Header from '../core/Header';
import { BackButton, Input } from '../components';
import '../styles/AgentsAssembly.css';

// Placeholder for 11 parameters (to be defined later)
const parameterNames = Array.from({ length: 11 }, (_, i) => `Parameter ${i + 1}`);

function InvestAgent() {
  return (
    <>
      <Header />
      <BackButton />
      <div className="agent-dashboard invest-agent-dashboard">
        <h2 className="dashboard-title">Invest Agent Dashboard</h2>
        <div className="parameters-section">
          <h3>Financial Instrument Assessment</h3>
          <div className="parameters-grid">
            {parameterNames.map((param, idx) => (
              <div className="parameter-card" key={idx}>
                <div className="parameter-label">{param}</div>
                <input className="parameter-input" type="text" placeholder="Enter value..." disabled />
              </div>
            ))}
          </div>
          <div className="parameters-note">(Parameter names and logic will be defined soon.)</div>
        </div>
      </div>
    </>
  );
}

export default InvestAgent;
