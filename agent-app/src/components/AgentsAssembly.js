import React from 'react';
import Header from './Header';
import '../styles/AgentsAssembly.css';

function AgentsAssembly() {
  return (
    <div className="agents-page">
      <Header />
      <div className="agents-assembly">
        <h2>Agents Assembly</h2>
        <p>Here you can assemble your agents based on the gathered requirements.</p>
        <button>Start Assembly</button>
      </div>
    </div>
  );
}

export default AgentsAssembly;