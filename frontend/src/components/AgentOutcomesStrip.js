import React from 'react';
import './AgentOutcomesStrip.css';

function AgentOutcomesStrip({ items = [] }) {
  if (!items.length) return null;

  return (
    <div className="agent-outcomes">
      {items.map((item) => (
        <div key={item.title} className="agent-outcome-card">
          <span className="agent-outcome-icon" aria-hidden="true">
            {item.iconSrc ? (
              <img src={item.iconSrc} alt="" className="agent-outcome-icon-img" />
            ) : (
              item.icon
            )}
          </span>
          <div>
            <strong>{item.title}</strong>
            <p>{item.description}</p>
          </div>
        </div>
      ))}
    </div>
  );
}

export default AgentOutcomesStrip;
