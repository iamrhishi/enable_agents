import React, { useState } from 'react';
import { useNavigate } from 'react-router-dom';
import Header from './Header';
import '../styles/AgentsAssembly.css';

function AgentsAssembly() {
  const [searchTerm, setSearchTerm] = useState('');
  const [selectedIndustry, setSelectedIndustry] = useState('');
  const [selectedProcess, setSelectedProcess] = useState('');
  const navigate = useNavigate();

  const businessModules = [
    { name: 'Market Research', icon: '/assets/icons/search-analysis.png' },
    { name: 'Orders', icon: '/assets/icons/orders.png' },
    { name: 'Inventory', icon: '/assets/icons/inventory.png' },
    { name: 'Suppliers', icon: '/assets/icons/suppliers.png' },
    { name: 'Documents', icon: '/assets/icons/documents.png' },
    { name: 'Reports', icon: '/assets/icons/reports.png' },
    { name: 'Invoices', icon: '/assets/icons/invoices.png' },
  ];

  const technicalModules = [
    { name: 'Community Network', icon: '/assets/icons/users.png' },
    { name: 'Testing AI', icon: '/assets/icons/checklist.png' },
    { name: 'LLM Benchmarking', icon: '/assets/icons/bar-chart.png' },
    { name: 'Data Discovery', icon: '/assets/icons/data-discovery.png' },
    { name: 'Users', icon: '/assets/icons/users.png' },
    { name: 'Data Security', icon: '/assets/icons/data-security.png' },
    { name: 'Alerts', icon: '/assets/icons/alerts.png' },
    { name: 'Notifications', icon: '/assets/icons/notifications.png' },
    { name: 'Dashboards', icon: '/assets/icons/dashboards.png' },
    { name: 'AI Chatbot', icon: '/assets/icons/ai-chatbots.png' },
    { name: 'Monitoring', icon: '/assets/icons/monitoring.png' },
    { name: 'Analytics', icon: '/assets/icons/analytics.png' },
    { name: 'Data Transformation', icon: '/assets/icons/data-transformation.png' },
    { name: 'Integration', icon: '/assets/icons/integration.png' },
    { name: 'Automation', icon: '/assets/icons/automation.png' }
  ];

  const filteredModules = [...businessModules, ...technicalModules].filter((module) =>
    module.name.toLowerCase().includes(searchTerm.toLowerCase())
  );

  // Card click handler
  const handleCardClick = (moduleName) => {
    if (moduleName === 'Data Discovery') {
      navigate('/datainsights');
    }
    else if (moduleName === 'Market Research') {
      navigate('/requirements');
    }
    else if (moduleName === 'AI Chatbot') {
      navigate('/aichatbot');
    }
    else if (moduleName === 'Community Network') {
    navigate('/community-network');
   }
    // You can add more navigation logic for other cards here if needed
  };

  return (
    <div className="agents-page">
      <Header />
      <div className="agents-assembly">
        <h2>Agents Assembly</h2>
        {/* <p className="highlight-blue">configure AI Agents that  </p> */}

        <div className="search-bar">
          <input
            type="text"
            placeholder="Describe your business requirements here to get Agentic module and feature recommendations"
            value={searchTerm}
            onChange={(e) => setSearchTerm(e.target.value)}
            className="search-input"
          />
        </div>
        <div className="dropdown-container">
          <select
            value={selectedIndustry}
            onChange={(e) => setSelectedIndustry(e.target.value)}
            className="dropdown"
          >
            <option value="">Select Industry</option>
            <option value="Retail">Retail</option>
            <option value="Manufacturing">Manufacturing</option>
            <option value="Healthcare">Healthcare</option>
            <option value="Finance">Finance</option>
          </select>
          <select
            value={selectedProcess}
            onChange={(e) => setSelectedProcess(e.target.value)}
            className="dropdown"
          >
            <option value="">Select Process</option>
            <option value="Sales">Sales</option>
            <option value="Procurement">Procurement</option>
            <option value="HR">HR</option>
            <option value="Operations">Operations</option>
          </select>
        </div>

        {/* Modules Section */}
        <div className="modules-container">
          {filteredModules.map((module, index) => (
            <div
              key={index}
              className={`module-card ${
                businessModules.some((b) => b.name === module.name)
                  ? 'business-module'
                  : 'technical-module'
              }`}
              onClick={() => handleCardClick(module.name)}
              style={{ cursor: 'pointer' }}
            >
              <img src={module.icon} alt={module.name} />
              <p>{module.name}</p>
            </div>
          ))}
        </div>
      </div>
    </div>
  );
}

export default AgentsAssembly;