import React from 'react';
import { BrowserRouter as Router, Routes, Route, Navigate } from 'react-router-dom';

// Shared UI
import Login from './core/Login';
import SkipLink from './components/SkipLink';
import RegisterUser from './core/RegisterUser';
import Settings from './settings/Settings';

// Agent components
import AgentsAssembly from './components/AgentsAssembly';
import RequirementsGathering from './agents/RequirementsGathering';
import CampaignDashboard from './agents/CampaignDashboard';
import EventNetworkingAgent from './agents/EventNetworkingAgent';
import DataInsights from './agents/DataInsights';
import Chatbot from './agents/Chatbot';
import CommunityNetworkAgent from './agents/CommunityNetworkAgent';
import SalesHelperAgent from './agents/SalesHelperAgent';
import ContentMarketingAgent from './agents/ContentMarketingAgent';
import InvestAgent from './agents/InvestAgent';
import SupplyChainAgent from './agents/SupplyChainAgent';
import ExecutiveAssistantPage from './agents/ExecutiveAssistantPage';


// Check if user is logged in (has session token)
function isLoggedIn() {
  return Boolean(localStorage.getItem('sessionToken') || localStorage.getItem('userEmail'));
}

// Root redirect - go to /agents if logged in, /login otherwise
function RootRedirect() {
  return <Navigate to={isLoggedIn() ? '/agents' : '/login'} replace />;
}

function App() {
  return (
    <Router>
      <SkipLink />
      <div className="App">
        <main id="main-content">
        <Routes>
          <Route path="/" element={<RootRedirect />} />
          <Route path="/login" element={<Login />} />
           <Route path="/register" element={<RegisterUser />} />
          <Route path="/market-research" element={<RequirementsGathering />} />
          <Route path="/market-research/campaigns" element={<CampaignDashboard />} />
          {/* Redirects for old paths */}
          <Route path="/requirements" element={<Navigate to="/market-research" replace />} />
          <Route path="/campaign-dashboard" element={<Navigate to="/market-research/campaigns" replace />} />
          <Route path="/agents" element={<AgentsAssembly />} />
          <Route path="/agents-assembly" element={<AgentsAssembly />} />
          <Route path="/datainsights" element={<DataInsights />} /> 
          <Route path="/aichatbot" element={<Chatbot />} />
          <Route path="/community-network" element={<CommunityNetworkAgent />} />
          <Route path="/sales-helper" element={<SalesHelperAgent />} />
          <Route path="/content-marketing" element={<ContentMarketingAgent />} />
          <Route path="/event-networking-agent" element={<EventNetworkingAgent />} />
          <Route path="/invest-agent" element={<InvestAgent />} />
          <Route path="/supply-chain-agent" element={<SupplyChainAgent />} />
          <Route path="/executive-assistant" element={<ExecutiveAssistantPage />} />
          <Route path="/settings" element={<Settings />} />
        </Routes>
        </main>
      </div>
    </Router>
  );
}

export default App;