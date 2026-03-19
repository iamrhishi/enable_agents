import React from 'react';
import { BrowserRouter as Router, Routes, Route } from 'react-router-dom';
import Login from './components/Login';
import RegisterUser from './components/RegisterUser';
import RequirementsGathering from './components/RequirementsGathering';
import AgentsAssembly from './components/AgentsAssembly';
import EventNetworkingAgent from './components/EventNetworkingAgent';
import DataInsights from './components/DataInsights'; 
import Chatbot from './components/Chatbot';
import CommunityNetworkAgent from './components/CommunityNetworkAgent';
import SalesHelperAgent from './components/SalesHelperAgent';
import ContentMarketingAgent from './components/ContentMarketingAgent';
import InvestAgent from './components/InvestAgent';
import SupplyChainAgent from './components/SupplyChainAgent';
import ExecutiveAssistantPage from './components/ExecutiveAssistantPage';


function App() {
  return (
    <Router>
      <div className="App">
        <Routes>
          <Route path="/" element={<Login />} />
           <Route path="/register" element={<RegisterUser />} />
          <Route path="/requirements" element={<RequirementsGathering />} />
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
        </Routes>
      </div>
    </Router>
  );
}

export default App;