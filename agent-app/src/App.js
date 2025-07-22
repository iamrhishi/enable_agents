import React from 'react';
import { BrowserRouter as Router, Routes, Route } from 'react-router-dom';
import Login from './components/Login';
import RegisterUser from './components/RegisterUser';
import RequirementsGathering from './components/RequirementsGathering';
import AgentsAssembly from './components/AgentsAssembly';
import DataInsights from './components/DataInsights'; 
import Chatbot from './components/Chatbot';
import CommunityNetworkAgent from './components/CommunityNetworkAgent';


function App() {
  return (
    <Router>
      <div className="App">
        <Routes>
          <Route path="/" element={<Login />} />
           <Route path="/register" element={<RegisterUser />} />
          <Route path="/requirements" element={<RequirementsGathering />} />
          <Route path="/agents" element={<AgentsAssembly />} />
          <Route path="/datainsights" element={<DataInsights />} /> 
          <Route path="/aichatbot" element={<Chatbot />} />
          <Route path="/community-network" element={<CommunityNetworkAgent />} />
        </Routes>
      </div>
    </Router>
  );
}

export default App;