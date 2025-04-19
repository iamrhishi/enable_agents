import React from 'react';
import { BrowserRouter as Router, Routes, Route } from 'react-router-dom';
import Login from './components/Login';
import RequirementsGathering from './components/RequirementsGathering';
import AgentsAssembly from './components/AgentsAssembly';

function App() {
  return (
    <Router>
      <div className="App">
        <Routes>
          <Route path="/" element={<Login />} />
          <Route path="/requirements" element={<RequirementsGathering />} />
          <Route path="/agents" element={<AgentsAssembly />} />
        </Routes>
      </div>
    </Router>
  );
}

export default App;