import React, { useState, useEffect } from 'react';
import Header from './Header';
import '../styles/RequirementsGathering.css';

function RequirementsGathering() {
  const [overview, setOverview] = useState('');
  const [countries, setCountries] = useState('');
  const [industries, setIndustries] = useState('');
  const [businessFunctions, setBusinessFunctions] = useState('');
  const [finalRequirements, setFinalRequirements] = useState([]);
  const [aiRequirements, setAiRequirements] = useState([]);

  // Update final requirements whenever any input changes
  useEffect(() => {
    const combinedRequirements = [
      countries.trim(),
      industries.trim(),
      businessFunctions.trim(),
    ].filter((item) => item !== ''); // Filter out empty inputs
    setFinalRequirements(combinedRequirements);
  }, [countries, industries, businessFunctions]);

  const handleDelete = (index) => {
    setFinalRequirements((prev) => prev.filter((_, i) => i !== index));
  };

  const handleGenerate = async () => {
    try {
      // Create the payload for the API
      const payload = {
        overview,
        keywords: finalRequirements,
      };

      // Call the generate-requirements API
      const response = await fetch('http://127.0.0.1:5000/generate-requirements', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify(payload),
      });

      if (!response.ok) {
        throw new Error('Failed to generate requirements');
      }

      const data = await response.json();

      // Update the AI-Assisted Requirements section with the API response
      setAiRequirements(data.requirements.split('\n')); // Split the response into a list
    } catch (error) {
      console.error('Error generating requirements:', error);
    }
  };

  return (
    <div className="requirements-page">
      <Header />
      <div className="requirements-container">
        {/* Left Section: User Input */}
        <div className="user-input">
          <h2>User Input</h2>
          <div className="input-group">
            <label>Requirement Overview</label>
            <textarea
              placeholder="Write 1-2 lines about your software requirements, context, and problem."
              value={overview}
              onChange={(e) => setOverview(e.target.value)}
              rows="3"
            />
          </div>
          <div className="input-group">
            <label>Country</label>
            <input
              type="text"
              placeholder="Enter country"
              value={countries}
              onChange={(e) => setCountries(e.target.value)}
            />
          </div>
          <div className="input-group">
            <label>Industry</label>
            <input
              type="text"
              placeholder="Enter industry"
              value={industries}
              onChange={(e) => setIndustries(e.target.value)}
            />
          </div>
          <div className="input-group">
            <label>Business Function</label>
            <input
              type="text"
              placeholder="Enter business function"
              value={businessFunctions}
              onChange={(e) => setBusinessFunctions(e.target.value)}
            />
          </div>
          <div className="input-group">
            <label>Final Requirements Prompt</label>
            <div className="tags">
              {finalRequirements.map((requirement, index) => (
                <span key={index} className="tag">
                  {requirement}
                  <button
                    className="delete-tag"
                    onClick={() => handleDelete(index)}
                  >
                    ✖
                  </button>
                </span>
              ))}
            </div>
          </div>
          <button className="generate-button" onClick={handleGenerate}>
            Generate
          </button>
        </div>

        {/* Right Section: AI-Assisted Requirements */}
        <div className="ai-assisted">
          <h2>AI-Assisted Requirements</h2>
          <ul>
            {aiRequirements.map((requirement, index) => (
              <li key={index}>{requirement}</li>
            ))}
          </ul>
        </div>
      </div>
    </div>
  );
}

export default RequirementsGathering;