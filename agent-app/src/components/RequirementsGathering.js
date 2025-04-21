import React, { useState, useEffect } from 'react';
import Header from './Header';
import '../styles/RequirementsGathering.css';

function RequirementsGathering() {
  const [overview, setOverview] = useState('');
  const [context, setContext] = useState('');
  const [countries, setCountries] = useState('');
  const [industries, setIndustries] = useState('');
  const [businessFunctions, setBusinessFunctions] = useState('');
  const [analysisFrameworks, setAnalysisFrameworks] = useState([]);
  const [responseFormat, setResponseFormat] = useState('');
  const [uploadedFile, setUploadedFile] = useState(null);
  const [aiRequirements, setAiRequirements] = useState([]);
  const [frameworkAnalysis, setFrameworkAnalysis] = useState('');
  const [previousPrompts, setPreviousPrompts] = useState([]); // State to store previous prompts
  const [showPromptsPopup, setShowPromptsPopup] = useState(false); // State to control prompts popup visibility
  const [showPopup, setShowPopup] = useState(false); // State to control popup visibility

  const handleFileUpload = (e) => {
    const file = e.target.files[0];
    setUploadedFile(file);
    console.log('Uploaded file:', file);
  };

  const handleGenerate = async () => {
    try {
      const payload = {
        overview,
        context,
        countries,
        industries,
        businessFunctions,
        analysisFrameworks,
        responseFormat,
      };

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
      setAiRequirements(data.requirements.split('\n'));
    } catch (error) {
      console.error('Error generating requirements:', error);
    }
  };

  const handleSavePrompt = async () => {
    try {
      const payload = {
        overview,
        context,
        countries,
        industries,
        businessFunctions,
        analysisFrameworks,
        responseFormat,
      };

      const response = await fetch('http://127.0.0.1:5000/save-prompt', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify(payload),
      });

      if (!response.ok) {
        throw new Error('Failed to save prompt');
      }

      alert('Prompt saved successfully!');
    } catch (error) {
      console.error('Error saving prompt:', error);
      alert('Failed to save prompt.');
    }
  };

  const handleFetchPreviousPrompts = async () => {
    try {
      const response = await fetch('http://127.0.0.1:5000/previous-prompts', {
        method: 'GET',
      });

      if (!response.ok) {
        throw new Error('Failed to fetch previous prompts');
      }

      const data = await response.json();
      setPreviousPrompts(data.prompts);
      setShowPromptsPopup(true);
    } catch (error) {
      console.error('Error fetching previous prompts:', error);
    }
  };
  const handleExportResponse = () => {
    if (aiRequirements.length === 0) {
      alert('No response to export.');
      return;
    }

    // Show the popup with export options
    setShowPopup(true);
  };

  const closePopup = () => {
    setShowPopup(false);
  };

  return (
    <div className="requirements-page">
      <Header />
      <div className="requirements-container">
        <div className="user-input">
          <h2>User Input</h2>
          <div className="input-group">
            <label>Requirement Overview</label>
            <textarea
              placeholder="Write 1-2 lines about your business requirements."
              value={overview}
              onChange={(e) => setOverview(e.target.value)}
              rows="3"
            />
          </div>
          <div className="input-group">
            <label>Context</label>
            <textarea
              placeholder="Provide additional context for your requirements."
              value={context}
              onChange={(e) => setContext(e.target.value)}
              rows="3"
            />
          </div>
          <div className="input-group">
            <label>Region / Country of Interest</label>
            <input
              type="text"
              placeholder="Country or Region of interest"
              value={countries}
              onChange={(e) => setCountries(e.target.value)}
            />
          </div>
          <div className="input-group">
            <label>Industry</label>
            <input
              type="text"
              placeholder="Enter Relevant industry"
              value={industries}
              onChange={(e) => setIndustries(e.target.value)}
            />
          </div>
          <div className="input-group">
            <label>Business Function</label>
            <input
              type="text"
              placeholder="Business function: Marketing, Sales, Finance, etc."
              value={businessFunctions}
              onChange={(e) => setBusinessFunctions(e.target.value)}
            />
          </div>
          <div className="input-group">
            <label>Analysis Frameworks</label>
            <select
              multiple
              value={analysisFrameworks}
              onChange={(e) =>
                setAnalysisFrameworks(Array.from(e.target.selectedOptions, (option) => option.value))
              }
            >
              <option value="PESTLE">PESTLE</option>
              <option value="VRIO">VRIO</option>
              <option value="3-Horizon">3-Horizon</option>
              <option value="5 Forces">5 Forces</option>
            </select>
          </div>
          <div className="input-group">
            <label>Response Format</label>
            <textarea
              placeholder="Specify the desired response format: User Stories, Use Cases, etc."
              value={responseFormat}
              onChange={(e) => setResponseFormat(e.target.value)}
              rows="3"
            />
          </div>
          <div className="input-group">
            <label>Upload File</label>
            <input type="file" onChange={handleFileUpload} />
            {uploadedFile && <p>Uploaded File: {uploadedFile.name}</p>}
          </div>
          <div className="button-group">
            <button className="generate-button" onClick={handleGenerate}>
              Generate
            </button>
            <button className="save-button" onClick={handleSavePrompt}>
              Save Prompt
            </button>
            <button className="previous-prompts-button" onClick={handleFetchPreviousPrompts}>
              Previous Prompts
            </button>
          </div>
        </div>

        <div className="ai-assisted">
          <h2>AI-Assisted Requirements</h2>
          <ul>
            {aiRequirements.map((requirement, index) => (
              <li key={index}>{requirement}</li>
            ))}
          </ul>
          <button className="export-button" onClick={handleExportResponse}>
            Export Response
          </button>
        </div>
      </div>
      {/* Popup for Export Options */}
      {showPopup && (
        <div className="popup-overlay">
          <div className="popup-content">
            <h3>Export Options</h3>
            <div className="export-icons">
              <img src="/assets/icons/gmail.png" alt="Gmail" title="Gmail" />
              <img src="/assets/icons/word.png" alt="Word" title="Word" />
              <img src="/assets/icons/pdf.png" alt="PDF" title="PDF" />
              <img src="/assets/icons/canva.png" alt="Canva" title="Canva" />
              <img src="/assets/icons/figma.png" alt="Figma" title="Figma" />
              <img src="/assets/icons/powerpoint.png" alt="PowerPoint" title="PowerPoint" />
            </div>
            <button className="close-popup-button" onClick={closePopup}>
              Close
            </button>
          </div>
        </div>
      )}
      {showPromptsPopup && (
        <div className="popup-overlay">
          <div className="popup-content">
            <h3>Previous Prompts</h3>
            <ul>
              {previousPrompts.map((prompt, index) => (
                <li key={index}>
                  <strong>Prompt ID:</strong> {prompt.id}
                  <br />
                  <strong>Overview:</strong> {prompt.overview}
                  <br />
                  <strong>Context:</strong> {prompt.context}
                  <br />
                  <strong>Countries:</strong> {prompt.countries}
                  <br />
                  <strong>Industries:</strong> {prompt.industries}
                  <br />
                  <strong>Business Functions:</strong> {prompt.businessFunctions}
                  <br />
                  <strong>Frameworks:</strong> {prompt.analysisFrameworks.join(', ')}
                  <br />
                  <strong>Response Format:</strong> {prompt.responseFormat}
                </li>
              ))}
            </ul>
            <button className="close-popup-button" onClick={() => setShowPromptsPopup(false)}>
              Close
            </button>
          </div>
        </div>
      )}
    </div>
  );
}

export default RequirementsGathering;