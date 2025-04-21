import React, { useState } from 'react';
import Header from './Header';
import '../styles/DataInsights.css';

function DataInsights() {
  const [uploadedFile, setUploadedFile] = useState(null);
  const [inputPrompt, setInputPrompt] = useState('');
  const [insights, setInsights] = useState('');
  const [previousPrompts, setPreviousPrompts] = useState([]);

  const handleFileUpload = (e) => {
    const file = e.target.files[0];
    setUploadedFile(file);
    console.log('Uploaded file:', file);
  };

  const handleGetInsights = () => {
    if (inputPrompt.trim() === '') {
      alert('Please enter a prompt to get insights.');
      return;
    }

    // Simulate fetching insights (replace with API call if needed)
    const generatedInsights = `Insights for: "${inputPrompt}"`;
    setInsights(generatedInsights);

    // Add the prompt to the previous prompts list
    setPreviousPrompts((prev) => [inputPrompt, ...prev]);
    setInputPrompt(''); // Clear the input box
  };

  return (
    <div className="data-insights-page">
      <Header />
      <div className="data-insights-container">
        {/* Left Section: File Upload and Input */}
        <div className="input-section">
          <h2>Data Insights</h2>
          <div className="input-group">
            <label>Upload File</label>
            <input type="file" onChange={handleFileUpload} />
            {uploadedFile && <p>Uploaded File: {uploadedFile.name}</p>}
          </div>
          <div className="input-group">
            <label>Enter Prompt</label>
            <input
              type="text"
              placeholder="Enter your prompt here"
              value={inputPrompt}
              onChange={(e) => setInputPrompt(e.target.value)}
            />
          </div>
          <button className="get-insights-button" onClick={handleGetInsights}>
            Get Insights
          </button>
          <div className="insights-output">
            <label>Insights</label>
            <textarea
              value={insights}
              readOnly
              rows="6"
              placeholder="Generated insights will appear here..."
            />
          </div>
        </div>

        {/* Right Section: Previous Prompts */}
        <div className="previous-prompts">
          <h2>Previous Prompts</h2>
          <ul>
            {previousPrompts.map((prompt, index) => (
              <li key={index}>{prompt}</li>
            ))}
          </ul>
        </div>
      </div>
    </div>
  );
}

export default DataInsights;