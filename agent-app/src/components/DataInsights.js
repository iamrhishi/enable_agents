import React, { useState } from 'react';
import Header from './Header';
import '../styles/DataInsights.css';

function DataInsights() {
  const [uploadedFile, setUploadedFile] = useState(null);
  const [inputPrompt, setInputPrompt] = useState('');
  const [insights, setInsights] = useState('');
  const [previousPrompts, setPreviousPrompts] = useState([]);
  const [dataSource, setDataSource] = useState('');
  const [isConnected, setIsConnected] = useState(false);
  const [insightsEngine, setInsightsEngine] = useState(''); // State for Insights Engine selection
  const [operationCost, setOperationCost] = useState(null);
  const [operationTime, setOperationTime] = useState(null);

  const handleFileUpload = (e) => {
    const file = e.target.files[0];
    setUploadedFile(file);
    setDataSource(''); // Clear data source selection
    setIsConnected(false); // Reset connection status
    console.log('Uploaded file:', file);
  };

  const handleConnect = () => {
    if (dataSource) {
      setUploadedFile(null); // Clear uploaded file
      setIsConnected(true);
      console.log(`Connected to data source: ${dataSource}`);
    } else {
      alert('Please select a data source.');
    }
  };

  const handleGetInsights = async () => {
    if (!uploadedFile) {
      alert('Please upload a file first.');
      return;
    }
  
    if (insightsEngine !== 'Contextual Insights with RAG') {
      alert('Please select "Contextual Insights with RAG" as the Insights Engine.');
      return;
    }
  
    if (inputPrompt.trim() === '') {
      alert('Please enter a prompt to get insights.');
      return;
    }
  
    try {

      const startTime = performance.now();

      // Step 1: Upload the file
      const formData = new FormData();
      formData.append('file', uploadedFile);
  
      const uploadResponse = await fetch('http://localhost:5000/upload', {
        method: 'POST',
        body: formData,
      });
  
      if (!uploadResponse.ok) {
        throw new Error('Failed to upload file.');
      }
  
      const uploadData = await uploadResponse.json();
      console.log('File upload response:', uploadData);
  
      if (uploadData.message === 'File already exists') {
        alert('The file already exists on the server. It will not be uploaded again.');
      } else {
        console.log('File uploaded successfully:', uploadData);
      }
  
      // Step 2: Call the /rag_test API
      const ragTestResponse = await fetch(
        `http://localhost:5000/rag_test?query=${encodeURIComponent(inputPrompt)}&file_name=${encodeURIComponent(uploadedFile.name)}`,
        {
          method: 'GET',
        }
      );
  
      if (!ragTestResponse.ok) {
        throw new Error('Failed to fetch insights from the server.');
      }
  
      const ragTestData = await ragTestResponse.json();
  
      // Access the "answer" property from the response
      const answer = ragTestData.answer;
  
      // Set the insights in the textarea
      setInsights(answer);
      console.log('Insights:', answer);
  
      // Add the prompt to the previous prompts list
      setPreviousPrompts((prev) => [inputPrompt, ...prev]); // Save all prompts
      setInputPrompt(''); // Clear the input box
      
          // Calculate and display cost and time
      const endTime = performance.now(); // End timer
      const timeTaken = ((endTime - startTime) / 1000).toFixed(2); // Time in seconds
      const cost = (timeTaken * 0.01).toFixed(2); // Example cost calculation (e.g., $0.01 per second)

      setOperationTime(timeTaken);
      setOperationCost(cost);
    } catch (error) {
      console.error('Error fetching insights:', error);
      alert('An error occurred while fetching insights. Please try again.');
    }
  };

  return (
    <div className="data-insights-page">
      <Header />
      <div className="data-insights-container">
        {/* Left Section */}
        <div className="left-section">
          <h2>Data Source</h2>

          {/* File Upload Section */}
          <div className="input-group">
            <label>Upload File</label>
            <input
              type="file"
              onChange={handleFileUpload}
              disabled={isConnected} // Disable if connected to a data source
            />
            {uploadedFile && <p>Uploaded File: {uploadedFile.name}</p>}
          </div>

          {/* Data Source Connection Section */}
          <div className="input-group">
            <label>Connect to Data Source</label>
            <select
              value={dataSource}
              onChange={(e) => setDataSource(e.target.value)}
              disabled={!!uploadedFile} // Disable if a file is uploaded
              className="data-source-select"
            >
              <option value="">Select Data Source</option>
              <option value="API">API</option>
              <option value="Database">Database</option>
              <option value="Cloud Storage">Cloud Storage</option>
            </select>
            <button
              onClick={handleConnect}
              className="connect-button"
              disabled={!!uploadedFile} // Disable if a file is uploaded
            >
              Connect
            </button>
            {isConnected && <p>Connected to: {dataSource}</p>}
          </div>

          {/* Insights Engine Dropdown */}
          <div className="input-group">
            <label>Select Insights Engine</label>
            <select
              value={insightsEngine}
              onChange={(e) => setInsightsEngine(e.target.value)}
              className="insights-engine-select"
            >
              <option value="">Select an Insights Engine</option>
              <option value="Contextual Insights with RAG">
                Contextual Insights with RAG
              </option>
              <option value="Data Discovery with Graphs">
                Data Discovery with Graphs
              </option>
              <option value="Data Mapping">Data Mapping</option>
              <option value="OCR Scans">OCR Scans</option>
            </select>
          </div>

          {/* Prompt Input Section */}
          <div className="input-group">
            <label>Context & Prompt</label>
            <input
              type="text"
              placeholder="Provide a short context and ask you question"
              value={inputPrompt}
              onChange={(e) => setInputPrompt(e.target.value)}
            />
          </div>

          {/* Get Insights Button */}
          <button className="get-insights-button" onClick={handleGetInsights}>
            Generate Insights
          </button>

          {/* Cost and Time Summary */}
          <div className="operation-summary">
            {operationTime && operationCost && (
              <p>
                Operation Time: <strong>{operationTime} seconds</strong> | Cost: <strong>${operationCost}</strong>
              </p>
            )}
          </div>

          {/* Previous Prompts Section */}
          <div className="previous-prompts">
            <h3>Previous Prompts</h3>
            <ul>
              {previousPrompts.map((prompt, index) => (
                <li key={index}>{prompt}</li>
              ))}
            </ul>
          </div>
        </div>

        {/* Right Section */}
        <div className="right-section">
          <h2>Data Insights</h2>
          <textarea
            value={insights}
            readOnly
            rows="20"
            placeholder="Generated insights will appear here..."
          />
        </div>
      </div>
    </div>
  );
}

export default DataInsights;