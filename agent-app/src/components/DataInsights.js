import React, { useState } from 'react';
import Header from './Header';
import '../styles/DataInsights.css';
import { PDFDocument } from 'pdf-lib';

function DataInsights() {
  const [uploadedFile, setUploadedFile] = useState(null);
  const [bulkFiles, setBulkFiles] = useState([]);
  const [inputPrompt, setInputPrompt] = useState('');
  const [insights, setInsights] = useState('');
  const [previousPrompts, setPreviousPrompts] = useState([]);
  const [dataSource, setDataSource] = useState('');
  const [isConnected, setIsConnected] = useState(false);
  const [insightsEngine, setInsightsEngine] = useState(''); // State for Insights Engine selection
  const [operationCost, setOperationCost] = useState(null);
  const [operationTime, setOperationTime] = useState(null);
  const [sourceLocation, setSourceLocation] = useState('');

  const handleFileUpload = (e) => {
    const file = e.target.files[0];
    setUploadedFile(file);
    setDataSource(''); // Clear data source selection
    setIsConnected(false); // Reset connection status
    console.log('Uploaded file:', file);
  };

  const handleBulkFileUpload = (e) => {
    let files = Array.from(e.target.files);
    if (files.length > 10) {
      alert('You can upload a maximum of 10 files at a time.');
      files = files.slice(0, 10);
    }
    setBulkFiles(files);
    console.log('Bulk uploaded files:', files);
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
      const ragTestResponse = await fetch('http://localhost:5000/rag_test', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          query: inputPrompt,
          file_name: uploadedFile.name
        }),
      });
  
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


  const handleBulkGetInsights = async () => {
  if (bulkFiles.length === 0) {
    alert('Please select up to 10 PDF files for bulk upload.');
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
    // Only include PDF files
    const pdfFiles = bulkFiles.filter(file => file.type === 'application/pdf');
    if (pdfFiles.length === 0) {
      alert('No PDF files found in your selection.');
      return;
    }

    // Create a new PDFDocument
    const mergedPdf = await PDFDocument.create();

    // For each PDF, fetch its ArrayBuffer and copy its pages into the merged PDF
    for (const file of pdfFiles) {
      const arrayBuffer = await file.arrayBuffer();
      const pdf = await PDFDocument.load(arrayBuffer);
      const copiedPages = await mergedPdf.copyPages(pdf, pdf.getPageIndices());
      copiedPages.forEach((page) => mergedPdf.addPage(page));
    }

    // Serialize the merged PDF to bytes (Uint8Array)
    const mergedPdfBytes = await mergedPdf.save();

    // Create a Blob and File from the merged PDF
    const mergedPdfBlob = new Blob([mergedPdfBytes], { type: 'application/pdf' });
    const mergedPdfFile = new File([mergedPdfBlob], 'merged_bulk_upload.pdf', { type: 'application/pdf' });

    // Upload the merged PDF file
    const formData = new FormData();
    formData.append('file', mergedPdfFile);

    const uploadResponse = await fetch('http://localhost:5000/upload', {
      method: 'POST',
      body: formData,
    });

    if (!uploadResponse.ok) {
      throw new Error('Failed to upload merged PDF file.');
    }

    const uploadData = await uploadResponse.json();
    console.log('Merged PDF upload response:', uploadData);

    // Now call your RAG API with the merged PDF file name
    const ragTestResponse = await fetch('http://localhost:5000/rag_test', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        query: inputPrompt,
        file_name: mergedPdfFile.name
      }),
    });

    if (!ragTestResponse.ok) {
      throw new Error('Failed to fetch insights from the server.');
    }

    const ragTestData = await ragTestResponse.json();
    const answer = ragTestData.answer;

    setInsights(answer);
    setPreviousPrompts((prev) => [inputPrompt, ...prev]);
    setInputPrompt('');
  } catch (error) {
    alert('An error occurred while processing your files.');
    console.error('Error in bulk PDF processing:', error);
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

          {/* Bulk Upload Section */}
          <div className="input-group">
            <label>Bulk Upload (up to 10 files)</label>
            <input
              type="file"
              multiple
              onChange={handleBulkFileUpload}
              disabled={isConnected}
            />
            {bulkFiles.length > 0 && (
              <ul>
                {bulkFiles.map((file, idx) => (
                  <li key={idx}>{file.name}</li>
                ))}
              </ul>
            )}
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
              <option value="Local File System">Local File System</option>
            </select>
            <div className="input-group">
              <label>Source Location</label>
              <input
                type="text"
                placeholder="Enter source folder or file location"
                value={sourceLocation}
                onChange={e => setSourceLocation(e.target.value)}
              />
            </div>
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

          <button
              className="get-insights-button"
              onClick={handleBulkGetInsights}
              disabled={bulkFiles.length === 0}
              style={{ marginTop: '10px' }}
            >
              Generate Insights (Bulk)
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