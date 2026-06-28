import React, { useState } from 'react';
import { useNavigate } from 'react-router-dom';
import Header from '../core/Header';
import { BackButton, Input, Textarea, Select } from '../components';
import '../styles/DataInsights.css';
import { PDFDocument } from 'pdf-lib';
import { API_CONFIG } from '../config/apiConfig';
import { showToast } from '../core/toast';

function DataInsights() {
  const navigate = useNavigate();
  const [files, setFiles] = useState([]);
  const [isBulkMode, setIsBulkMode] = useState(false);
  const [inputPrompt, setInputPrompt] = useState('');
  const [insights, setInsights] = useState('');
  const [previousPrompts, setPreviousPrompts] = useState([]);
  const [insightsEngine, setInsightsEngine] = useState('');
  const [operationCost, setOperationCost] = useState(null);
  const [operationTime, setOperationTime] = useState(null);
  const [isLoading, setIsLoading] = useState(false);

  const handleFileUpload = (e) => {
    let selectedFiles = Array.from(e.target.files);
    if (isBulkMode) {
      if (selectedFiles.length > 10) {
        showToast('Maximum 10 files allowed. Using first 10.', 'warning');
        selectedFiles = selectedFiles.slice(0, 10);
      }
    } else {
      selectedFiles = selectedFiles.slice(0, 1);
    }
    setFiles(selectedFiles);
  };

  const clearFiles = () => {
    setFiles([]);
  };

  const handleGenerateInsights = async () => {
    if (files.length === 0) {
      showToast('Please upload a file first.', 'warning');
      return;
    }

    if (insightsEngine !== 'Contextual Insights with RAG') {
      showToast('Please select "Contextual Insights with RAG" as the Insights Engine.', 'warning');
      return;
    }

    if (inputPrompt.trim() === '') {
      showToast('Please enter a prompt to get insights.', 'warning');
      return;
    }

    setIsLoading(true);
    const startTime = performance.now();

    try {
      let fileToUpload;
      let fileName;

      if (isBulkMode && files.length > 1) {
        // Merge PDFs for bulk mode
        const pdfFiles = files.filter(file => file.type === 'application/pdf');
        if (pdfFiles.length === 0) {
          showToast('No PDF files found in your selection.', 'warning');
          setIsLoading(false);
          return;
        }

        const mergedPdf = await PDFDocument.create();
        for (const file of pdfFiles) {
          const arrayBuffer = await file.arrayBuffer();
          const pdf = await PDFDocument.load(arrayBuffer);
          const copiedPages = await mergedPdf.copyPages(pdf, pdf.getPageIndices());
          copiedPages.forEach((page) => mergedPdf.addPage(page));
        }

        const mergedPdfBytes = await mergedPdf.save();
        const mergedPdfBlob = new Blob([mergedPdfBytes], { type: 'application/pdf' });
        fileToUpload = new File([mergedPdfBlob], 'merged_bulk_upload.pdf', { type: 'application/pdf' });
        fileName = 'merged_bulk_upload.pdf';
      } else {
        fileToUpload = files[0];
        fileName = files[0].name;
      }

      // Upload file
      const formData = new FormData();
      formData.append('file', fileToUpload);

      const uploadResponse = await fetch(`${API_CONFIG.API_URL}/upload`, {
        method: 'POST',
        body: formData,
      });

      if (!uploadResponse.ok) {
        throw new Error('Failed to upload file.');
      }

      const uploadData = await uploadResponse.json();
      if (uploadData.message === 'File already exists') {
        showToast('File already exists on server. Using cached version.', 'info');
      }

      // Get insights via RAG
      const ragResponse = await fetch(`${API_CONFIG.API_URL}/rag_test`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          query: inputPrompt,
          file_name: fileName
        }),
      });

      if (!ragResponse.ok) {
        throw new Error('Failed to fetch insights from the server.');
      }

      const ragData = await ragResponse.json();
      setInsights(ragData.answer);
      setPreviousPrompts((prev) => [inputPrompt, ...prev]);
      setInputPrompt('');

      // Calculate metrics
      const endTime = performance.now();
      const timeTaken = ((endTime - startTime) / 1000).toFixed(2);
      const cost = (timeTaken * 0.01).toFixed(2);
      setOperationTime(timeTaken);
      setOperationCost(cost);

    } catch (error) {
      console.error('Error fetching insights:', error);
      showToast('An error occurred while fetching insights. Please try again.', 'error');
    } finally {
      setIsLoading(false);
    }
  };

  return (
    <div className="data-insights-page">
      <Header />
      <BackButton />

      <div className="data-insights-container">
        {/* Configuration Panel */}
        <div className="config-panel">
          <div className="panel-header">
            <h2>Configuration</h2>
            <button
              className="settings-link"
              onClick={() => navigate('/settings?tab=connectors')}
            >
              Manage Connectors
            </button>
          </div>

          {/* Upload Section */}
          <div className="config-section">
            <div className="section-title">
              <span>Data Source</span>
              <label className="bulk-toggle">
                <input
                  type="checkbox"
                  checked={isBulkMode}
                  onChange={(e) => {
                    setIsBulkMode(e.target.checked);
                    setFiles([]);
                  }}
                />
                <span>Bulk mode (up to 10 files)</span>
              </label>
            </div>

            <div className="file-upload-area">
              <input
                type="file"
                id="file-input"
                onChange={handleFileUpload}
                multiple={isBulkMode}
                accept=".pdf,.csv,.xlsx,.xls,.txt,.json"
                className="file-input-hidden"
              />
              <label htmlFor="file-input" className="file-upload-label">
                <span className="upload-icon">+</span>
                <span>{isBulkMode ? 'Choose files (max 10)' : 'Choose a file'}</span>
              </label>
            </div>

            {files.length > 0 && (
              <div className="file-list">
                <div className="file-list-header">
                  <span>{files.length} file{files.length > 1 ? 's' : ''} selected</span>
                  <button className="clear-btn" onClick={clearFiles}>Clear</button>
                </div>
                <ul>
                  {files.map((file, idx) => (
                    <li key={idx}>{file.name}</li>
                  ))}
                </ul>
              </div>
            )}
          </div>

          {/* Insights Engine */}
          <div className="config-section">
            <label className="section-title">Insights Engine</label>
            <Select
              value={insightsEngine}
              onChange={(e) => setInsightsEngine(e.target.value)}
              variant="outlined"
            >
              <option value="">Select an engine...</option>
              <option value="Contextual Insights with RAG">Contextual Insights with RAG</option>
              <option value="Data Discovery with Graphs">Data Discovery with Graphs</option>
              <option value="Data Mapping">Data Mapping</option>
              <option value="OCR Scans">OCR Scans</option>
            </Select>
          </div>

          {/* Prompt */}
          <div className="config-section">
            <label className="section-title">Your Question</label>
            <Textarea
              placeholder="Ask a question about your data..."
              value={inputPrompt}
              onChange={(e) => setInputPrompt(e.target.value)}
              rows={3}
            />
          </div>

          {/* Generate Button */}
          <button
            className="generate-btn"
            onClick={handleGenerateInsights}
            disabled={isLoading || files.length === 0}
          >
            {isLoading ? 'Generating...' : 'Generate Insights'}
          </button>

          {/* Metrics */}
          {operationTime && operationCost && (
            <div className="metrics">
              <span>Time: {operationTime}s</span>
              <span>Cost: ${operationCost}</span>
            </div>
          )}

          {/* Previous Prompts */}
          {previousPrompts.length > 0 && (
            <div className="previous-prompts">
              <h4>Previous Queries</h4>
              <ul>
                {previousPrompts.slice(0, 5).map((prompt, index) => (
                  <li key={index} onClick={() => setInputPrompt(prompt)}>
                    {prompt}
                  </li>
                ))}
              </ul>
            </div>
          )}
        </div>

        {/* Results Panel */}
        <div className="results-panel">
          <h2>Insights</h2>
          <div className="insights-output">
            {insights ? (
              <div className="insights-content">{insights}</div>
            ) : (
              <div className="insights-placeholder">
                <p>Your generated insights will appear here.</p>
                <p className="hint">Upload a file, select an engine, and ask a question to get started.</p>
              </div>
            )}
          </div>
        </div>
      </div>
    </div>
  );
}

export default DataInsights;
