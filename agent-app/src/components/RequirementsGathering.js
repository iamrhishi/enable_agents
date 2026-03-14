import React, { useState, useEffect } from 'react';
import Header from './Header';
import '../styles/RequirementsGathering.css';

function RequirementsGathering() {
  const [overview, setOverview] = useState('');
  const [context, setContext] = useState('');
  const [countries, setCountries] = useState('');
  const [industries, setIndustries] = useState('');
  const [businessFunctions, setBusinessFunctions] = useState('');
  const [analysisFrameworks, setAnalysisFrameworks] = useState('');
  const [responseFormat, setResponseFormat] = useState('');
  const [uploadedFile, setUploadedFile] = useState(null);
  const [aiRequirements, setAiRequirements] = useState([]);
  const [previousPrompts, setPreviousPrompts] = useState([]);
  const [showPromptsPopup, setShowPromptsPopup] = useState(false);
  const [showPopup, setShowPopup] = useState(false);
  const [googleBusinessConnected, setGoogleBusinessConnected] = useState(false);
  const [customerResearchResults, setCustomerResearchResults] = useState(null);
  const [showCustomerResearchTable, setShowCustomerResearchTable] = useState(false);
  const [minimizedCustomerResearch, setMinimizedCustomerResearch] = useState(false);
  const [isLoadingResearch, setIsLoadingResearch] = useState(false);
  const [isLoadingEmails, setIsLoadingEmails] = useState(false);
  const [showIntegrationModal, setShowIntegrationModal] = useState(false);
  const [googleBusinessForm, setGoogleBusinessForm] = useState({
    clientId: '',
    clientSecret: '',
    redirectUri: ''
  });

  // Check if user just returned from Google OAuth authorization
  useEffect(() => {
    const params = new URLSearchParams(window.location.search);
    if (params.get('google_connected') === 'true') {
      setGoogleBusinessConnected(true);
      alert('Google Business Account connected successfully!');
      // Clean up URL
      window.history.replaceState({}, document.title, window.location.pathname);
    }
    
    // Fetch pre-configured Google credentials from .env
    const fetchCredentials = async () => {
      try {
        const response = await fetch('http://127.0.0.1:5000/get-google-credentials');
        const data = await response.json();
        
        if (data.success && data.credentials) {
          setGoogleBusinessForm({
            clientId: data.credentials.clientId || '',
            clientSecret: data.credentials.clientSecret || '',
            redirectUri: data.credentials.redirectUri || ''
          });
          
          // If credentials are configured in .env, mark as connected
          if (data.credentials.hasCredentials) {
            setGoogleBusinessConnected(true);
          }
        }
      } catch (error) {
        console.error('Error fetching credentials:', error);
      }
    };
    
    fetchCredentials();
  }, []);

  const handleFileUpload = (e) => {
    const file = e.target.files[0];
    setUploadedFile(file);
    console.log('Uploaded file:', file);
  };

  const handleGenerateAnalysisFrameworks = (e) => {
    setAnalysisFrameworks(e.target.value);
  };

  const handleGenerate = async () => {
    try {
      // Check if Customer Research format is selected
      if (responseFormat === 'Customer Research') {
        // Check if Google Business is connected
        if (!googleBusinessConnected) {
          alert('Google Business Account is not connected. Please connect first to perform customer research.');
          return;
        }

        // Validate required inputs for customer research
        if (!overview || !industries || !countries) {
          alert('Please fill in Overview, Industries, and Region/Countries for customer research');
          return;
        }

        setIsLoadingResearch(true);

        // Call the search-google-businesses API
        const searchResponse = await fetch('http://127.0.0.1:5000/search-google-businesses', {
          method: 'POST',
          headers: {
            'Content-Type': 'application/json',
          },
          body: JSON.stringify({
            query: overview, // Use overview as the search query
            location: countries, // Use countries as location
            limit: 15 // Get 15 matching businesses
          }),
        });

        if (!searchResponse.ok) {
          const errorData = await searchResponse.json();
          const errorMessage = errorData.error || 'Failed to fetch customer research data';
          console.error('Search API error:', errorData);
          throw new Error(errorMessage);
        }

        const searchData = await searchResponse.json();

        if (!searchData.success) {
          alert(`Error: ${searchData.error}`);
          setIsLoadingResearch(false);
          return;
        }

        // Store results and show table
        setCustomerResearchResults({
          query: overview,
          location: countries,
          industry: industries,
          context: context,
          businesses: searchData.businesses || [],
          totalResults: searchData.totalResults || 0
        });

        setShowCustomerResearchTable(true);
        setIsLoadingResearch(false);
        return;
      }

      // Original behavior for other response formats
      const googleData = await fetchGoogleBusinessData();

      const payload = {
        overview,
        context,
        countries,
        industries,
        businessFunctions,
        analysisFrameworks,
        responseFormat,
        googleBusinessData: googleData,
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
      alert('Error: ' + error.message);
      setIsLoadingResearch(false);
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

  const handleGetEmails = async () => {
    if (!customerResearchResults || !customerResearchResults.businesses || customerResearchResults.businesses.length === 0) {
      alert('No businesses to enrich with emails');
      return;
    }

    setIsLoadingEmails(true);

    try {
      const response = await fetch('http://127.0.0.1:5000/enrich-businesses-with-emails', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          businesses: customerResearchResults.businesses
        }),
      });

      if (!response.ok) {
        const errorData = await response.json();
        throw new Error(errorData.error || 'Failed to enrich businesses with emails');
      }

      const enrichedData = await response.json();

      if (enrichedData.success && enrichedData.businesses) {
        // Update the customer research results with enriched businesses
        setCustomerResearchResults({
          ...customerResearchResults,
          businesses: enrichedData.businesses
        });
        alert(`Successfully enriched ${enrichedData.enrichedCount} businesses with email data!`);
      } else {
        alert('Failed to enrich businesses with emails');
      }
    } catch (error) {
      console.error('Error getting emails:', error);
      alert(`Error: ${error.message}`);
    } finally {
      setIsLoadingEmails(false);
    }
  };

  const handleCopyToClipboard = () => {
    if (!customerResearchResults || !customerResearchResults.businesses || customerResearchResults.businesses.length === 0) {
      alert('No data to copy.');
      return;
    }

    try {
      // Create tab-separated values format for easy pasting into Excel
      const headers = ['Business Name', 'Address', 'Phone', 'Website', 'Email', 'Match Accuracy', 'Primary'];
      const rows = customerResearchResults.businesses.map(business => [
        business.name || 'N/A',
        business.address || 'N/A',
        (business.phone || 'N/A').replace(/^\+/, ''),
        business.website || 'N/A',
        business.email || 'N/A',
        business.matchAccuracy || 'N/A',
        business.isPrimary ? 'Yes' : 'No'
      ]);

      // Create TSV (tab-separated values) content
      const tsvContent = [
        headers.join('\t'),
        ...rows.map(row => row.join('\t'))
      ].join('\n');

      // Copy to clipboard
      navigator.clipboard.writeText(tsvContent).then(() => {
        alert(`Successfully copied ${customerResearchResults.businesses.length} businesses to clipboard!`);
      }).catch(err => {
        console.error('Failed to copy:', err);
        alert('Failed to copy to clipboard');
      });
    } catch (error) {
      console.error('Error copying to clipboard:', error);
      alert('Failed to copy data to clipboard');
    }
  };

  const closePopup = () => {
    setShowPopup(false);
  };

  const handleGoogleBusinessInputChange = (e) => {
    const { name, value } = e.target;
    setGoogleBusinessForm(prev => ({
      ...prev,
      [name]: value
    }));
  };

  const handleGoogleBusinessConnect = async () => {
    // If credentials are from .env and empty, just authorize without showing modal
    const hasEnvCredentials = googleBusinessForm.clientId && 
                              googleBusinessForm.clientSecret && 
                              googleBusinessForm.redirectUri;
    
    if (!hasEnvCredentials) {
      alert('Please fill in all fields');
      return;
    }
    
    try {
      const response = await fetch('http://127.0.0.1:5000/connect-google-business', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify(googleBusinessForm),
      });

      const data = await response.json();
      
      if (response.ok && data.authUrl) {
        // Open Google authorization URL in new window
        setShowIntegrationModal(false);
        window.open(data.authUrl, 'google_auth', 'width=500,height=600');
        
        // After user authorizes, the app will redirect to localhost:3000?google_connected=true
        // We'll handle that with a URL parameter check in useEffect
      } else {
        alert(data.error || 'Failed to generate authorization URL');
      }
    } catch (error) {
      console.error('Error connecting Google Business:', error);
      alert('Error connecting to Google Business');
    }
  };

  const fetchGoogleBusinessData = async () => {
    try {
      const response = await fetch('http://127.0.0.1:5000/get-google-business-data', {
        method: 'GET',
        headers: {
          'Content-Type': 'application/json',
        },
      });

      if (response.ok) {
        const data = await response.json();
        return data;
      } else {
        console.log('Google Business not connected or data unavailable');
        return null;
      }
    } catch (error) {
      console.error('Error fetching Google Business data:', error);
      return null;
    }
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
              placeholder="Your business requirements: Product, Solution, Service, Business Idea/Name "
              value={overview}
              onChange={(e) => setOverview(e.target.value)}
              rows="3"
            />
          </div>

          <div className="input-row">
            <div className="input-group input-col-third">
              <label>Response Format</label>
              <select
                value={responseFormat}
                onChange={(e) => setResponseFormat(e.target.value)}
              >
                <option value="">Select a format</option>
                <option value="Customer Research">Customer Research</option>
                <option value="Industry Use Cases">Industry Use Cases</option>
                <option value="Product Requirements">Product Requirements</option>
                <option value="Competitive Research">Competitive Research</option>
              </select>
            </div>
            <div className="input-group input-col-third">
              <label>Reference File</label>
              <div className="file-upload-wrapper">
                <input 
                  type="file" 
                  id="file-input"
                  onChange={handleFileUpload}
                  style={{ display: 'none' }}
                />
                <button 
                  className="upload-button"
                  onClick={() => document.getElementById('file-input').click()}
                >
                  {uploadedFile ? `✓ ${uploadedFile.name}` : 'Upload'}
                </button>
              </div>
            </div>
            <div className="input-group input-col-third">
              <label>3rd Party Integration</label>
              <button 
                className={`google-business-button ${googleBusinessConnected ? 'connected' : ''}`}
                onClick={() => setShowIntegrationModal(true)}
              >
                {googleBusinessConnected ? (
                  <>
                    <span className="button-text-normal">Connected</span>
                    <span className="button-text-hover">Reconnect</span>
                  </>
                ) : 'Connect Google Business'}
              </button>
            </div>
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

          <div className="input-row">
            <div className="input-group input-col-half">
              <label>Region / Country of Interest</label>
              <input
                type="text"
                placeholder="Country or Region of interest"
                value={countries}
                onChange={(e) => setCountries(e.target.value)}
              />
            </div>
            <div className="input-group input-col-half">
              <label>Industry</label>
              <input
                type="text"
                placeholder="Enter Relevant industry"
                value={industries}
                onChange={(e) => setIndustries(e.target.value)}
              />
            </div>
          </div>

          <div className="input-row">
            <div className="input-group input-col-half">
              <label>Business Function</label>
              <input
                type="text"
                placeholder="Marketing, Sales, Finance, etc."
                value={businessFunctions}
                onChange={(e) => setBusinessFunctions(e.target.value)}
              />
            </div>
            <div className="input-group input-col-half">
              <label>Analysis Frameworks</label>
              <select
                value={analysisFrameworks}
                onChange={handleGenerateAnalysisFrameworks}
              >
                <option value="">Select a framework</option>
                <option value="PESTLE">PESTLE</option>
                <option value="VRIO">VRIO</option>
                <option value="3-Horizon">3-Horizon</option>
                <option value="5 Forces">5 Forces</option>
              </select>
            </div>
          </div>

          <div className="button-group">
            <button className="generate-button large-action-btn" onClick={handleGenerate}>
              Generate Requirements
            </button>
            <button className="save-button large-action-btn" onClick={handleSavePrompt}>
              Save Prompt
            </button>
            <button className="previous-prompts-button large-action-btn" onClick={handleFetchPreviousPrompts}>
              Previous Prompts
            </button>
          </div>
        </div>

        <div className="ai-assisted">
          <h2>AI-Assisted Requirements</h2>

          {/* Show Customer Research Results */}
          {customerResearchResults && (
              <div className="minimized-customer-research-box">
                <div className="research-summary-row minimized">
                  <div className="summary-badges">
                    <div className="summary-badge">
                      <span className="badge-label">Search</span>
                      <span className="badge-value">{customerResearchResults.query}</span>
                    </div>
                    <div className="summary-badge">
                      <span className="badge-label">Location</span>
                      <span className="badge-value">{customerResearchResults.location}</span>
                    </div>
                    <div className="summary-badge">
                      <span className="badge-label">Industry</span>
                      <span className="badge-value">{customerResearchResults.industry}</span>
                    </div>
                    <div className="summary-badge results-badge">
                      <span className="badge-label">Results</span>
                      <span className="badge-value">{customerResearchResults.totalResults}</span>
                    </div>
                  </div>
                  <button className="restore-popup-button" style={{marginLeft: 'auto'}} onClick={() => { setShowCustomerResearchTable(true); setMinimizedCustomerResearch(false); }}>
                    Maximize
                  </button>
                </div>
                <div className="minimized-content-scroll">
                  {customerResearchResults.businesses && customerResearchResults.businesses.length > 0 ? (
                    <div className="table-wrapper minimized-table-wrapper">
                      <table className="businesses-table">
                        <thead>
                          <tr>
                            <th>Business Name</th>
                            <th>Address</th>
                            <th>Phone</th>
                            <th>Website</th>
                            <th>Email</th>
                            <th>Match Accuracy</th>
                            <th>Primary</th>
                          </tr>
                        </thead>
                        <tbody>
                          {customerResearchResults.businesses.map((business, index) => (
                            <tr key={index}>
                              <td>{business.name || 'N/A'}</td>
                              <td>{business.address || 'N/A'}</td>
                              <td>{business.phone || 'N/A'}</td>
                              <td>
                                {business.website ? (
                                  <a href={business.website} target="_blank" rel="noopener noreferrer">
                                    Visit
                                  </a>
                                ) : (
                                  'N/A'
                                )}
                              </td>
                              <td>{business.email || 'N/A'}</td>
                              <td>{business.matchAccuracy || 'N/A'}</td>
                              <td>{business.isPrimary ? 'Yes' : 'No'}</td>
                            </tr>
                          ))}
                        </tbody>
                      </table>
                    </div>
                  ) : (
                    <div className="no-results">No businesses found matching your search criteria.</div>
                  )}
                </div>
              </div>
          )}

          {/* Show AI Requirements */}
          {aiRequirements.length > 0 && (
            <div className="requirements-list-section">
              <h3>Generated Requirements</h3>
              <ul>
                {aiRequirements.map((requirement, index) => (
                  <li key={index}>{requirement}</li>
                ))}
              </ul>
            </div>
          )}

          {aiRequirements.length === 0 && !customerResearchResults && (
            <p className="empty-message">Generate requirements to see results here...</p>
          )}

          {customerResearchResults && (
            <button className="copy-button" onClick={handleCopyToClipboard}>
              Copy to Clipboard
            </button>
          )}
        </div>
      </div>

      {/* Google Business Integration Modal */}
      {showIntegrationModal && (
        <div className="popup-overlay">
          <div className="popup-content integration-modal">
            <h3>{googleBusinessConnected ? 'Reconnect Google Business Account' : 'Connect Google Business Account'}</h3>
            <div className="integration-form">
              <div className="form-group">
                <label>Client ID</label>
                <input
                  type="text"
                  name="clientId"
                  value={googleBusinessForm.clientId}
                  onChange={handleGoogleBusinessInputChange}
                  placeholder="Enter Client ID"
                />
              </div>
              <div className="form-group">
                <label>Client Secret</label>
                <input
                  type="text"
                  name="clientSecret"
                  value={googleBusinessForm.clientSecret}
                  onChange={handleGoogleBusinessInputChange}
                  placeholder="Enter Client Secret"
                />
              </div>
              <div className="form-group">
                <label>Redirect URI</label>
                <input
                  type="text"
                  name="redirectUri"
                  value={googleBusinessForm.redirectUri}
                  onChange={handleGoogleBusinessInputChange}
                  placeholder="Enter Redirect URI"
                />
              </div>
            </div>
            <div className="modal-buttons">
              <button className="connect-submit-button" onClick={handleGoogleBusinessConnect}>
                Connect
              </button>
              <button className="close-popup-button" onClick={() => setShowIntegrationModal(false)}>
                Cancel
              </button>
            </div>
          </div>
        </div>
      )}

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

      {/* Customer Research Results Table */}

      {showCustomerResearchTable && customerResearchResults && !minimizedCustomerResearch && (
        <div className="popup-overlay">
          <div className="popup-content customer-research-table">
            <div className="research-summary-row">
              <span><strong>Search:</strong> {customerResearchResults.query}</span>
              <span><strong>Location:</strong> {customerResearchResults.location}</span>
              <span><strong>Industry:</strong> {customerResearchResults.industry}</span>
              <span><strong>Total Results:</strong> {customerResearchResults.totalResults}</span>
            </div>

            {isLoadingResearch ? (
              <div className="loading">Loading businesses...</div>
            ) : customerResearchResults.businesses && customerResearchResults.businesses.length > 0 ? (
              <div className="table-wrapper">
                <table className="businesses-table">
                  <thead>
                    <tr>
                      <th>Business Name</th>
                      <th>Address</th>
                      <th>Phone</th>
                      <th>Website</th>
                      <th>Email</th>
                      <th>Match Accuracy</th>
                      <th>Primary</th>
                    </tr>
                  </thead>
                  <tbody>
                    {customerResearchResults.businesses.map((business, index) => (
                      <tr key={index}>
                        <td>{business.name || 'N/A'}</td>
                        <td>{business.address || 'N/A'}</td>
                        <td>{business.phone || 'N/A'}</td>
                        <td>
                          {business.website ? (
                            <a href={business.website} target="_blank" rel="noopener noreferrer">
                              Visit
                            </a>
                          ) : (
                            'N/A'
                          )}
                        </td>
                        <td>{business.email || 'N/A'}</td>
                        <td>{business.matchAccuracy || 'N/A'}</td>
                        <td>{business.isPrimary ? 'Yes' : 'No'}</td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            ) : (
              <div className="no-results">No businesses found matching your search criteria.</div>
            )}

            <div className="modal-buttons">
              <button 
                className="get-emails-button" 
                onClick={handleGetEmails}
                disabled={isLoadingEmails}
              >
                {isLoadingEmails ? (
                  <>
                    <span className="spinner"></span>
                    Extracting Emails...
                  </>
                ) : 'Get Emails'}
              </button>
              <button className="minimize-popup-button" onClick={() => { setMinimizedCustomerResearch(true); setShowCustomerResearchTable(false); }}>
                Minimize
              </button>
            </div>
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