import React, { useState } from 'react';
import { Link } from 'react-router-dom';
import { PDFDocument } from 'pdf-lib';
import '../styles/Header.css';

function Header({ onProcessClick }) {
  const [uploadedFile, setUploadedFile] = useState(null);
  const [dataSource, setDataSource] = useState('');
  const [isConnected, setIsConnected] = useState(false);
  const [bulkFiles, setBulkFiles] = useState([]);
  const firstName = localStorage.getItem('firstName'); 
  const [showModal, setShowModal] = useState(false);
  const [showUserDropdown, setShowUserDropdown] = useState(false);
  const [selectedSystemTab, setSelectedSystemTab] = useState(0);
  const handleUserIconClick = () => {
    setShowUserDropdown((prev) => !prev);
  };

  const handleProfileClick = () => {
    // Implement navigation or modal for profile
    alert('Profile clicked!');
    setShowUserDropdown(false);
  };

  const handleSignOutClick = () => {
    // Implement sign out logic here
    alert('Sign out clicked!');
    setShowUserDropdown(false);
  };
  const [testResult, setTestResult] = useState('');
  const [dbRows, setDbRows] = useState([]);
  const [selectedSource, setSelectedSource] = useState('Database');
  const [isLoading, setIsLoading] = useState(false);
  const [historyData, setHistoryData] = useState(null);
  const [isLoadingHistory, setIsLoadingHistory] = useState(false);
  const [connectionDetails, setConnectionDetails] = useState({
    host: '',
    user: '',
    password: '',
    database: ''
  });

  const handleOpenModal = () => setShowModal(true);
  const handleCloseModal = () => setShowModal(false);

  const handleBulkFileChange = (e) => {
    setBulkFiles(Array.from(e.target.files));
  };

  // Handle input changes (including folderName)
  const handleInputChange = (e) => {
    setConnectionDetails({ ...connectionDetails, [e.target.name]: e.target.value });
  };

  // Update your handleLandscapeClick function in Header.js
  const handleLandscapeClick = async () => {
    setIsLoadingHistory(true);
    
    try {
      const response = await fetch('http://localhost:5000/chrome_history?user_id='+firstName);
      const result = await response.json();
      
      if (result.success) {
        // Use your existing modal instead of console.table and alert
        setHistoryData(result.data);
        showHistoryModal(result.data);
      } else {
        alert(`Error: ${result.error}`);
      }
      
    } catch (error) {
      console.error('Error:', error);
      alert('Please close Chrome completely and try again.');
    } finally {
      setIsLoadingHistory(false);
    }
  };


  const displayHistoryData = (data) => {
    showHistoryModal(data);
  };

  const showHistoryModal = (data) => {
  // Separate web tools from regular URLs (using new field name)
  const webTools = data.filter(item => item.is_tool === true);
  const regularUrls = data.filter(item => item.is_tool !== true);
  
  // Group tools by type and category
  const toolsByType = {};
  const toolsByCategory = {};
  
  webTools.forEach(tool => {
    const type = tool.tool_type || 'Unknown';
    const category = tool.category || 'Other';
    
    if (!toolsByType[type]) toolsByType[type] = [];
    if (!toolsByCategory[category]) toolsByCategory[category] = [];
    
    toolsByType[type].push(tool);
    toolsByCategory[category].push(tool);
  });
  
  // Create modal content
  const modal = document.createElement('div');
  modal.className = 'history-modal';
  modal.innerHTML = `
    <div class="history-modal-content">
      <div class="history-modal-header">
        <h2> Application Landscape </h2>
        <button class="close-modal">&times;</button>
      </div>
      <div class="history-modal-body">
        <div class="history-summary">
          ${Object.keys(toolsByType).length > 0 ? `
          <div class="tools-breakdown">
            <strong>Tool Types:</strong> 
            ${Object.entries(toolsByType).map(([type, tools]) => 
              `<span class="type-badge type-${type.toLowerCase().replace(/\s+/g, '-')}">${type}: ${tools.length}</span>`
            ).join(' ')}
          </div>
          ` : ''}
        </div>
        
        ${webTools.length > 0 ? `
        <div class="tools-section">
          <h3>Web Tools & Applications Found (${webTools.length})</h3>
          
          ${Object.keys(toolsByCategory).length > 1 ? `
          <div class="category-tabs">
            ${Object.keys(toolsByCategory).map((category, index) => `
              <button class="category-tab ${index === 0 ? 'active' : ''}" data-category="${category}">
                ${category} (${toolsByCategory[category].length})
              </button>
            `).join('')}
          </div>
          ` : ''}
          
          ${Object.entries(toolsByCategory).map(([category, tools], index) => `
            <div class="category-content ${index === 0 ? 'active' : ''}" data-category="${category}">
              <div class="tools-grid">
                ${tools.map((item, toolIndex) => `
                  <div class="tool-card">
                    <div class="tool-header">
                      <h4>${item.tool_name || item.title}</h4>
                      <div class="tool-badges">
                        <span class="tool-type-badge">${item.tool_type || 'Tool'}</span>
                        <span class="tool-category-badge">${item.category || 'Other'}</span>
                      </div>
                    </div>
                    <p class="tool-description">${item.description || 'No description available'}</p>
                    <div class="tool-url">
                      🔗 <a href="${item.url}" target="_blank" rel="noopener noreferrer">${item.url}</a>
                    </div>
                    <div class="tool-meta">
                      📅 ${item.visit_date} | 👀 ${item.visit_count} visits
                    </div>
                  </div>
                `).join('')}
              </div>
            </div>
          `).join('')}
        </div>
        ` : ''}
      </div>
    </div>
  `;
  
  document.body.appendChild(modal);
  
  // Add category tab functionality
  modal.querySelectorAll('.category-tab').forEach(tab => {
    tab.addEventListener('click', (e) => {
      const category = e.target.dataset.category;
      
      // Update active tab
      modal.querySelectorAll('.category-tab').forEach(t => t.classList.remove('active'));
      e.target.classList.add('active');
      
      // Update active content
      modal.querySelectorAll('.category-content').forEach(content => {
        content.classList.toggle('active', content.dataset.category === category);
      });
    });
  });
  
  // Close modal functionality
  modal.querySelector('.close-modal').onclick = () => {
    document.body.removeChild(modal);
  };
  
  modal.onclick = (e) => {
    if (e.target === modal) {
      document.body.removeChild(modal);
    }
  };
};

  // Bulk upload handler
  const handleBulkUpload = async () => {
    if (bulkFiles.length === 0) {
      alert('Please select files to upload.');
      return;
    }

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
    const mergedPdfFile = new File([mergedPdfBlob], 'merged.pdf');

    const formData = new FormData();
    formData.append('file', mergedPdfFile);
    if (connectionDetails.folderName) {
      formData.append('folder_name', connectionDetails.folderName);
    }
    try {
      const response = await fetch('http://localhost:5000/upload', {
        method: 'POST',
        body: formData,
      });
      const data = await response.json();
      if (response.ok) {
        alert('Files uploaded successfully!');
      } else {
        alert(data.error || 'Upload failed.');
      }
    } catch (error) {
      alert('Upload failed.');
    }
  };

  const handleTestConnection = async () => {
    // Example API call to test connection
    try {
      const response = await fetch('http://localhost:5000/test-connection', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(connectionDetails),
      });
      const data = await response.json();
      setTestResult(data.message || 'Test completed');
      setDbRows(data.rows || []);
    } catch (error) {
      setTestResult('Connection failed');
      setDbRows([]);
    }
  };

  const handleCreateConnection = () => {
    // Implement your create connection logic here
    alert('Create Connection clicked!');
  };

  const [showSystemModal, setShowSystemModal] = useState(false);
  const [systemTools, setSystemTools] = useState([]);
  const [toolsSortAsc, setToolsSortAsc] = useState(true);
  const [business, setBusiness] = useState('');
  const [role, setRole] = useState('');
  const [businessDesc, setBusinessDesc] = useState('');
  const [contextStep, setContextStep] = useState(0);
  const [contextConfirmed, setContextConfirmed] = useState(false);
  const [editingContext, setEditingContext] = useState(false);
  const [recommendedAgents, setRecommendedAgents] = useState('');

  function handleBusinessSubmit(e) {
    e.preventDefault();
    setContextStep(1);
  }
  function handleBusinessDescSubmit(e) {
    e.preventDefault();
    setContextStep(2);
  }
  function handleRoleSubmit(e) {
    e.preventDefault();
    setContextStep(3);
  }
  function handleContextConfirm() {
    setContextConfirmed(true);
  }
  function handleContextEdit() {
    setContextStep(0);
    setContextConfirmed(false);
  }

  function handleContextSave(e) {
    e.preventDefault();
    setEditingContext(false);
    // Optionally, save to API or localStorage here
  }

  const handleSystemClick = async () => {
    setShowSystemModal(true);
    try {
      const toolsRes = await fetch('http://localhost:5000/get_tools_landscape');
      const toolsResult = await toolsRes.json();
      let tools = Array.isArray(toolsResult.tools) ? toolsResult.tools : [];
      setSystemTools(tools);

      // Get business, role, and description from state (or default)
      const payload = {
        tools_landscape: tools,
        industry: business,
        role: role,
        business_description: businessDesc
      };
      const agentsRes = await fetch('http://localhost:5000/recommend_agents', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(payload)
      });
      const agentsResult = await agentsRes.json();
      setRecommendedAgents(agentsResult.recommendations || 'No recommendations found.');
    } catch (error) {
      setSystemTools([]);
      setRecommendedAgents('Error fetching recommendations.');
    }
  };
  const handleCloseSystemModal = () => {
    setShowSystemModal(false);
  };

  const roleOptionsByIndustry = {
    Finance: ['Manager', 'Analyst', 'Accountant', 'Auditor', 'Consultant', 'Other'],
    Healthcare: ['Doctor', 'Nurse', 'Administrator', 'Technician', 'Consultant', 'Other'],
    Education: ['Teacher', 'Principal', 'Administrator', 'Counselor', 'Other'],
    Technology: ['Developer', 'Product Manager', 'Designer', 'QA Engineer', 'Consultant', 'Other'],
    Retail: ['Store Manager', 'Sales Associate', 'Inventory Specialist', 'Buyer', 'Other'],
    Other: ['Manager', 'Consultant', 'Specialist', 'Other']
  };

  const getRoleOptions = () => {
    if (!business || !roleOptionsByIndustry[business]) {
      return ['Manager', 'Analyst', 'Developer', 'Consultant', 'Other'];
    }
    return roleOptionsByIndustry[business];
  };

  return (
    <>
      <header className="header">
        <Link to="/">
          <img
            src={`${process.env.PUBLIC_URL}/logo192.svg`}
            alt="Enable Logo"
            className="logo"
            style={{ height: '48px' }}
          />
        </Link>
        
        <div className="header-icons">
          <div style={{ display: 'flex', flexDirection: 'column', alignItems: 'center' }}>
            <img
              src="/assets/icons/dashboards.png"
              alt="Campaigns"
              className="icon"
              style={{ cursor: 'pointer' }}
              onClick={() => window.location.href='/campaign-dashboard'}
            />
            <span className="icon-label" style={{ fontSize: '0.95em', marginTop: '2px' }}>campaigns</span>
          </div>
          <div style={{ display: 'flex', flexDirection: 'column', alignItems: 'center' }}>
            <img
              src="/assets/icons/api.png"
              alt="Data"
              className="icon"
              style={{ cursor: 'pointer' }}
              onClick={handleOpenModal}
            />
            <span className="icon-label" style={{ fontSize: '0.95em', marginTop: '2px' }}>connection</span>
          </div>
          {/* NEW SYSTEM OPTION */}
          <div style={{ display: 'flex', flexDirection: 'column', alignItems: 'center' }}>
            <img
              src="/assets/icons/puzzle.png"
              alt="Agent System"
              className="icon"
              style={{ cursor: 'pointer' }}
              onClick={handleSystemClick}
            />
            <span className="icon-label" style={{ fontSize: '0.95em', marginTop: '2px' }}>system</span>
          </div>
          <div style={{ display: 'flex', flexDirection: 'column', alignItems: 'center' }}>
            <img 
              src="/assets/icons/layout-grid.png"  
              onClick={handleLandscapeClick}
              alt="Landscape" 
              className="icon"
              style={{ cursor: 'pointer' }}
            />
            <span className="icon-label" style={{ fontSize: '0.95em', marginTop: '2px' }}>landscape</span>
          </div>
            <div style={{ display: 'flex', flexDirection: 'column', alignItems: 'center' }}>
              <img 
                src="/assets/icons/process.png"  
                alt="Process" 
                className="icon"
                style={{ cursor: 'pointer' }}
                onClick={onProcessClick}
              />
              <span className="icon-label" style={{ fontSize: '0.95em', marginTop: '2px' }}>process</span>
            </div>
          <div className="user-icon-wrapper" style={{ display: 'flex', flexDirection: 'column', alignItems: 'center', position: 'relative' }}>
            <img src="/assets/icons/user.png" alt="User" className="icon" style={{ cursor: 'pointer' }} onClick={handleUserIconClick} />
            {firstName && (
              <span className="user-first-name" style={{ fontSize: '0.95em', marginTop: '2px' }}>
                {firstName}
              </span>
            )}
            {showUserDropdown && (
              <div className="user-dropdown-menu">
                <button className="user-dropdown-item" onClick={handleProfileClick}>Profile</button>
                <button className="user-dropdown-item" onClick={handleSignOutClick}>Sign Out</button>
              </div>
            )}
          </div>
        </div>
      </header>

      {showSystemModal && (
        <div className="history-modal">
          <div className="history-modal-content system-modal-content">
            <div className="history-modal-header">
              <h2 className="history-modal-title">System Overview</h2>
              <button className="close-modal" onClick={handleCloseSystemModal}>×</button>
            </div>
            <div className="system-tabs">
              {['Existing Software Tools', 'Business & Role Context', 'Recommended Agents'].map((tab, idx) => (
                <button
                  key={tab}
                  className={`system-tab${selectedSystemTab === idx ? ' active' : ''}`}
                  onClick={() => setSelectedSystemTab(idx)}
                >
                  {tab}
                </button>
              ))}
            </div>
            <div className="system-tab-content">
              {selectedSystemTab === 0 && (
                <div className="system-section-content">
                  {systemTools.length === 0 ? (
                    <div>No tools found.</div>
                  ) : (
                    <>
                      <div className="tools-analytics-summary">
                        <div className="tools-analytics-item">Total Tools: {systemTools.length}</div>
                        <div className="tools-analytics-item">Categories: {Array.from(new Set(systemTools.map(t => t.category))).length}</div>
                      </div>
                      <div className="tools-list-row-wrapper">
                        <div className="tools-list-row-header">
                          <span className="tools-list-row-title">Tool Name</span>
                          <span className="tools-list-row-category" style={{cursor: 'pointer'}} onClick={() => setToolsSortAsc((asc) => !asc)}>
                            Category {toolsSortAsc ? '▲' : '▼'}
                          </span>
                          <span className="tools-list-row-description">Description</span>
                        </div>
                        <div className="tools-list-row-scroll">
                          {[...systemTools]
                            .sort((a, b) => {
                              if (!a.category) return 1;
                              if (!b.category) return -1;
                              if (a.category.toLowerCase() < b.category.toLowerCase()) return toolsSortAsc ? -1 : 1;
                              if (a.category.toLowerCase() > b.category.toLowerCase()) return toolsSortAsc ? 1 : -1;
                              return 0;
                            })
                            .map((tool, idx) => (
                              <div key={idx} className="tools-list-row">
                                <span className="tools-list-row-title">{tool.tool_name}</span>
                                <span className="tools-list-row-category">{tool.category}</span>
                                <span className="tools-list-row-description">{tool.description}</span>
                              </div>
                            ))}
                        </div>
                      </div>
                    </>
                  )}
                </div>
              )}
              {selectedSystemTab === 1 && (
                <div className="system-section-content">
                  {!contextConfirmed ? (
                    <>
                      {contextStep === 0 && (
                        <form className="context-form" onSubmit={handleBusinessSubmit}>
                          <div className="context-chat-bubble bot">Hi! Let's set up your business context. What industry are you in?</div>
                          <select value={business} onChange={e => setBusiness(e.target.value)} className="context-select">
                            <option value="">Select your industry...</option>
                            <option value="Finance">Finance</option>
                            <option value="Healthcare">Healthcare</option>
                            <option value="Education">Education</option>
                            <option value="Technology">Technology</option>
                            <option value="Retail">Retail</option>
                            <option value="Other">Other</option>
                          </select>
                          <input type="text" className="context-input" placeholder="Or type your business..." value={business} onChange={e => setBusiness(e.target.value)} />
                          <button type="submit" className="context-save-btn">Next</button>
                        </form>
                      )}
                      {contextStep === 1 && (
                        <form className="context-form" onSubmit={handleBusinessDescSubmit}>
                          <div className="context-chat-bubble bot">Can you briefly describe your main product or service? This helps us recommend agents for your entire business, not just your role.</div>
                          <textarea className="context-input" placeholder="Describe your product or service..." value={businessDesc} onChange={e => setBusinessDesc(e.target.value)} rows={3} />
                          <button type="submit" className="context-save-btn">Next</button>
                        </form>
                      )}
                      {contextStep === 2 && (
                        <form className="context-form" onSubmit={handleRoleSubmit}>
                          <div className="context-chat-bubble bot">Great! And what is your role?</div>
                          <select value={role} onChange={e => setRole(e.target.value)} className="context-select">
                            <option value="">Select your role...</option>
                            {getRoleOptions().map(opt => (
                              <option key={opt} value={opt}>{opt}</option>
                            ))}
                          </select>
                          <input type="text" className="context-input" placeholder="Or type your role..." value={role} onChange={e => setRole(e.target.value)} />
                          <button type="submit" className="context-save-btn">Next</button>
                        </form>
                      )}
                      {contextStep === 3 && (
                        <div className="context-chat-bubble bot">Awesome! You're a <strong>{role || '...'}</strong> in <strong>{business || '...'}</strong>.<br />Your business offers: <strong>{businessDesc || '...'}</strong><br />Is this correct?</div>
                      )}
                      {contextStep === 3 && (
                        <div style={{marginTop: '18px'}}>
                          <button className="context-save-btn" onClick={handleContextConfirm}>Yes, Confirm</button>
                          <button className="context-edit-btn" style={{marginLeft: '12px'}} onClick={handleContextEdit}>Edit</button>
                        </div>
                      )}
                    </>
                  ) : (
                    <div className="context-summary">
                      <div className="context-chat-bubble bot">✅ Your business context is set!</div>
                      <div><span className="icon-briefcase" /> Business: <strong>{business}</strong></div>
                      <div><span className="icon-user" /> Role: <strong>{role}</strong></div>
                      <div><span className="icon-briefcase" /> Product/Service: <strong>{businessDesc}</strong></div>
                      <button className="context-edit-btn" onClick={handleContextEdit}>Edit</button>
                    </div>
                  )}
                </div>
              )}
              {selectedSystemTab === 2 && (
                <div className="system-section-content">
                  <div className="agents-recommendations">
                    {recommendedAgents ? (
                      <pre style={{whiteSpace: 'pre-wrap', fontSize: '1.05em', color: '#334155', background: '#f8fafc', borderRadius: '8px', padding: '18px', border: '1px solid #e2e8f0'}}>{recommendedAgents}</pre>
                    ) : (
                      <div>Loading recommendations...</div>
                    )}
                  </div>
                </div>
              )}
            </div>
          </div>
        </div>
      )}

      {showModal && (
        <div className="modal-overlay">
          <div className="modal-content">
            <div className="modal-source-select">
              <button
                className={`source-btn${selectedSource === 'Database' ? ' selected' : ''}`}
                title="Database"
                onClick={() => setSelectedSource('Database')}
              >
                <span role="img" aria-label="database"><img src="/assets/icons/database.png" alt="Database" /></span>
                <span>Database</span>
              </button>
              <button
                className={`source-btn${selectedSource === 'File' ? ' selected' : ''}`}
                title="File"
                onClick={() => setSelectedSource('File')}
              >
                <span role="img" aria-label="file"><img src="/assets/icons/documents.png" alt="File" /></span>
                <span>File</span>
              </button>
              <button
                className={`source-btn${selectedSource === 'API' ? ' selected' : ''}`}
                title="API"
                onClick={() => setSelectedSource('API')}
              >
                <span role="img" aria-label="api"><img src="/assets/icons/api.png" alt="API" /></span>
                <span>API</span>
              </button>
            </div>
            <button className="modal-close" onClick={handleCloseModal}>×</button>
            <h2>Connect to Data Source</h2>
            <div className="modal-form">
            {selectedSource === 'Database' && (
              <>
                <input
                  type="text"
                  name="host"
                  placeholder="Host"
                  value={connectionDetails.host}
                  onChange={handleInputChange}
                />
                <input
                  type="text"
                  name="user"
                  placeholder="User"
                  value={connectionDetails.user}
                  onChange={handleInputChange}
                />
                <input
                  type="password"
                  name="password"
                  placeholder="Password"
                  value={connectionDetails.password}
                  onChange={handleInputChange}
                />
                <input
                  type="text"
                  name="database"
                  placeholder="Database"
                  value={connectionDetails.database}
                  onChange={handleInputChange}
                />
                <div style={{ display: 'flex', gap: '12px', margin: '16px 0' }}>
                  <button onClick={handleCreateConnection}>Create Connection</button>
                  <button onClick={handleTestConnection}>Test Connection</button>
                </div>
                <textarea
                  rows={3}
                  value={testResult}
                  readOnly
                  placeholder="Test results will appear here"
                  style={{ width: '100%', marginBottom: '12px' }}
                />
                {dbRows.length > 0 && (
                  <div className="db-preview">
                    <h4>Sample Rows:</h4>
                    <table>
                      <thead>
                        <tr>
                          {Object.keys(dbRows[0]).map((col, idx) => (
                            <th key={idx}>{col}</th>
                          ))}
                        </tr>
                      </thead>
                      <tbody>
                        {dbRows.map((row, idx) => (
                          <tr key={idx}>
                            {Object.values(row).map((val, i) => (
                              <td key={i}>{val}</td>
                            ))}
                          </tr>
                        ))}
                      </tbody>
                    </table>
                  </div>
                )}
              </>
            )}

            {selectedSource === 'File' && (
              <>
                <input
                  type="text"
                  name="fileLocation"
                  placeholder="File Location"
                  value={connectionDetails.fileLocation || ''}
                  onChange={handleInputChange}
                />
                <input
                  type="text"
                  name="fileDescription"
                  placeholder="File Description"
                  value={connectionDetails.fileDescription || ''}
                  onChange={handleInputChange}
                />
                <div style={{ display: 'flex', gap: '12px', margin: '16px 0' }}>
                  <button onClick={() => alert('Upload logic here')}>Create Connection</button>
                </div>
                <input
                  type="text"
                  name="folderName"
                  placeholder="Folder Name"
                  value={connectionDetails.folderName || ''}
                  onChange={handleInputChange}
                />
                <input
                  type="file"
                  name="file"
                  multiple
                  onChange={handleBulkFileChange}
                />
                <div style={{ display: 'flex', gap: '12px', margin: '16px 0' }}>
                  <button onClick={handleBulkUpload}>Upload Files</button>
                </div>
              </>
            )}

            {selectedSource === 'API' && (
              <>
                <input
                  type="text"
                  name="apiUrl"
                  placeholder="API URL"
                  value={connectionDetails.apiUrl || ''}
                  onChange={handleInputChange}
                />
                <textarea
                  name="apiParams"
                  placeholder="Enter JSON query parameters"
                  value={connectionDetails.apiParams || ''}
                  onChange={handleInputChange}
                  rows={4}
                  style={{ width: '100%', marginBottom: '12px' }}
                />
                <div style={{ display: 'flex', gap: '12px', margin: '16px 0' }}>
                  <button onClick={() => alert('Create API Connection logic here')}>Create Connection</button>
                  <button onClick={() => alert('Test API Connection logic here')}>Test Connection</button>
                </div>
                <textarea
                  rows={3}
                  value={testResult}
                  readOnly
                  placeholder="Test results will appear here"
                  style={{ width: '100%', marginBottom: '12px' }}
                />
              </>
            )}
          </div>
          </div>
        </div>
      )}
    </>
  );
}

export default Header;