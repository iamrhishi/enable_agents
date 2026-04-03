import React, { useState, useEffect } from 'react';
import * as XLSX from 'xlsx';
import { jsPDF } from 'jspdf';
import autoTable from 'jspdf-autotable';
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
  const [extractingEmailRows, setExtractingEmailRows] = useState({});
  const [showIntegrationModal, setShowIntegrationModal] = useState(false);
  const [showExportModal, setShowExportModal] = useState(false);
  const [extractionUsage, setExtractionUsage] = useState(null);
  const [showEmailModal, setShowEmailModal] = useState(false);
  const [campaignName, setCampaignName] = useState('');
  const [emailSubject, setEmailSubject] = useState('');
  const [emailBody, setEmailBody] = useState('');
  const [isSendingEmails, setIsSendingEmails] = useState(false);
  const [googleBusinessForm, setGoogleBusinessForm] = useState({
    clientId: '',
    clientSecret: '',
    redirectUri: ''
  });

  const getCurrentUsername = () => {
    return localStorage.getItem('username') || localStorage.getItem('firstName') || 'anonymous';
  };

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

    const fetchEmailUsage = async () => {
      try {
        const username = getCurrentUsername();
        const response = await fetch(`http://127.0.0.1:5000/email-extraction-usage?username=${encodeURIComponent(username)}`);
        const data = await response.json();
        if (response.ok && data.success && data.usageSummary) {
          setExtractionUsage(data.usageSummary);
        }
      } catch (error) {
        console.error('Error fetching email extraction usage:', error);
      }
    };

    fetchEmailUsage();
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
        // Allow customer research even when OAuth is not connected.
        // Backend can run this flow via Google Places API key.

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
            page_size: 200 // Get 200 matching businesses
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
          businesses: customerResearchResults.businesses,
          username: getCurrentUsername()
        }),
      });

      if (!response.ok) {
        const errorData = await response.json();
        throw new Error(errorData.error || 'Failed to enrich businesses with emails');
      }

      const enrichedData = await response.json();
      if (enrichedData.usageSummary) {
        setExtractionUsage(enrichedData.usageSummary);
      }

      if (enrichedData.success && enrichedData.businesses) {
        // Update the customer research results with enriched businesses
        setCustomerResearchResults({
          ...customerResearchResults,
          businesses: enrichedData.businesses
        });
        const usageLine = enrichedData.usageSummary
          ? `\nUsed: ${enrichedData.usageSummary.usedCount}/${enrichedData.usageSummary.totalAllowed} | Remaining: ${enrichedData.usageSummary.remainingCount}`
          : '';
        alert(`Successfully enriched ${enrichedData.enrichedCount} businesses with email data!${usageLine}`);
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

  const handleExtractEmailForBusiness = async (business, index) => {
    if (!business || !business.website) {
      alert('Website not available for this business.');
      return;
    }

    setExtractingEmailRows((prev) => ({ ...prev, [index]: true }));

    try {
      const response = await fetch('http://127.0.0.1:5000/enrich-businesses-with-emails', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          businesses: [business],
          username: getCurrentUsername()
        }),
      });

      if (!response.ok) {
        const errorData = await response.json();
        throw new Error(errorData.error || 'Failed to extract email for this business');
      }

      const enrichedData = await response.json();
      if (enrichedData.usageSummary) {
        setExtractionUsage(enrichedData.usageSummary);
      }
      const enrichedBusiness = enrichedData?.businesses?.[0];

      if (!enrichedBusiness) {
        throw new Error('No enriched business data returned');
      }

      setCustomerResearchResults((prev) => {
        if (!prev || !prev.businesses) {
          return prev;
        }

        const updatedBusinesses = [...prev.businesses];
        updatedBusinesses[index] = {
          ...updatedBusinesses[index],
          email: enrichedBusiness.email || 'N/A'
        };

        return {
          ...prev,
          businesses: updatedBusinesses
        };
      });
    } catch (error) {
      console.error('Error extracting email for business:', error);
      alert(`Error: ${error.message}`);
    } finally {
      setExtractingEmailRows((prev) => ({ ...prev, [index]: false }));
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

  const getCustomerResearchRows = () => {
    if (!customerResearchResults || !customerResearchResults.businesses || customerResearchResults.businesses.length === 0) {
      return [];
    }

    return customerResearchResults.businesses.map((business) => ({
      businessName: business.name || 'N/A',
      address: business.address || 'N/A',
      phone: business.phone || 'N/A',
      website: business.website || 'N/A',
      email: business.email || 'N/A',
      matchAccuracy: business.matchAccuracy || 'N/A',
      primary: business.isPrimary ? 'Yes' : 'No'
    }));
  };

  const downloadTextFile = (filename, content, mimeType) => {
    const blob = new Blob([content], { type: mimeType });
    const url = window.URL.createObjectURL(blob);
    const anchor = document.createElement('a');
    anchor.href = url;
    anchor.download = filename;
    document.body.appendChild(anchor);
    anchor.click();
    document.body.removeChild(anchor);
    window.URL.revokeObjectURL(url);
  };

  const escapeCsvValue = (value) => {
    const safeValue = String(value ?? '');
    if (safeValue.includes('"') || safeValue.includes(',') || safeValue.includes('\n')) {
      return `"${safeValue.replace(/"/g, '""')}"`;
    }
    return safeValue;
  };

  const buildCsvContent = (rows) => {
    const headers = ['Business Name', 'Address', 'Phone', 'Website', 'Email', 'Match Accuracy', 'Primary'];
    const csvRows = rows.map((row) => [
      row.businessName,
      row.address,
      row.phone,
      row.website,
      row.email,
      row.matchAccuracy,
      row.primary
    ].map(escapeCsvValue).join(','));

    return [headers.join(','), ...csvRows].join('\n');
  };

  const handleExport = async (format) => {
    const rows = getCustomerResearchRows();
    if (rows.length === 0) {
      alert('No data available to export.');
      return;
    }

    const fileBaseName = `market_research_${new Date().toISOString().slice(0, 10)}`;
    const csvContent = buildCsvContent(rows);

    try {
      if (format === 'excel') {
        const worksheet = XLSX.utils.json_to_sheet(rows);
        const workbook = XLSX.utils.book_new();
        XLSX.utils.book_append_sheet(workbook, worksheet, 'Market Research');
        XLSX.writeFile(workbook, `${fileBaseName}.xlsx`);
      }

      if (format === 'csv') {
        downloadTextFile(`${fileBaseName}.csv`, csvContent, 'text/csv;charset=utf-8;');
      }

      if (format === 'json') {
        downloadTextFile(`${fileBaseName}.json`, JSON.stringify(rows, null, 2), 'application/json;charset=utf-8;');
      }

      if (format === 'pdf') {
        const doc = new jsPDF({ orientation: 'landscape', unit: 'pt', format: 'a4' });
        doc.setFontSize(12);
        doc.text('Market Research Export', 40, 36);

        autoTable(doc, {
          startY: 50,
          head: [['Business Name', 'Address', 'Phone', 'Website', 'Email', 'Match Accuracy', 'Primary']],
          body: rows.map((row) => [
            row.businessName,
            row.address,
            row.phone,
            row.website,
            row.email,
            row.matchAccuracy,
            row.primary
          ]),
          styles: { fontSize: 8, cellPadding: 4 },
          headStyles: { fillColor: [30, 58, 95] }
        });

        doc.save(`${fileBaseName}.pdf`);
      }

      if (format === 'sheets') {
        const headers = ['Business Name', 'Address', 'Phone', 'Website', 'Email', 'Match Accuracy', 'Primary'];
        const matrixRows = rows.map((row) => [
          row.businessName,
          row.address,
          row.phone,
          row.website,
          row.email,
          row.matchAccuracy,
          row.primary
        ]);

        const toSheetsSafeValue = (value) => {
          const strValue = String(value ?? '');
          return /^[=+\-@]/.test(strValue) ? `'${strValue}` : strValue;
        };

        const tsvContent = [
          headers.join('\t'),
          ...matrixRows.map((row) => row.map(toSheetsSafeValue).join('\t'))
        ].join('\n');

        await navigator.clipboard.writeText(tsvContent);
        window.open('https://docs.google.com/spreadsheets/create', '_blank', 'noopener,noreferrer');
        alert('Google Sheets opened. Data is copied to clipboard, paste with Ctrl+V.');
      }
    } catch (error) {
      console.error('Export failed:', error);
      alert('Export failed. Please try again.');
    } finally {
      setShowExportModal(false);
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

    const handleSendEmails = async () => {
    if (!emailSubject || !emailBody) {
      alert("Subject and Body are required");
      return;
    }
    const validEmails = customerResearchResults?.businesses?.filter(b => b.email && b.email !== 'N/A' && b.email.includes('@')) || [];
    if (validEmails.length === 0) {
      alert("No valid emails found to send to");
      return;
    }

    setIsSendingEmails(true);
    try {
      const response = await fetch('http://127.0.0.1:5000/send-bulk-emails', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          campaignName: campaignName,
          subject: emailSubject,
          body: emailBody,
          businesses: validEmails
        })
      });
      const data = await response.json();
      if (response.ok && data.success) {
        alert('Successfully sent ' + validEmails.length + ' emails!');
        setShowEmailModal(false);
        setEmailSubject('');
        setEmailBody('');
      } else {
        alert('Error : ' + data.error);
      }
    } catch (e) {
      alert('Error sending emails: ' + e.message);
    } finally {
      setIsSendingEmails(false);
    }
  };

    return (
    <div className="requirements-page">
      <Header />
      <div className="requirements-container">
        
        <div className="requirements-header-bar">
          <div className="header-bar-top">
            <div className="overview-title">
              <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" style={{marginRight: '8px', color: '#1E3A5F'}}>
                <path d="M14 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V8z"></path>
                <polyline points="14 2 14 8 20 8"></polyline>
                <line x1="16" y1="13" x2="8" y2="13"></line>
                <line x1="16" y1="17" x2="8" y2="17"></line>
                <polyline points="10 9 9 9 8 9"></polyline>
              </svg>
              <span>Requirement Overview</span>
            </div>
            <div className="integration-badge">
              <span className="dot"></span> 3RD PARTY INTEGRATION READY
            </div>
          </div>
          <div className="header-bar-inputs">
            <div className="input-block flex-grow">
              <label>PROJECT CONTEXT & DESCRIPTION</label>
              <input
                type="text"
                placeholder="Enter goals..."
                value={overview}
                onChange={(e) => setOverview(e.target.value)}
              />
            </div>
            <div className="input-block">
              <label>INDUSTRY</label>
              <input
                type="text"
                placeholder="e.g. Fintech"
                value={industries}
                onChange={(e) => setIndustries(e.target.value)}
              />
            </div>
            <div className="input-block">
              <label>REGION</label>
              <input
                type="text"
                placeholder="e.g. North Ame"
                value={countries}
                onChange={(e) => setCountries(e.target.value)}
              />
            </div>
            <div className="input-block">
              <label>FORMAT</label>
              <select
                value={responseFormat}
                onChange={(e) => setResponseFormat(e.target.value)}
              >
                 <option value="Detailed PRD">Detailed PRD</option>
                 <option value="Customer Research">Customer Research</option>
                 <option value="Industry Use Cases">Industry Use Cases</option>
                 <option value="Product Requirements">Product Requirements</option>
                 <option value="Competitive Research">Competitive Research</option>
              </select>
            </div>
            <div className="input-block button-block">
              <input type="file" id="file-input" onChange={handleFileUpload} style={{ display: 'none' }} />
              <button className="upload-btn" onClick={() => document.getElementById('file-input').click()}>
                <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" style={{marginRight: '6px'}}>
                  <path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4"></path>
                  <polyline points="17 8 12 3 7 8"></polyline>
                  <line x1="12" y1="3" x2="12" y2="15"></line>
                </svg>
                {uploadedFile ? uploadedFile.name : 'Upload Ref'}
              </button>
            </div>
            <div className="input-block button-block">
              <button className="generate-req-btn" onClick={handleGenerate}>
                Generate Requirements
              </button>
            </div>
          </div>
        </div>

        <div className="main-workspace-area">

          <div className="tabs-container">
            <button className="workspace-tab active-tab">Leads</button>
            <button className="workspace-tab" onClick={() => window.location.href='/campaign-dashboard'}>Campaign Dashboard</button>
          </div>

          <div className="workspace-content-box">
            <div className="ai-assisted" style={{ background: 'transparent', boxShadow: 'none' }}>
              {(!aiRequirements && !customerResearchResults) ? (
                <div style={{ display: 'flex', flexDirection: 'column', alignItems: 'center', justifyContent: 'center', height: '100%', minHeight: '200px' }}>
                  <div style={{ background: '#EAE1D9', borderRadius: '12px', width: '64px', height: '64px', display: 'flex', alignItems: 'center', justifyContent: 'center', marginBottom: '12px' }}>
                    <svg width="32" height="32" viewBox="0 0 24 24" fill="none" stroke="#8E9BAb" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
                      <path d="M12 2l3.09 6.26L22 9.27l-5 4.87 1.18 6.88L12 17.77l-6.18 3.25L7 14.14 2 9.27l6.91-1.01L12 2z"></path>
                    </svg>
                  </div>
                  <h2 style={{ color: '#0D2644', fontSize: '1.5rem', marginBottom: '12px' }}>Awaiting Configuration</h2>
                  <p style={{ color: '#6C7F99', textAlign: 'center', maxWidth: '400px', fontSize: '0.95rem', lineHeight: '1.5', marginBottom: '16px' }}>
                    Refine the requirements in the bar above to generate structured architectural specifications. Our AI will analyze your context, industry, and region to produce a precise specification.
                  </p>
                  <div style={{ display: 'flex', gap: '16px' }}>
                    <div style={{ background: 'white', padding: '8px 16px', borderRadius: '4px', fontSize: '0.75rem', fontWeight: 'bold', color: '#1E3A5F', boxShadow: '0 2px 4px rgba(0,0,0,0.05)' }}>READY TO ANALYZE</div>
                    <div style={{ background: 'white', padding: '8px 16px', borderRadius: '4px', fontSize: '0.75rem', fontWeight: 'bold', color: '#1E3A5F', boxShadow: '0 2px 4px rgba(0,0,0,0.05)' }}>SECURE END-TO-END</div>
                    <div style={{ background: 'white', padding: '8px 16px', borderRadius: '4px', fontSize: '0.75rem', fontWeight: 'bold', color: '#1E3A5F', boxShadow: '0 2px 4px rgba(0,0,0,0.05)' }}>ENTERPRISE LLM</div>
                  </div>
                </div>
              ) : (
                <>



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

                  <div style={{ marginLeft: 'auto', display: 'flex', flexDirection: 'column', alignItems: 'flex-end', gap: '8px' }}>
                    <div style={{ display: 'flex', flexDirection: 'column', alignItems: 'center', marginRight: '4px', background: '#fff', padding: '6px 12px', borderRadius: '8px', boxShadow: '0 1px 3px rgba(0,0,0,0.05)' }}>
                      <span style={{ fontSize: '10px', fontWeight: 'bold', color: '#6b7280', letterSpacing: '0.05em', marginBottom: '4px' }}>EMAILS EXTRACTED</span>
                      <div style={{ position: 'relative', width: '46px', height: '46px' }}>
                        <svg fill="none" viewBox="0 0 50 50" style={{ transform: 'rotate(-90deg)' }}>
                          <circle cx="25" cy="25" r="21" stroke="#333" strokeWidth="5" />
                          <circle 
                            cx="25" cy="25" r="21" 
                            stroke="#10B981" 
                            strokeWidth="5" 
                            strokeDasharray={2 * Math.PI * 21} 
                            strokeDashoffset={(2 * Math.PI * 21) - ((customerResearchResults.businesses ? customerResearchResults.businesses.filter(b => b.email && b.email !== 'N/A').length : 0) / 100) * (2 * Math.PI * 21)} 
                            strokeLinecap="round" 
                          />
                        </svg>
                        <div style={{ position: 'absolute', top: 0, left: 0, width: '100%', height: '100%', display: 'flex', flexDirection: 'column', alignItems: 'center', justifyContent: 'center', color: '#333' }}>
                          <span style={{ fontSize: '13px', fontWeight: '800', borderBottom: '1px solid #ccc', lineHeight: '1.1', width: '50%', textAlign: 'center', paddingBottom: '1px', marginBottom: '1px' }}>
                            {customerResearchResults.businesses ? customerResearchResults.businesses.filter(b => b.email && b.email !== 'N/A').length : 0}
                          </span>
                          <span style={{ fontSize: '11px', fontWeight: '800', lineHeight: '1' }}>100</span>
                        </div>
                      </div>
                    </div>

                    <div style={{ display: 'flex', gap: '8px' }}>
                      <button 
                        className="get-emails-button compact"
                        onClick={handleGetEmails}
                        disabled={isLoadingEmails}
                      >
                        {isLoadingEmails ? (
                          <>
                            <span className="spinner"></span>
                            Extracting...
                          </>
                        ) : 'Get All Emails'}
                      </button>
                      <button className="restore-popup-button" onClick={() => { setShowCustomerResearchTable(true); setMinimizedCustomerResearch(false); }}>
                        Maximize
                      </button>
                    </div>
                  </div>
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
                              <td>
                                {business.email && business.email !== 'N/A' ? (
                                  business.email
                                ) : (
                                  <button
                                    className="extract-email-button"
                                    onClick={() => handleExtractEmailForBusiness(business, index)}
                                    disabled={!!extractingEmailRows[index]}
                                  >
                                    {extractingEmailRows[index] ? 'Extracting...' : 'Extract Email'}
                                  </button>
                                )}
                              </td>
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
            <div className="research-actions-row">
              <button className="copy-button compact" onClick={handleCopyToClipboard}>
                Copy to Clipboard
              </button>
              <button className="export-button compact" onClick={() => setShowExportModal(true)}>
                  Export Data
                </button>
                <button className="send-emails-button compact" onClick={() => setShowEmailModal(true)} style={{ backgroundColor: '#10B981', color: '#fff', border: 'none', marginLeft: '10px' }}>
                  Send Emails
                </button>
                </div>
          )}
                </>
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
                        <td>
                          {business.email && business.email !== 'N/A' ? (
                            business.email
                          ) : (
                            <button
                              className="extract-email-button"
                              onClick={() => handleExtractEmailForBusiness(business, index)}
                              disabled={!!extractingEmailRows[index]}
                            >
                              {extractingEmailRows[index] ? 'Extracting...' : 'Extract Email'}
                            </button>
                          )}
                        </td>
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

      {showExportModal && (
        <div className="popup-overlay">
          <div className="popup-content export-options-modal">
            <h3>Export Market Research</h3>
            <div className="export-options-grid">
              <button onClick={() => handleExport('excel')}>Download Excel (.xlsx)</button>
              <button onClick={() => handleExport('csv')}>Download CSV (.csv)</button>
              <button onClick={() => handleExport('pdf')}>Download PDF (.pdf)</button>
              <button onClick={() => handleExport('json')}>Download JSON (.json)</button>
              <button onClick={() => handleExport('sheets')}>Open in Google Sheets</button>
            </div>
            <div className="modal-buttons">
              <button className="close-popup-button" onClick={() => setShowExportModal(false)}>
                Close
              </button>
            </div>
          </div>
        </div>
      )}


      

      

        {showEmailModal && (
        <div className="popup-overlay">
          <div className="popup-content email-modal" style={{ maxWidth: '600px', width: '90%' }}>
            <h3>Draft Email Campaign</h3>
            <div className="input-group" style={{ marginBottom: '15px' }}>
              <label style={{ display: 'block', marginBottom: '5px', fontWeight: 'bold' }}>Campaign Name</label>
              <input 
                type="text" 
                value={campaignName} 
                onChange={(e) => setCampaignName(e.target.value)} 
                placeholder="e.g., Tech Startups Dec 2026"
                style={{ width: '100%', padding: '10px', border: '1px solid #ccc', borderRadius: '4px' }}
              />
            </div>
            <div className="input-group" style={{ marginBottom: '15px' }}>
              <label style={{ display: 'block', marginBottom: '5px', fontWeight: 'bold' }}>Subject</label>
              <input 
                type="text" 
                value={emailSubject} 
                onChange={(e) => setEmailSubject(e.target.value)} 
                placeholder="Email Subject"
                style={{ width: '100%', padding: '10px', border: '1px solid #ccc', borderRadius: '4px' }}
              />
            </div>
            <div className="input-group" style={{ marginBottom: '15px' }}>
              <label style={{ display: 'block', marginBottom: '5px', fontWeight: 'bold' }}>Body</label>
              <textarea 
                value={emailBody} 
                onChange={(e) => setEmailBody(e.target.value)} 
                placeholder="Type your email body here...\n\nYou can use {{Company}} to automatically insert the business's name." 
                rows={8}
                style={{ width: '100%', padding: '10px', border: '1px solid #ccc', borderRadius: '4px', resize: 'vertical', fontFamily: 'inherit' }}
              ></textarea>
            </div>
            <div className="modal-buttons" style={{ display: 'flex', gap: '10px', justifyContent: 'flex-end', marginTop: '15px' }}>
              <button 
                className="cancel-button" 
                onClick={() => setShowEmailModal(false)}
                style={{ backgroundColor: '#f3f4f6', color: '#374151', border: '1px solid #d1d5db', padding: '8px 16px', borderRadius: '4px', cursor: 'pointer' }}
              >
                Cancel
              </button>
              <button 
                className="send-button" 
                onClick={handleSendEmails} 
                disabled={isSendingEmails}
                style={{ backgroundColor: '#10B981', color: 'white', border: 'none', padding: '8px 16px', borderRadius: '4px', cursor: isSendingEmails ? 'not-allowed' : 'pointer', opacity: isSendingEmails ? 0.7 : 1 }}
              >
                {isSendingEmails ? 'Sending...' : 'Send'}
              </button>
            </div>
          </div>
        </div>
      )}

          </div>
          
        </div>
      </div>
  );
}

export default RequirementsGathering;
