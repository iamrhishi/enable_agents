import React, { useState, useRef, useEffect } from 'react';
import ReactMarkdown from 'react-markdown';
import Header from '../core/Header';
import '../styles/SalesHelperAgent.css';
import { API_CONFIG } from '../config/apiConfig';

function SalesHelperAgent() {
  const [inputMessage, setInputMessage] = useState('');
  const [isLoading, setIsLoading] = useState(false);
  const [csvData, setCsvData] = useState(null);
  const [userProfile, setUserProfile] = useState(null);
  const [isAnalyzing, setIsAnalyzing] = useState(false);
  const [savedProjects, setSavedProjects] = useState([]);
  const [selectedSavedProject, setSelectedSavedProject] = useState(null);
  const [selectedSavedProjectLeads, setSelectedSavedProjectLeads] = useState([]);
  const [savedProjectSelection, setSavedProjectSelection] = useState('');
  const [isLoadingSavedProjects, setIsLoadingSavedProjects] = useState(false);
  const [isLoadingSavedProjectLeads, setIsLoadingSavedProjectLeads] = useState(false);
  const messagesEndRef = useRef(null);
  const csvFileRef = useRef(null);
  const cvFileRef = useRef(null);
  const [existingFiles, setExistingFiles] = useState(new Map());
  
  const [messages, setMessages] = useState([
    {
      id: 1,
      text: "Welcome to the Sales Helper Agent! I can help you analyze prospects, track leads, and optimize your sales pipeline!",
      sender: 'agent',
      timestamp: new Date().toLocaleTimeString(),
      format: 'markdown'
    }
  ]);

  // Add these state variables at the top with other state
  const [currentUserId, setCurrentUserId] = useState('user_001'); // Replace with actual user ID
  const [userFavorites, setUserFavorites] = useState([]);

  const getCurrentUsername = () => {
    return localStorage.getItem('userEmail') || localStorage.getItem('username') || localStorage.getItem('firstName') || currentUserId || 'anonymous';
  };

  const scrollToBottom = () => {
    messagesEndRef.current?.scrollIntoView({ behavior: "smooth" });
  };

  useEffect(() => {
    scrollToBottom();
  }, [messages]);

  useEffect(() => {
    fetchSavedProjects();
  }, []);

  // Updated addMessage function with format parameter
  const addMessage = (text, sender, data = null, format = 'markdown') => {
    const newMessage = {
      id: Date.now(),
      text,
      sender,
      timestamp: new Date().toLocaleTimeString(),
      data,
      format // 'html' or 'markdown'
    };
    setMessages(prev => [...prev, newMessage]);
  };

  // Custom message content renderer
  const MessageContent = ({ message }) => {
    if (message.format === 'html') {
      return (
        <div 
          className="message-text"
          dangerouslySetInnerHTML={{ __html: message.text }}
        />
      );
    } else {
      return (
        <div className="message-text">
          <ReactMarkdown>{message.text}</ReactMarkdown>
        </div>
      );
    }
  };

  // Helper function to check existing files and compare sizes
  const checkExistingFile = async (fileName, newFileSize) => {
    try {
      const response = await fetch(`${API_CONFIG.API_URL}/check_existing_file`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          file_name: fileName,
          new_file_size: newFileSize
        })
      });

      const result = await response.json();
      return result;
    } catch (error) {
      console.error('Error checking existing file:', error);
      return { exists: false, should_skip: false };
    }
  };

  // Function to save JSON data to file
  const saveJSONToFile = async (jsonData, originalFileName) => {
    try {
      const jsonFileName = originalFileName.replace(/\.(csv|xlsx|xls)$/i, '.json');
      
      const response = await fetch(`${API_CONFIG.API_URL}/save_json_file`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          data: jsonData,
          file_name: jsonFileName,
          folder_name: 'sales_data'
        })
      });

      const result = await response.json();
      return result;
    } catch (error) {
      console.error('Error saving JSON file:', error);
      return { success: false, error: 'Failed to save JSON file' };
    }
  };

  // Enhanced handleSearch function for sales data
  const handleSearch = async (query) => {
    try {
      addMessage("🔍 Searching through sales data...", 'agent', null, 'markdown');
      
      const response = await fetch(`${API_CONFIG.API_URL}/simple_search`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          query: query,
          data: csvData
        })
      });

      const result = await response.json();

      if (result.success) {
        if (result.total_found > 0) {
          // Show what keywords were extracted
          const keywordsText = Object.entries(result.keywords)
            .filter(([key, values]) => values && values.length > 0)
            .map(([key, values]) => `${key}: ${values.join(', ')}`)
            .join(' | ');
          
          const resultsText = formatSalesResults(result.results);
          
          // Use HTML format for search results with formatted profiles
          addMessage(`🔍 <strong>Sales Search Results</strong> (${result.total_found} found)<br><br><strong>Keywords extracted:</strong> ${keywordsText}<br><br>${resultsText}`, 'agent', {
            type: 'search_results',
            data: {
              query: query,
              results: result.results,
              total_found: result.total_found,
              keywords: result.keywords
            }
          }, 'html');
          
          // Show more results option if there are many
          if (result.total_found > 5) {
            addMessage(`💡 **Found ${result.total_found} total prospects.** Showing top 5. Try being more specific to narrow down results.`, 'agent', null, 'markdown');
          }
          
          // Suggest related searches if results are limited
          if (result.total_found < 3 && result.total_found > 0) {
            // addMessage("💡 **Want more results?** Try:\n• Using broader terms\n• Searching by different fields\n• Checking spelling of keywords", 'agent', null, 'markdown');
          }
        } else {
          // No results found - provide helpful suggestions
          const keywordsText = Object.entries(result.keywords)
            .filter(([key, values]) => values && values.length > 0)
            .map(([key, values]) => `${key}: ${values.join(', ')}`)
            .join(' | ');
          
          addMessage(`🔍 **No prospects found** for: "${query}"\n\n**Keywords I looked for:** ${keywordsText}\n\n💡 **Try:**\n• Different spelling or synonyms\n• Broader search terms\n• Different fields (company vs industry vs deal value)\n\n**Example searches:**\n• "Enterprise clients" (instead of "Fortune 500 clients")\n• "SaaS companies"\n• "High value deals"`, 'agent', null, 'markdown');
        }
      } else {
        addMessage(`**Search needs more specific input:** ${result.error}\n\nPlease try rephrasing your request e.g., mention company type or deal stage`, 'agent', null, 'markdown');
      }
    } catch (error) {
      console.error('Search error:', error);
      addMessage("❌ **Connection error.** Please check your connection and try again.", 'agent', null, 'markdown');
    }
  };

  // Sales-specific results formatter
  const formatSalesResults = (results) => {
    return results.map((result, index) => {
      const leadScore = calculateLeadScore(result);
      const priorityClass = leadScore >= 80 ? 'high-priority' : leadScore >= 60 ? 'medium-priority' : 'low-priority';
      
      return `
        <div class="sales-profile-card ${priorityClass}">
          <div class="profile-header">
            <div class="profile-main-info">
              <h4>${result.company || result.name || 'Prospect'}</h4>
              <p class="contact-info">${result.contact_name || result.lead_name || ''} ${result.email ? `• ${result.email}` : ''}</p>
            </div>
            <div class="lead-score">
              <span class="score-badge ${priorityClass}">Score: ${leadScore}</span>
            </div>
          </div>
          
          <div class="profile-details">
            ${result.industry ? `<span class="detail-tag industry">🏢 ${result.industry}</span>` : ''}
            ${result.deal_value ? `<span class="detail-tag deal-value">💰 ${result.deal_value}</span>` : ''}
            ${result.stage ? `<span class="detail-tag stage">📊 ${result.stage}</span>` : ''}
            ${result.location ? `<span class="detail-tag location">📍 ${result.location}</span>` : ''}
          </div>
          
          <div class="profile-actions">
            <button class="action-btn contact" onclick="window.salesHelper.contactProspect(${index})">📞 Contact</button>
            <button class="action-btn favorite" onclick="window.salesHelper.addToFavorites(${index})">⭐ Favorite</button>
            <button class="action-btn notes" onclick="window.salesHelper.addNotes(${index})">📝 Notes</button>
          </div>
        </div>
      `;
    }).join('');
  };

  // Calculate lead score based on available data
  const calculateLeadScore = (prospect) => {
    let score = 0;
    
    // Company size factor
    if (prospect.company_size) {
      const size = prospect.company_size.toLowerCase();
      if (size.includes('enterprise') || size.includes('large')) score += 30;
      else if (size.includes('medium') || size.includes('mid')) score += 20;
      else score += 10;
    }
    
    // Deal value factor
    if (prospect.deal_value) {
      const value = parseFloat(prospect.deal_value.toString().replace(/[^0-9.]/g, ''));
      if (value >= 100000) score += 25;
      else if (value >= 50000) score += 20;
      else if (value >= 10000) score += 15;
      else score += 5;
    }
    
    // Stage factor
    if (prospect.stage) {
      const stage = prospect.stage.toLowerCase();
      if (stage.includes('qualified') || stage.includes('proposal')) score += 20;
      else if (stage.includes('interested') || stage.includes('demo')) score += 15;
      else if (stage.includes('contacted')) score += 10;
      else score += 5;
    }
    
    // Industry factor
    if (prospect.industry) {
      const industry = prospect.industry.toLowerCase();
      if (industry.includes('tech') || industry.includes('software') || industry.includes('saas')) score += 15;
      else if (industry.includes('finance') || industry.includes('healthcare')) score += 10;
      else score += 5;
    }
    
    // Contact information completeness
    if (prospect.email) score += 5;
    if (prospect.phone) score += 5;
    if (prospect.linkedin) score += 5;
    
    return Math.min(score, 100); // Cap at 100
  };

  // Sales-specific CSV Upload Handler
  const handleCSVUpload = async (event) => {
    const file = event.target.files[0];
    if (!file) return;

    const allowedTypes = ['.csv', '.xlsx', '.xls'];
    const fileExtension = file.name.toLowerCase().substring(file.name.lastIndexOf('.'));
    
    if (!allowedTypes.includes(fileExtension)) {
      addMessage("Please upload a valid CSV or XLSX file with sales data.", 'agent', null, 'markdown');
      return;
    }

    setIsLoading(true);

    try {
      // Step 1: Check if file already exists and compare sizes
      const fileCheckResult = await checkExistingFile(file.name, file.size);
      
      if (fileCheckResult.should_skip) {
        // Load existing JSON data
        const jsonFileName = file.name.replace(/\.(csv|xlsx|xls)$/i, '.json');
        const loadResponse = await fetch(`${API_CONFIG.API_URL}/load_json_file`, {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({
            file_name: jsonFileName,
            folder_name: 'sales_data'
          })
        });

        const loadResult = await loadResponse.json();
        
        if (loadResult.success) {
          setCsvData(loadResult.data);

          // Update existing files tracking
          setExistingFiles(prev => new Map(prev.set(file.name, {
            size: file.size,
            processed: true,
            json_file: jsonFileName,
            prospects_count: loadResult.data.length
          })));

          addMessage(`📊 Loaded existing sales data: **${loadResult.data.length} prospects**\n\nData is ready for analysis!`, 'agent', null, 'markdown');

          setIsLoading(false);
          event.target.value = '';
          return;
        }
      }

      // Step 2: File is new or has more data, proceed with conversion
      if (fileCheckResult.exists && !fileCheckResult.should_skip) {
        addMessage(`File exists but new version has more data (${file.size} vs ${fileCheckResult.existing_size} bytes). Updating...`, 'agent', null, 'markdown');
      }

      // Step 3: Upload and convert file to JSON
      const formData = new FormData();
      formData.append('file', file);
      formData.append('folder_name', 'sales_data');
      formData.append('multiple_sheets', 'false');

      const response = await fetch(`${API_CONFIG.API_URL}/file_to_json_convert`, {
        method: 'POST',
        body: formData
      });

      const conversionResult = await response.json();

      if (conversionResult.success) {
        const prospects = conversionResult.data;
        setCsvData(prospects);
        
        // Step 4: Save JSON file for future use
        const saveResult = await saveJSONToFile(prospects, file.name);
        
        if (saveResult.success) {
          addMessage(`📊 Sales data processed successfully! Found ${prospects.length} prospects.\n\n🎯 **Pipeline Analysis:**\n${analyzePipeline(prospects)}`, 'agent', {
            type: 'csv_upload_success',
            data: { 
              prospects_count: prospects.length,
              columns: Object.keys(prospects[0] || {}),
              file_type: fileExtension.substring(1).toUpperCase(),
              json_saved: true,
              json_file: saveResult.file_path
            }
          }, 'markdown');

          // Update existing files tracking
          setExistingFiles(prev => new Map(prev.set(file.name, {
            size: file.size,
            processed: true,
            json_file: saveResult.file_name,
            prospects_count: prospects.length
          })));

        } else {
          addMessage(`File converted successfully! Found ${prospects.length} prospects. (JSON save failed: ${saveResult.error})`, 'agent', {
            type: 'csv_upload_success',
            data: { 
              prospects_count: prospects.length,
              columns: Object.keys(prospects[0] || {}),
              file_type: fileExtension.substring(1).toUpperCase(),
              json_saved: false
            }
          }, 'markdown');
        }

        // Show a preview of the data structure
        const sampleProspect = prospects[0];
        if (sampleProspect) {
          const availableFields = Object.keys(sampleProspect);
          addMessage(`**Data Structure Preview:**\nAvailable fields: ${availableFields.join(', ')}\n\n🔍 You can now search and analyze your sales data!`, 'agent', null, 'markdown');
        }
        
      } else {
        addMessage(`Error processing file: ${conversionResult.error}`, 'agent', null, 'markdown');
      }

    } catch (error) {
      console.error('Upload error:', error);
      addMessage("Error uploading file. Please try again.", 'agent', null, 'markdown');
    } finally {
      setIsLoading(false);
      event.target.value = '';
    }
  };

  // Analyze pipeline data
  const analyzePipeline = (prospects) => {
    const total = prospects.length;
    const stages = {};
    const industries = {};
    let totalValue = 0;
    let avgScore = 0;

    prospects.forEach(prospect => {
      // Stage analysis
      const stage = prospect.stage || 'Unknown';
      stages[stage] = (stages[stage] || 0) + 1;

      // Industry analysis
      const industry = prospect.industry || 'Unknown';
      industries[industry] = (industries[industry] || 0) + 1;

      // Value analysis
      if (prospect.deal_value) {
        const value = parseFloat(prospect.deal_value.toString().replace(/[^0-9.]/g, ''));
        if (!isNaN(value)) totalValue += value;
      }

      // Score analysis
      avgScore += calculateLeadScore(prospect);
    });

    avgScore = Math.round(avgScore / total);

    const topStages = Object.entries(stages)
      .sort(([,a], [,b]) => b - a)
      .slice(0, 3)
      .map(([stage, count]) => `${stage}: ${count}`)
      .join(', ');

    const topIndustries = Object.entries(industries)
      .sort(([,a], [,b]) => b - a)
      .slice(0, 3)
      .map(([industry, count]) => `${industry}: ${count}`)
      .join(', ');

    return `• **Total Prospects:** ${total}
• **Avg Lead Score:** ${avgScore}/100
• **Top Stages:** ${topStages}
• **Top Industries:** ${topIndustries}
• **Pipeline Value:** $${totalValue.toLocaleString()}`;
  };

  // Handle AI-powered sales insights
  const handleSalesInsights = async () => {
    if (!csvData || csvData.length === 0) {
      addMessage("Please upload sales data first to get insights.", 'agent', null, 'markdown');
      return;
    }

    // Use the same sales-helper-chat backend route used for saved lists
    setIsAnalyzing(true);
    addMessage("🔍 Analyzing your sales pipeline for insights...", 'agent', null, 'markdown');

    try {
      const sampleLeads = csvData.slice(0, 25);
      const question = `Provide concise sales pipeline insights for these leads. Focus on lead quality trends, pipeline bottlenecks, revenue opportunities, and recommendations.`;

      const response = await fetch(API_CONFIG.SALES_HELPER_CHAT, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ question, project: { name: 'Uploaded Sales Data' }, leads: sampleLeads })
      });

      const result = await response.json();
      if (result.success && result.answer) {
        addMessage(`🎯 **Sales Pipeline Insights:**\n\n${result.answer}`, 'agent', null, 'markdown');
      } else {
        addMessage(result.error || "Unable to generate insights at this time. Please try again.", 'agent', null, 'markdown');
      }
    } catch (error) {
      console.error('Insights error:', error);
      addMessage("Error generating insights. Please check your connection.", 'agent', null, 'markdown');
    } finally {
      setIsAnalyzing(false);
    }
  };

  // Ask about CSV-uploaded leads using the sales helper chat backend
  const handleAskCsvLeads = async (question) => {
    if (!csvData || csvData.length === 0) {
      addMessage('Please upload sales data first.', 'agent', null, 'markdown');
      return;
    }

    try {
      setIsLoading(true);
      const sampleLeads = csvData.slice(0, 25);
      const response = await fetch(API_CONFIG.SALES_HELPER_CHAT, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ question, project: { name: 'Uploaded Sales Data' }, leads: sampleLeads })
      });

      const result = await response.json();
      if (result.success && result.answer) {
        addMessage(result.answer, 'agent', null, 'markdown');
      } else {
        addMessage(result.error || 'I could not analyze the uploaded leads right now.', 'agent', null, 'markdown');
      }
    } catch (error) {
      console.error('CSV leads chat error:', error);
      addMessage('Error analyzing the uploaded leads. Please try again.', 'agent', null, 'markdown');
    } finally {
      setIsLoading(false);
    }
  };

  const fetchSavedProjects = async () => {
    try {
      setIsLoadingSavedProjects(true);
      const username = getCurrentUsername();
      const response = await fetch(`${API_CONFIG.GET_SAVED_PROJECTS}?username=${encodeURIComponent(username)}`);
      const result = await response.json();

      if (result.success && Array.isArray(result.projects)) {
        setSavedProjects(result.projects);
        if (result.projects.length > 0 && !savedProjectSelection) {
          setSavedProjectSelection(String(result.projects[0].id));
        }
      } else {
        setSavedProjects([]);
      }
    } catch (error) {
      console.error('Error loading saved projects:', error);
      setSavedProjects([]);
    } finally {
      setIsLoadingSavedProjects(false);
    }
  };

  const loadSavedProjectLeads = async (projectId) => {
    if (!projectId) return;

    try {
      setIsLoadingSavedProjectLeads(true);
      const username = getCurrentUsername();
      const response = await fetch(`${API_CONFIG.GET_SAVED_PROJECT_LEADS}/${projectId}/leads?username=${encodeURIComponent(username)}`);
      const result = await response.json();

      if (result.success) {
        setSelectedSavedProject(result.project);
        setSelectedSavedProjectLeads(result.leads || []);
        setSavedProjectSelection(String(projectId));
        addMessage(
          `📂 **Loaded saved leads list:** ${result.project?.name || 'Untitled'}\n\nI can now answer questions about these ${result.leads?.length || 0} leads.`,
          'agent',
          null,
          'markdown'
        );
      } else {
        addMessage(`Unable to load saved list: ${result.error || 'Unknown error'}`, 'agent', null, 'markdown');
      }
    } catch (error) {
      console.error('Error loading saved project leads:', error);
      addMessage('Error loading the selected saved leads list. Please try again.', 'agent', null, 'markdown');
    } finally {
      setIsLoadingSavedProjectLeads(false);
    }
  };

  const handleAskSavedLeads = async (question) => {
    if (!selectedSavedProjectLeads || selectedSavedProjectLeads.length === 0) {
      addMessage('Please open a saved leads list first.', 'agent', null, 'markdown');
      return;
    }

    try {
      setIsLoading(true);
      const response = await fetch(API_CONFIG.SALES_HELPER_CHAT, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          question,
          project: selectedSavedProject,
          leads: selectedSavedProjectLeads
        })
      });

      const result = await response.json();

      if (result.success && result.answer) {
        addMessage(result.answer, 'agent', null, 'markdown');
      } else {
        addMessage(result.error || 'I could not analyze the selected leads list right now.', 'agent', null, 'markdown');
      }
    } catch (error) {
      console.error('Saved leads chat error:', error);
      addMessage('Error analyzing the saved leads list. Please try again.', 'agent', null, 'markdown');
    } finally {
      setIsLoading(false);
    }
  };

  // Handle message sending
  const handleSendMessage = async (e) => {
    e.preventDefault();
    if (!inputMessage.trim() || isLoading) return;

    const message = inputMessage.trim();
    setInputMessage('');

    // Add user message
    addMessage(message, 'user', null, 'markdown');

    if (selectedSavedProjectLeads && selectedSavedProjectLeads.length > 0) {
      await handleAskSavedLeads(message);
      return;
    }
    // If CSV data is loaded, route the question to the sales-helper-chat LLM
    if (csvData && csvData.length > 0) {
      await handleAskCsvLeads(message);
    } else {
      addMessage("Please upload your sales data (CSV/XLSX) or open a saved leads list first, then I can help you search and analyze prospects!", 'agent', null, 'markdown');
    }
  };

  // Global functions for prospect actions
  useEffect(() => {
    window.salesHelper = {
      contactProspect: (index) => {
        const prospect = csvData[index];
        addMessage(`📞 **Initiating contact with ${prospect.company || prospect.name}**\n\n📋 **Contact Details:**\n${prospect.email ? `📧 Email: ${prospect.email}` : ''}\n${prospect.phone ? `📱 Phone: ${prospect.phone}` : ''}\n${prospect.linkedin ? `🔗 LinkedIn: ${prospect.linkedin}` : ''}`, 'agent', null, 'markdown');
      },
      addToFavorites: (index) => {
        const prospect = csvData[index];
        addMessage(`⭐ Added ${prospect.company || prospect.name} to favorites!`, 'agent', null, 'markdown');
      },
      addNotes: (index) => {
        const prospect = csvData[index];
        addMessage(`📝 **Add notes for ${prospect.company || prospect.name}:**\n\nYou can track:\n• Last contact date\n• Discussion points\n• Next follow-up action\n• Decision makers involved`, 'agent', null, 'markdown');
      }
    };

    return () => {
      delete window.salesHelper;
    };
  }, [csvData, addMessage]);

  return (
    <div className="sales-helper-agent">
      <Header />
      
      <div className="main-container">
        {/* Left Section - Upload and Tools */}
        <div className="upload-section">
          <h3>📊 Sales Pipeline Manager</h3>
          
          {/* Sales Data Upload */}
          <div className="upload-card">
            <div className="upload-header">
              <h4>📈 Sales Data</h4>
              {csvData && <span className="status-indicator">✅ Loaded</span>}
            </div>
            <p>Upload your prospects, leads, or customer data in CSV/XLSX format.</p>
            <input
              type="file"
              ref={csvFileRef}
              onChange={handleCSVUpload}
              accept=".csv,.xlsx,.xls"
              style={{ display: 'none' }}
            />
            <button 
              className="upload-btn"
              onClick={() => csvFileRef.current?.click()}
              disabled={isLoading}
            >
              {isLoading ? '⏳ Processing...' : '📊 Upload Sales Data'}
            </button>
            {csvData && (
              <div className="file-info">
                <span>📋 {csvData.length} prospects loaded</span>
              </div>
            )}
          </div>

          {/* Sales Insights */}
          <div className="upload-card">
            <div className="upload-header">
              <h4>🎯 AI Insights</h4>
            </div>
            <p>Get AI-powered insights about your sales pipeline, conversion opportunities, and lead quality.</p>
            <button 
              className="upload-btn secondary"
              onClick={handleSalesInsights}
              disabled={!csvData || isAnalyzing}
            >
              {isAnalyzing ? '🔍 Analyzing...' : '🧠 Generate Insights'}
            </button>
          </div>

          {/* Quick Actions */}
          <div className="upload-card">
            <div className="upload-header">
              <h4>⚡ Quick Actions</h4>
            </div>
            <div className="quick-actions">
              <button 
                className="quick-action-btn"
                onClick={() => csvData && handleAskCsvLeads('Which leads look like high value deals?')}
                disabled={!csvData}
              >
                💰 High Value Deals
              </button>
              <button 
                className="quick-action-btn"
                onClick={() => csvData && handleAskCsvLeads('Show me qualified leads and why they are qualified')}
                disabled={!csvData}
              >
                ✅ Qualified Leads
              </button>
              <button 
                className="quick-action-btn"
                onClick={() => csvData && handleAskCsvLeads('Which leads need follow up and what should the follow up be?')}
                disabled={!csvData}
              >
                📅 Follow-ups
              </button>
              <button 
                className="quick-action-btn"
                onClick={() => csvData && handleAskCsvLeads('Identify enterprise-level prospects from this data')}
                disabled={!csvData}
              >
                🏢 Enterprise
              </button>
            </div>
          </div>
        </div>

        <div className="assistant-workspace">
          <div className="saved-leads-panel">
            <div className="saved-leads-panel-header">
              <div>
                <h2>💾 Saved Leads</h2>
                <p>Select a saved list to open it in the panel.</p>
              </div>
              <button className="refresh-saved-btn" onClick={fetchSavedProjects} disabled={isLoadingSavedProjects}>
                {isLoadingSavedProjects ? 'Refreshing...' : 'Refresh'}
              </button>
            </div>

            <div className="saved-leads-controls">
              <select
                value={savedProjectSelection}
                onChange={(e) => setSavedProjectSelection(e.target.value)}
                className="saved-leads-select"
              >
                <option value="">Select a saved leads list</option>
                {savedProjects.map((project) => (
                  <option key={project.id} value={project.id}>
                    {project.name} ({project.lead_count})
                  </option>
                ))}
              </select>
              <button
                className="open-saved-list-btn"
                onClick={() => loadSavedProjectLeads(savedProjectSelection)}
                disabled={!savedProjectSelection || isLoadingSavedProjectLeads}
              >
                {isLoadingSavedProjectLeads ? 'Opening...' : 'Open List'}
              </button>
            </div>

            {!selectedSavedProject ? (
              <div className="saved-leads-table-scroll">
                {isLoadingSavedProjects ? (
                  <div className="empty-saved-state">Loading saved lists...</div>
                ) : savedProjects.length === 0 ? (
                  <div className="empty-saved-state">No saved leads lists found.</div>
                ) : (
                  <table className="businesses-table saved-projects-table">
                    <thead>
                      <tr>
                        <th>List Name</th>
                        <th>Query</th>
                        <th>Leads</th>
                        <th>Created</th>
                        <th>Action</th>
                      </tr>
                    </thead>
                    <tbody>
                      {savedProjects.map((project) => (
                        <tr key={project.id}>
                          <td>{project.name}</td>
                          <td>{project.query_used || 'N/A'}</td>
                          <td>{project.lead_count}</td>
                          <td>{project.created_at ? new Date(project.created_at).toLocaleDateString() : 'N/A'}</td>
                          <td>
                            <button className="open-row-btn" onClick={() => loadSavedProjectLeads(project.id)}>
                              Open
                            </button>
                          </td>
                        </tr>
                      ))}
                    </tbody>
                  </table>
                )}
              </div>
            ) : (
              <>
                <div className="selected-saved-list-bar">
                  <div>
                    <h3>{selectedSavedProject?.name}</h3>
                    <p>{selectedSavedProjectLeads.length} leads loaded from the saved list.</p>
                  </div>
                  <button
                    className="back-to-lists-btn"
                    onClick={() => {
                      setSelectedSavedProject(null);
                      setSelectedSavedProjectLeads([]);
                    }}
                  >
                    Back to Lists
                  </button>
                </div>

                <div className="saved-leads-table-scroll">
                  <table className="businesses-table saved-leads-table">
                    <thead>
                      <tr>
                        <th>Business Name</th>
                        <th>Website</th>
                        <th>Phone</th>
                        <th>Email</th>
                        <th>LinkedIn</th>
                        <th>Summary</th>
                      </tr>
                    </thead>
                    <tbody>
                      {selectedSavedProjectLeads.map((lead, index) => (
                        <tr key={`${lead.id || index}`}>
                          <td>{lead.name || 'N/A'}</td>
                          <td>{lead.website ? <a href={lead.website} target="_blank" rel="noopener noreferrer">Visit</a> : 'N/A'}</td>
                          <td>{lead.phone || 'N/A'}</td>
                          <td>{lead.email || (Array.isArray(lead.emails) && lead.emails[0]) || 'N/A'}</td>
                          <td>{lead.linkedin ? <a href={lead.linkedin} target="_blank" rel="noopener noreferrer">View</a> : 'N/A'}</td>
                          <td className="lead-summary-cell">{lead.summary || lead.description || 'N/A'}</td>
                        </tr>
                      ))}
                    </tbody>
                  </table>
                </div>
              </>
            )}
          </div>

          {/* Right Section - Chat Interface */}
          <div className="chat-section">
            <div className="chat-header">
              <h2>💼 Sales Assistant</h2>
              <p>
                {selectedSavedProject
                  ? `Asking questions about ${selectedSavedProject.name}`
                  : 'Search prospects, analyze deals, and get sales insights'}
              </p>
            </div>
            
            <div className="messages-container">
              {messages.map((message) => (
                <div key={message.id} className={`message ${message.sender}`}>
                  <div className="message-content">
                    <MessageContent message={message} />
                    <span className="timestamp">{message.timestamp}</span>
                  </div>
                </div>
              ))}
              <div ref={messagesEndRef} />
            </div>

            <form className="message-form" onSubmit={handleSendMessage}>
              <div className="input-container">
                <input
                  type="text"
                  value={inputMessage}
                  onChange={(e) => setInputMessage(e.target.value)}
                  placeholder={
                    selectedSavedProject
                      ? 'Ask a question about the selected saved leads list...'
                      : 'Search prospects, ask for insights, or analyze deals...'
                  }
                  disabled={isLoading}
                  className="message-input"
                />
                <button 
                  type="submit" 
                  disabled={isLoading || !inputMessage.trim()}
                  className="send-button"
                >
                  {isLoading ? '⏳' : '📤'}
                </button>
              </div>
            </form>
          </div>
        </div>
      </div>
    </div>
  );
}

export default SalesHelperAgent;
