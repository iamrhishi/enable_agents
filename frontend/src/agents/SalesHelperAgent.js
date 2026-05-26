import React, { useState, useRef, useEffect } from 'react';
import Header from '../core/Header';
import '../styles/SalesHelperAgent.css';
import { API_CONFIG } from '../config/apiConfig';
import { authJsonHeaders, authOptionalHeaders } from '../core/authHeaders';
import { useAgentChat } from '../hooks/useAgentChat';
import MessageContent from '../components/MessageContent';

function SalesHelperAgent() {
  const {
    messages, inputMessage, setInputMessage,
    isLoading, setIsLoading, messagesEndRef,
    addMessage, checkExistingFile, saveJSONToFile,
  } = useAgentChat(
    "Welcome to the Sales Helper Agent! I can help you analyze prospects, track leads, and optimize your sales pipeline!",
    'sales_data'
  );

  const [csvData, setCsvData] = useState(null);
  const [userProfile, setUserProfile] = useState(null);
  const [isAnalyzing, setIsAnalyzing] = useState(false);
  const [savedProjects, setSavedProjects] = useState([]);
  const [selectedSavedProject, setSelectedSavedProject] = useState(null);
  const [selectedSavedProjectLeads, setSelectedSavedProjectLeads] = useState([]);
  const [savedProjectSelection, setSavedProjectSelection] = useState('');
  const [isLoadingSavedProjects, setIsLoadingSavedProjects] = useState(false);
  const [isLoadingSavedProjectLeads, setIsLoadingSavedProjectLeads] = useState(false);
  const [campaigns, setCampaigns] = useState([]);
  const [selectedRankingCampaignId, setSelectedRankingCampaignId] = useState('');
  const [rankingCriteria, setRankingCriteria] = useState(
    'Cost / Pricing, Quality of Materials or Products, Reliability & Delivery Performance, Vendor Reputation & Experience, Production Capacity, Compliance & Legal Requirements, Communication & Support, Location & Logistics, Technology & Innovation, Risk Factors, Sustainability & ESG Factors, Payment Terms, Lead Time, After-Sales Service, Customization Capability, Financial Stability, Scalability, Warranty & Return Policies, Inventory Availability, Contract Flexibility, Industry Certifications, Data Security & Confidentiality, Ethical Business Practices, Existing Client References, Supply Chain Stability'
  );
  const [isLoadingCampaigns, setIsLoadingCampaigns] = useState(false);
  const [isRankingVendors, setIsRankingVendors] = useState(false);
  const [rankedVendors, setRankedVendors] = useState([]);
  const [activeWorkspaceView, setActiveWorkspaceView] = useState('savedLeads');
  const csvFileRef = useRef(null);
  const cvFileRef = useRef(null);
  const rankingResultsRef = useRef(null);
  const [existingFiles, setExistingFiles] = useState(new Map());
  const [defaultUserId] = useState('user_001');
  const [userFavorites, setUserFavorites] = useState([]);
  const selectedRankingCampaign = campaigns.find((campaign) => String(campaign.id) === String(selectedRankingCampaignId));
  const savedLeadsCount = selectedSavedProjectLeads.length;

  const getCurrentUserIdentifier = () =>
    localStorage.getItem('userEmail') ||
    localStorage.getItem('username') ||
    localStorage.getItem('firstName') ||
    defaultUserId ||
    'anonymous';

  useEffect(() => {
    fetchSavedProjects();
    fetchCampaigns();
  }, []);

  useEffect(() => {
    if (activeWorkspaceView !== 'vendorRanking' || rankedVendors.length === 0) return;

    // If the parent panel is scrollable, scroll that container to show the results.
    const panel = rankingResultsRef.current?.closest('.saved-leads-panel');
    if (panel) {
      // compute target top relative to panel and perform a delayed scroll
      const panelRect = panel.getBoundingClientRect();
      const targetRect = rankingResultsRef.current.getBoundingClientRect();
      const offset = 16; // small padding
      const scrollTop = panel.scrollTop + (targetRect.top - panelRect.top) - offset;
      const timeoutId = setTimeout(() => {
        try {
          panel.scrollTo({ top: scrollTop, behavior: 'smooth' });
        } catch (e) {
          /* ignore scroll errors */
        }
      }, 120);
      return () => clearTimeout(timeoutId);
    }

    const scrollTimer = window.requestAnimationFrame(() => {
      rankingResultsRef.current?.scrollIntoView({ behavior: 'smooth', block: 'nearest' });
    });

    return () => window.cancelAnimationFrame(scrollTimer);
  }, [activeWorkspaceView, rankedVendors.length]);

  // Function to save JSON data to file
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

  const fetchCampaigns = async () => {
    try {
      setIsLoadingCampaigns(true);
      const userId = getCurrentUserIdentifier();
      const response = await fetch(`${API_CONFIG.GET_CAMPAIGNS_STATS}?username=${encodeURIComponent(userId)}`, {
        headers: authOptionalHeaders(),
      });
      const result = await response.json();

      if (result.success && Array.isArray(result.campaigns)) {
        setCampaigns(result.campaigns);
        if (!selectedRankingCampaignId && result.campaigns.length > 0) {
          setSelectedRankingCampaignId(String(result.campaigns[0].id));
        }
      } else {
        setCampaigns([]);
      }
    } catch (error) {
      console.error('Error fetching campaigns:', error);
      setCampaigns([]);
    } finally {
      setIsLoadingCampaigns(false);
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

      const userId = getCurrentUserIdentifier();
      const response = await fetch(API_CONFIG.SALES_HELPER_CHAT, {
        method: 'POST',
        headers: authJsonHeaders(),
        body: JSON.stringify({ question, project: { name: 'Uploaded Sales Data' }, leads: sampleLeads, user_id: userId })
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
      const userId = getCurrentUserIdentifier();
      const response = await fetch(API_CONFIG.SALES_HELPER_CHAT, {
        method: 'POST',
        headers: authJsonHeaders(),
        body: JSON.stringify({ question, project: { name: 'Uploaded Sales Data' }, leads: sampleLeads, user_id: userId })
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
      const userId = getCurrentUserIdentifier();
      const response = await fetch(`${API_CONFIG.GET_SAVED_PROJECTS}?username=${encodeURIComponent(userId)}`, {
        headers: authOptionalHeaders(),
      });
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
      const userId = getCurrentUserIdentifier();
      const response = await fetch(`${API_CONFIG.GET_SAVED_PROJECT_LEADS}/${projectId}/leads?username=${encodeURIComponent(userId)}`, {
        headers: authOptionalHeaders(),
      });
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
      const userId = getCurrentUserIdentifier();
      const response = await fetch(API_CONFIG.SALES_HELPER_CHAT, {
        method: 'POST',
        headers: authJsonHeaders(),
        body: JSON.stringify({
          question,
          project: selectedSavedProject,
          leads: selectedSavedProjectLeads,
          user_id: userId
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

  const handleRankVendorReplies = async () => {
    if (!selectedRankingCampaignId) {
      addMessage('Please select a campaign with vendor replies first.', 'agent', null, 'markdown');
      return;
    }

    try {
      setIsRankingVendors(true);
      addMessage('🧮 Ranking vendor replies by your criteria...', 'agent', null, 'markdown');
      const userId = getCurrentUserIdentifier();
      const response = await fetch(API_CONFIG.RANK_CAMPAIGN_VENDORS.replace('{campaignId}', selectedRankingCampaignId), {
        method: 'POST',
        headers: authJsonHeaders(),
        body: JSON.stringify({
          criteria: rankingCriteria,
          user_id: userId,
        }),
      });

      const result = await response.json();
      if (result.success && Array.isArray(result.vendors)) {
        setRankedVendors(result.vendors);
        const campaignName = result.campaign?.name || 'selected campaign';
        addMessage(
          `🏆 **Vendor ranking completed for ${campaignName}:**\n\n${result.vendors.map(v => `${v.rank}. ${v.vendor_name} - ${v.score}/100\n${v.reason || v.reply_summary || ''}`).join('\n\n')}`,
          'agent',
          null,
          'markdown'
        );
      } else {
        setRankedVendors([]);
        addMessage(result.error || 'Unable to rank vendor replies right now.', 'agent', null, 'markdown');
      }
    } catch (error) {
      console.error('Vendor ranking error:', error);
      addMessage('Error ranking vendor replies. Please try again.', 'agent', null, 'markdown');
    } finally {
      setIsRankingVendors(false);
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
        <div className="assistant-workspace">
          <div className="saved-leads-panel">
            <div className="saved-leads-panel-header">
              <div className="panel-title-block">
                <span className="panel-eyebrow">Sales workspace</span>
                <h2>{activeWorkspaceView === 'savedLeads' ? 'Saved Leads' : 'Vendor Reply Ranking'}</h2>
                <p>{activeWorkspaceView === 'savedLeads' ? 'Open any saved list and review it in a focused workspace.' : 'Compare vendor replies against your buying criteria with a cleaner ranking view.'}</p>
                <div className="workspace-summary">
                  {activeWorkspaceView === 'savedLeads' ? (
                    <>
                      <div className="summary-chip"><span>Lists</span><strong>{savedProjects.length}</strong></div>
                      <div className="summary-chip"><span>Loaded leads</span><strong>{savedLeadsCount}</strong></div>
                      <div className="summary-chip"><span>Status</span><strong>{selectedSavedProject ? 'Open' : 'Ready'}</strong></div>
                    </>
                  ) : (
                    <>
                      <div className="summary-chip"><span>Campaigns</span><strong>{campaigns.length}</strong></div>
                      <div className="summary-chip"><span>Ranked</span><strong>{rankedVendors.length}</strong></div>
                      <div className="summary-chip"><span>Replies</span><strong>{selectedRankingCampaign?.totalReplied || 0}</strong></div>
                    </>
                  )}
                </div>
              </div>
              <div style={{ display: 'flex', gap: '10px', alignItems: 'center', flexWrap: 'wrap', justifyContent: 'flex-end' }}>
                <div className="workspace-toggle-group" role="tablist" aria-label="Sales helper workspace toggle">
                  <button
                    type="button"
                    className={`workspace-toggle-btn ${activeWorkspaceView === 'savedLeads' ? 'active' : ''}`}
                    onClick={() => setActiveWorkspaceView('savedLeads')}
                    aria-pressed={activeWorkspaceView === 'savedLeads'}
                  >
                    💾 Saved Leads
                  </button>
                  <button
                    type="button"
                    className={`workspace-toggle-btn ${activeWorkspaceView === 'vendorRanking' ? 'active' : ''}`}
                    onClick={() => setActiveWorkspaceView('vendorRanking')}
                    aria-pressed={activeWorkspaceView === 'vendorRanking'}
                  >
                    🏷️ Vendor Reply Ranking
                  </button>
                </div>
                {activeWorkspaceView === 'savedLeads' ? (
                  <button className="refresh-saved-btn" onClick={fetchSavedProjects} disabled={isLoadingSavedProjects}>
                    {isLoadingSavedProjects ? 'Refreshing...' : 'Refresh'}
                  </button>
                ) : (
                  <button className="refresh-saved-btn" onClick={fetchCampaigns} disabled={isLoadingCampaigns}>
                    {isLoadingCampaigns ? 'Refreshing...' : 'Refresh'}
                  </button>
                )}
              </div>
            </div>

            {activeWorkspaceView === 'savedLeads' ? (
              <div className="panel-body-shell">
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
                      <table className="businesses-table saved-projects-table dashboard-table">
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
                              <td>
                                <div className="table-main-cell">{project.name}</div>
                                <div className="table-subtext">Saved lead list</div>
                              </td>
                              <td className="table-muted">{project.query_used || 'N/A'}</td>
                              <td><span className="inline-pill">{project.lead_count}</span></td>
                              <td className="table-muted">{project.created_at ? new Date(project.created_at).toLocaleDateString() : 'N/A'}</td>
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
                      <table className="businesses-table saved-leads-table dashboard-table">
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
                              <td>
                                <div className="table-main-cell">{lead.name || 'N/A'}</div>
                                <div className="table-subtext">Prospect profile</div>
                              </td>
                              <td>{lead.website ? <a href={lead.website} target="_blank" rel="noopener noreferrer">Visit</a> : 'N/A'}</td>
                              <td className="table-mono">{lead.phone || 'N/A'}</td>
                              <td className="table-mono">{lead.email || (Array.isArray(lead.emails) && lead.emails[0]) || 'N/A'}</td>
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
            ) : (
              <div className="panel-body-shell ranking-shell">
                <div className="ranking-form-card">
                  <div className="ranking-form-grid">
                    <label className="field-group">
                      <span className="field-label">Campaign</span>
                      <select
                        value={selectedRankingCampaignId}
                        onChange={(e) => {
                          setSelectedRankingCampaignId(e.target.value);
                          setRankedVendors([]);
                        }}
                        className="sales-input"
                      >
                        <option value="">Select a campaign with vendor replies</option>
                        {campaigns.map((campaign) => (
                          <option key={campaign.id} value={campaign.id}>
                            {campaign.name} ({campaign.totalReplied || 0} replies)
                          </option>
                        ))}
                      </select>
                    </label>

                    <label className="field-group field-group-full">
                      <span className="field-label">Ranking criteria</span>
                      <textarea
                        value={rankingCriteria}
                        onChange={(e) => setRankingCriteria(e.target.value)}
                        rows={6}
                        className="sales-textarea"
                        placeholder="Enter ranking criteria"
                      />
                    </label>
                  </div>

                  <div className="ranking-meta-row">
                    <button
                      type="button"
                      className="upload-btn secondary ranking-action-btn"
                      onClick={handleRankVendorReplies}
                      disabled={!selectedRankingCampaignId || isRankingVendors || isLoadingCampaigns}
                    >
                      {isRankingVendors ? '🏁 Ranking vendors...' : '🏆 Rank Vendor Replies'}
                    </button>
                  </div>
                </div>

                {/* Ranked replies removed as per UI revision */}
              </div>
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
