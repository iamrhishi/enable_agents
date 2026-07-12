import React, { useState, useRef, useEffect } from 'react';
import Header from '../core/Header';
import '../styles/SalesHelperAgent.css';
import { API_CONFIG } from '../config/apiConfig';
import { authJsonHeaders, authOptionalHeaders } from '../core/authHeaders';
import { useAgentChat } from '../hooks/useAgentChat';
import MessageContent from '../components/MessageContent';
import { formatDate, formatTime, getRelativeDateLabel, isSameDay } from '../utils/dateFormat';

// Demo mock data for Sales Helper
const DEMO_SAVED_PROJECTS = [
  { id: 'demo-1', name: 'Enterprise Prospects Q2', query_used: 'Enterprise software buyers', lead_count: 45, created_at: '2026-06-10' },
  { id: 'demo-2', name: 'SMB Tech Companies', query_used: 'Small business technology', lead_count: 32, created_at: '2026-06-15' },
  { id: 'demo-3', name: 'Healthcare Leads', query_used: 'Healthcare technology', lead_count: 28, created_at: '2026-06-20' },
];

const DEMO_PROJECT_LEADS = [
  { name: 'Acme Corp', website: 'https://acme.com', phone: '+1 (555) 123-4567', address: 'San Francisco, CA', email: 'sales@acme.com', summary: 'Enterprise software company' },
  { name: 'TechStart Inc', website: 'https://techstart.io', phone: '+1 (555) 234-5678', address: 'Austin, TX', email: 'info@techstart.io', summary: 'B2B SaaS platform' },
  { name: 'DataFlow Systems', website: 'https://dataflow.com', phone: '+1 (555) 345-6789', address: 'Seattle, WA', email: 'contact@dataflow.com', summary: 'Data analytics provider' },
];

const DEMO_CAMPAIGNS = [
  { id: 'demo-camp-1', name: 'Q2 Outreach Campaign', subject: 'Partnership Opportunity', totalSent: 45, totalReplied: 12, replyRate: 27, createdAt: '2026-06-12' },
  { id: 'demo-camp-2', name: 'Product Launch Follow-up', subject: 'New Features Available', totalSent: 32, totalReplied: 8, replyRate: 25, createdAt: '2026-06-18' },
];

function SalesHelperAgent() {
  // Demo mode detection
  const [isDemoMode, setIsDemoMode] = useState(() => {
    const stored = localStorage.getItem('enableAgentsMode');
    return stored !== 'live';
  });

  useEffect(() => {
    const handleModeChange = () => {
      const stored = localStorage.getItem('enableAgentsMode');
      setIsDemoMode(stored !== 'live');
    };
    window.addEventListener('storage', handleModeChange);
    const interval = setInterval(handleModeChange, 1000);
    return () => {
      window.removeEventListener('storage', handleModeChange);
      clearInterval(interval);
    };
  }, []);
  const {
    messages, inputMessage, setInputMessage,
    isLoading, setIsLoading, messagesEndRef,
    addMessage,
  } = useAgentChat(
    "Welcome to the Sales Helper Agent! I can help you analyze prospects, track leads, and optimize your sales pipeline!",
    'sales_data'
  );

  const [csvData] = useState(null);
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
  const [isLeadsPanelCollapsed, setIsLeadsPanelCollapsed] = useState(false);
  const [isRankingPanelCollapsed, setIsRankingPanelCollapsed] = useState(false);
  const [isChatOpen, setIsChatOpen] = useState(false);
  const rankingResultsRef = useRef(null);
  const [defaultUserId] = useState('user_001');
  const selectedRankingCampaign = campaigns.find((campaign) => String(campaign.id) === String(selectedRankingCampaignId));
  const savedLeadsCount = selectedSavedProjectLeads.length;

  const getCurrentUserIdentifier = () =>
    localStorage.getItem('userEmail') ||
    localStorage.getItem('username') ||
    localStorage.getItem('firstName') ||
    defaultUserId ||
    'anonymous';

  // Reload data when mode changes
  useEffect(() => {
    // Clear current data first when switching modes
    setSavedProjects([]);
    setCampaigns([]);
    setSelectedSavedProject(null);
    setSelectedSavedProjectLeads([]);
    setRankedVendors([]);
    // Then fetch fresh data for current mode
    fetchSavedProjects();
    fetchCampaigns();
  }, [isDemoMode]);

  // Scroll to ranking results when they appear
  useEffect(() => {
    if (rankedVendors.length === 0 || isRankingPanelCollapsed) return;

    const scrollTimer = window.requestAnimationFrame(() => {
      rankingResultsRef.current?.scrollIntoView({ behavior: 'smooth', block: 'nearest' });
    });

    return () => window.cancelAnimationFrame(scrollTimer);
  }, [rankedVendors.length, isRankingPanelCollapsed]);

  const fetchCampaigns = async () => {
    // Demo mode: use mock campaigns
    if (isDemoMode) {
      setCampaigns(DEMO_CAMPAIGNS);
      if (!selectedRankingCampaignId && DEMO_CAMPAIGNS.length > 0) {
        setSelectedRankingCampaignId(String(DEMO_CAMPAIGNS[0].id));
      }
      return;
    }

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
    // Demo mode: use mock projects
    if (isDemoMode) {
      setSavedProjects(DEMO_SAVED_PROJECTS);
      if (DEMO_SAVED_PROJECTS.length > 0 && !savedProjectSelection) {
        setSavedProjectSelection(String(DEMO_SAVED_PROJECTS[0].id));
      }
      return;
    }

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

    // Demo mode: use mock leads
    if (isDemoMode) {
      const demoProject = DEMO_SAVED_PROJECTS.find(p => p.id === projectId);
      setSelectedSavedProject(demoProject);
      setSelectedSavedProjectLeads(DEMO_PROJECT_LEADS);
      setSavedProjectSelection(String(projectId));
      addMessage(
        `📂 **Loaded saved leads list:** ${demoProject?.name || 'Demo List'}\n\nI can now answer questions about these ${DEMO_PROJECT_LEADS.length} leads. (Demo Mode)`,
        'agent',
        null,
        'markdown'
      );
      return;
    }

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
        <div className="assistant-workspace side-by-side-layout">
          {/* Left Panel - Saved Leads */}
          <div className={`split-panel leads-panel ${isLeadsPanelCollapsed ? 'collapsed' : ''}`}>
            <div className="split-panel-header">
              <div className="panel-title-row">
                <h3>Saved Leads</h3>
                <div className="panel-badges">
                  <span className="mini-badge">{savedProjects.length} lists</span>
                  <span className="mini-badge accent">{savedLeadsCount} loaded</span>
                </div>
              </div>
              <div className="panel-header-btns">
                <button className="icon-btn" onClick={fetchSavedProjects} disabled={isLoadingSavedProjects} title="Refresh">
                  {isLoadingSavedProjects ? '...' : '↻'}
                </button>
                <button
                  className="icon-btn collapse-btn"
                  onClick={() => setIsLeadsPanelCollapsed(!isLeadsPanelCollapsed)}
                  title={isLeadsPanelCollapsed ? 'Expand' : 'Collapse'}
                >
                  {isLeadsPanelCollapsed ? '▶' : '◀'}
                </button>
              </div>
            </div>

            {!isLeadsPanelCollapsed && (
              <div className="split-panel-body">
                <div className="saved-leads-controls">
                  <select
                    value={savedProjectSelection}
                    onChange={(e) => setSavedProjectSelection(e.target.value)}
                    className="saved-leads-select"
                  >
                    <option value="">Select a list</option>
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
                    {isLoadingSavedProjectLeads ? '...' : 'Open'}
                  </button>
                </div>

                {!selectedSavedProject ? (
                  <div className="saved-leads-table-scroll">
                    {isLoadingSavedProjects ? (
                      <div className="empty-saved-state">Loading...</div>
                    ) : savedProjects.length === 0 ? (
                      <div className="empty-state-compact">
                        <p>No saved lists yet. Create lead lists from Market Research.</p>
                      </div>
                    ) : (
                      <table className="businesses-table saved-projects-table dashboard-table compact-table">
                        <thead>
                          <tr>
                            <th>List</th>
                            <th>Leads</th>
                            <th></th>
                          </tr>
                        </thead>
                        <tbody>
                          {savedProjects.map((project) => (
                            <tr key={project.id}>
                              <td>
                                <div className="table-main-cell">{project.name}</div>
                                <div className="table-subtext">{project.query_used || ''}</div>
                              </td>
                              <td><span className="inline-pill">{project.lead_count}</span></td>
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
                    <div className="selected-saved-list-bar compact">
                      <div>
                        <h4>{selectedSavedProject?.name}</h4>
                        <span>{selectedSavedProjectLeads.length} leads</span>
                      </div>
                      <button
                        className="back-to-lists-btn"
                        onClick={() => {
                          setSelectedSavedProject(null);
                          setSelectedSavedProjectLeads([]);
                        }}
                      >
                        ← Back
                      </button>
                    </div>

                    <div className="saved-leads-table-scroll">
                      <table className="businesses-table saved-leads-table dashboard-table compact-table">
                        <thead>
                          <tr>
                            <th>Business</th>
                            <th>Contact</th>
                            <th>Email</th>
                          </tr>
                        </thead>
                        <tbody>
                          {selectedSavedProjectLeads.map((lead, index) => (
                            <tr key={`${lead.id || index}`}>
                              <td>
                                <div className="table-main-cell">{lead.name || 'N/A'}</div>
                                {lead.website && <a href={lead.website} target="_blank" rel="noopener noreferrer" className="table-subtext">Website</a>}
                              </td>
                              <td className="table-mono">{lead.phone || '-'}</td>
                              <td className="table-mono">{lead.email || (Array.isArray(lead.emails) && lead.emails[0]) || '-'}</td>
                            </tr>
                          ))}
                        </tbody>
                      </table>
                    </div>
                  </>
                )}
              </div>
            )}
          </div>

          {/* Right Panel - Vendor Ranking */}
          <div className={`split-panel ranking-panel ${isRankingPanelCollapsed ? 'collapsed' : ''}`}>
            <div className="split-panel-header">
              <div className="panel-title-row">
                <h3>Vendor Ranking</h3>
                <div className="panel-badges">
                  <span className="mini-badge">{campaigns.length} campaigns</span>
                  <span className="mini-badge accent">{rankedVendors.length} ranked</span>
                </div>
              </div>
              <div className="panel-header-btns">
                <button className="icon-btn" onClick={fetchCampaigns} disabled={isLoadingCampaigns} title="Refresh">
                  {isLoadingCampaigns ? '...' : '↻'}
                </button>
                <button
                  className="icon-btn collapse-btn"
                  onClick={() => setIsRankingPanelCollapsed(!isRankingPanelCollapsed)}
                  title={isRankingPanelCollapsed ? 'Expand' : 'Collapse'}
                >
                  {isRankingPanelCollapsed ? '◀' : '▶'}
                </button>
              </div>
            </div>

            {!isRankingPanelCollapsed && (
              <div className="split-panel-body">
                <div className="ranking-form-compact">
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
                      <option value="">Select campaign</option>
                      {campaigns.map((campaign) => (
                        <option key={campaign.id} value={campaign.id}>
                          {campaign.name} ({campaign.totalReplied || 0} replies)
                        </option>
                      ))}
                    </select>
                  </label>

                  <label className="field-group">
                    <span className="field-label">Criteria</span>
                    <textarea
                      value={rankingCriteria}
                      onChange={(e) => setRankingCriteria(e.target.value)}
                      rows={4}
                      className="sales-textarea"
                      placeholder="Cost, Quality, Reliability..."
                    />
                  </label>

                  <button
                    type="button"
                    className="rank-btn"
                    onClick={handleRankVendorReplies}
                    disabled={!selectedRankingCampaignId || isRankingVendors || isLoadingCampaigns}
                  >
                    {isRankingVendors ? 'Ranking...' : 'Rank Vendors'}
                  </button>
                </div>

                {/* Ranked Results */}
                {rankedVendors.length > 0 && (
                  <div className="ranked-results" ref={rankingResultsRef}>
                    <h4>Results</h4>
                    <div className="ranked-list">
                      {rankedVendors.map((v) => (
                        <div key={v.rank} className="ranked-item">
                          <span className="rank-num">#{v.rank}</span>
                          <div className="rank-info">
                            <span className="rank-name">{v.vendor_name}</span>
                            <span className="rank-reason">{v.reason || v.reply_summary || ''}</span>
                          </div>
                          <span className="rank-score">{v.score}/100</span>
                        </div>
                      ))}
                    </div>
                  </div>
                )}
              </div>
            )}
          </div>

          {/* Floating Chat Button */}
          {!isChatOpen && (
            <button className="floating-chat-btn" onClick={() => setIsChatOpen(true)}>
              ?
            </button>
          )}

          {/* Right Section - Chat Interface (Collapsible) */}
          {isChatOpen && (
            <div className="chat-section">
              <div className="chat-header">
                <div className="chat-header-content">
                  <h2>Sales Assistant</h2>
                  <p>
                    {selectedSavedProject
                      ? `Asking questions about ${selectedSavedProject.name}`
                      : 'Search prospects, analyze deals, and get sales insights'}
                  </p>
                </div>
                <button
                  className="chat-close-btn"
                  onClick={() => setIsChatOpen(false)}
                  aria-label="Close chat"
                >
                  ✕
                </button>
              </div>

              <div className="messages-container">
                {messages.map((message, index) => {
                  const prevMessage = messages[index - 1];
                  const showDateSeparator = !prevMessage || !isSameDay(message.timestamp, prevMessage.timestamp);
                  return (
                    <React.Fragment key={message.id}>
                      {showDateSeparator && (
                        <div className="date-separator">
                          <span>{getRelativeDateLabel(message.timestamp)}</span>
                        </div>
                      )}
                      <div className={`message ${message.sender}`}>
                        <div className="message-content">
                          <MessageContent message={message} />
                          <span className="timestamp">{formatTime(message.timestamp)}</span>
                        </div>
                      </div>
                    </React.Fragment>
                  );
                })}
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
                    {isLoading ? '...' : '→'}
                  </button>
                </div>
              </form>
            </div>
          )}
        </div>
      </div>
    </div>
  );
}

export default SalesHelperAgent;
