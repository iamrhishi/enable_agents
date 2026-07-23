import { API_CONFIG } from '../config/apiConfig';
import React, { useState, useRef, useEffect, useCallback } from 'react';
import ReactMarkdown from 'react-markdown';
import Header from '../core/Header';
import { BackButton, Textarea, ProjectSelector, LiveModeHint, AgentOutcomesStrip, ProjectGate, Modal } from '../components';
import '../styles/ContentMarketingAgent.css';
import { showToast } from '../core/toast';
import { formatTime, getRelativeDateLabel, isSameDay } from '../utils/dateFormat';
import { useSelectedProjectId } from '../hooks/useSelectedProjectId';
import { useWorkflowContext } from '../hooks';
import { useMode } from '../contexts';

// Storage key for state persistence
const STATE_KEY = 'contentMarketingState';

// Demo campaigns from Market Research
const DEMO_CAMPAIGNS = [
  { id: 'camp-1', name: 'Q2 Outreach', subject: 'Partnership Opportunity', lead_count: 45, status: 'draft' },
  { id: 'camp-2', name: 'Product Launch', subject: 'New Features', lead_count: 32, status: 'draft' },
  { id: 'camp-3', name: 'Enterprise Prospects', subject: 'Enterprise Solutions', lead_count: 28, status: 'active' },
];

// Demo content templates
const DEMO_GENERATED_CONTENT = {
  linkedin: {
    post: `Excited to share our latest innovation in AI-powered business automation!

At Enable Agents, we're transforming how businesses operate with intelligent automation that adapts to your needs.

Key highlights:
- 50% reduction in manual tasks
- Real-time insights and analytics
- Seamless integration with existing tools

The future of work is here. Are you ready?

#AI #Automation #BusinessInnovation #FutureOfWork`,
    article: `# The Future of Business Automation: A Deep Dive

In today's rapidly evolving business landscape, automation isn't just a luxury—it's a necessity. Here's how AI-powered automation is reshaping industries...

## Key Benefits
1. **Increased Efficiency**: Reduce manual tasks by up to 50%
2. **Better Decision Making**: Real-time analytics and insights
3. **Cost Savings**: Lower operational overhead

## Getting Started
The journey to automation begins with understanding your processes...`
  },
  twitter: {
    post: `AI automation is changing the game for businesses.

Our latest update brings:
- 50% faster workflows
- Smart task prioritization
- Real-time collaboration

The future is automated. #AI #BusinessTech`
  },
  email: {
    post: `Subject: Transform Your Business with AI Automation

Dear [Name],

I hope this email finds you well. I wanted to share some exciting developments in business automation that could benefit your organization.

Our AI-powered platform has helped companies achieve:
- 50% reduction in manual tasks
- 30% faster decision-making
- Significant cost savings

Would you be open to a brief call to explore how this could work for your team?

Best regards,
[Your Name]`
  }
};

function ContentMarketingAgent() {
  const selectedProjectId = useSelectedProjectId();
  const projectId = selectedProjectId;

  // Load persisted state
  const loadPersistedState = useCallback(() => {
    try {
      const saved = sessionStorage.getItem(STATE_KEY);
      return saved ? JSON.parse(saved) : {};
    } catch {
      return {};
    }
  }, []);

  const { isDemoMode } = useMode();
  const { isInWorkflow, isHistoryView, stageData, saveStageData } = useWorkflowContext();
  const prevModeRef = useRef(isDemoMode);

  const savedState = loadPersistedState();
  const [step, setStep] = useState(savedState.step || 'upload'); // upload | generate
  const [isChatOpen, setIsChatOpen] = useState(false);

  // Content Marketing keeps its own project record (CMProject) distinct from
  // the platform-wide project - this is its resolved/created id, used for
  // every content-marketing API call instead of the raw platform project id.
  const [cmProjectId, setCmProjectId] = useState(null);
  const [uploadedFiles, setUploadedFiles] = useState([]);
  const [knowledgeGraph, setKnowledgeGraph] = useState(null);
  // eslint-disable-next-line no-unused-vars
  const [domainContext, setDomainContext] = useState(null);

  const [selectedChannel, setSelectedChannel] = useState(savedState.selectedChannel || 'linkedin');
  const [contentType, setContentType] = useState(savedState.contentType || 'post');
  const [userContext, setUserContext] = useState('');
  const [generatedContent, setGeneratedContent] = useState(null);

  const [inputMessage, setInputMessage] = useState('');
  const defaultMessages = [{
    id: 1,
    text: "Welcome to the Content Marketing Agent! I'll help you create marketing content across all channels using your documents and knowledge graphs.",
    sender: 'agent',
    timestamp: new Date().toISOString(),
    format: 'markdown'
  }];
  const [messages, setMessages] = useState(savedState.messages || defaultMessages);

  // Persist state
  useEffect(() => {
    const state = { step, selectedChannel, contentType, messages };
    sessionStorage.setItem(STATE_KEY, JSON.stringify(state));
  }, [step, selectedChannel, contentType, messages]);

  // Clear state when project changes
  useEffect(() => {
    if (!selectedProjectId) {
      setMessages(defaultMessages);
      setStep('upload');
      setGeneratedContent(null);
    }
  // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [selectedProjectId]);
  
  const [isLoading, setIsLoading] = useState(false);
  const [showKGVisualization, setShowKGVisualization] = useState(false);
  const messagesEndRef = useRef(null);
  const fileInputRef = useRef(null);

  // Campaign integration state
  const [availableCampaigns, setAvailableCampaigns] = useState([]);
  const [showCampaignModal, setShowCampaignModal] = useState(false);
  const [selectedCampaignId, setSelectedCampaignId] = useState('');
  const [isSendingToCampaign, setIsSendingToCampaign] = useState(false);

  const scrollToBottom = () => {
    messagesEndRef.current?.scrollIntoView({ behavior: "smooth" });
  };

  useEffect(() => {
    scrollToBottom();
  }, [messages]);

  // Fetch available campaigns from Market Research
  const fetchCampaigns = async () => {
    if (isDemoMode) {
      setAvailableCampaigns(DEMO_CAMPAIGNS);
      return;
    }

    try {
      const userId = localStorage.getItem('userEmail') || 'user_001';
      const response = await fetch(`${API_CONFIG.API_URL}/api/campaigns?username=${encodeURIComponent(userId)}`);
      const data = await response.json();
      if (data.success) {
        setAvailableCampaigns(data.campaigns || []);
      }
    } catch (error) {
      console.error('Error fetching campaigns:', error);
    }
  };

  // Load campaigns when content is generated
  useEffect(() => {
    if (generatedContent) {
      fetchCampaigns();
    }
  }, [generatedContent, isDemoMode]);

  // Send content to campaign
  const handleSendToCampaign = async () => {
    if (!selectedCampaignId || !generatedContent) {
      showToast('Please select a campaign', 'warning');
      return;
    }

    if (isDemoMode) {
      const campaign = availableCampaigns.find(c => c.id === selectedCampaignId);
      showToast(`Content sent to "${campaign?.name}" campaign (Demo Mode)`, 'info');
      addMessage(
        `**Content linked to campaign:** ${campaign?.name}\n\nThe email content has been set for this campaign. You can now send it from the Market Research agent.`,
        'agent'
      );
      setShowCampaignModal(false);
      setSelectedCampaignId('');
      return;
    }

    try {
      setIsSendingToCampaign(true);
      const response = await fetch(`${API_CONFIG.API_URL}/api/campaigns/${selectedCampaignId}/content`, {
        method: 'PUT',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          email_body: generatedContent.content,
          content_type: contentType,
          channel: selectedChannel,
        }),
      });

      const data = await response.json();
      if (data.success) {
        const campaign = availableCampaigns.find(c => c.id === selectedCampaignId);
        showToast(`Content sent to "${campaign?.name}" campaign`, 'success');
        addMessage(
          `**Content linked to campaign:** ${campaign?.name}\n\nThe email content has been set for this campaign. You can now send it from the Market Research agent.`,
          'agent'
        );
        setShowCampaignModal(false);
        setSelectedCampaignId('');
      } else {
        showToast(data.error || 'Failed to send content', 'error');
      }
    } catch (error) {
      console.error('Error sending to campaign:', error);
      showToast('Error sending content to campaign', 'error');
    } finally {
      setIsSendingToCampaign(false);
    }
  };

  // Sync workspace when global project changes
  useEffect(() => {
    if (!selectedProjectId) {
      setStep('upload');
      setUploadedFiles([]);
      setKnowledgeGraph(null);
      setGeneratedContent(null);
      setMessages([]);
      setCmProjectId(null);
      return;
    }

    if (isDemoMode) {
      setAvailableCampaigns(DEMO_CAMPAIGNS);
      setStep('generate');
      setMessages([
        {
          id: 1,
          text: 'Demo project loaded. Choose a channel and content type, then generate sample marketing content.',
          sender: 'agent',
          timestamp: new Date().toISOString(),
          format: 'markdown',
        },
      ]);
      return;
    }

    setStep('upload');
    setCmProjectId(null);
    setMessages([
      {
        id: 1,
        text: "Welcome! Upload your documents to build a knowledge base, then generate content for any channel.",
        sender: 'agent',
        timestamp: new Date().toISOString(),
        format: 'markdown',
      },
    ]);

    // Resolve (or create) the content-marketing project backing this
    // platform project - every subsequent call needs this id, not the
    // platform project id directly.
    (async () => {
      try {
        const userId = localStorage.getItem('userEmail') || 'default_user';
        const response = await fetch(`${API_CONFIG.API_URL}/api/content-marketing/projects`, {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({
            user_id: userId,
            platform_project_id: selectedProjectId,
            project_name: `Platform project ${selectedProjectId}`,
          }),
        });
        const data = await response.json();
        if (data.success) {
          setCmProjectId(data.project_id);
        }
      } catch (error) {
        console.error('Error resolving content-marketing project:', error);
      }
    })();
  }, [isDemoMode, selectedProjectId]);

  // Load workflow data when viewing a completed stage's history
  useEffect(() => {
    if (!isHistoryView) return;

    const data = stageData && Object.keys(stageData).length > 0 ? stageData : null;
    if (!data) return;

    console.log('[ContentMarketing] Loading workflow history:', { isHistoryView, stageData: data });

    if (data.channel) setSelectedChannel(data.channel);
    if (data.content_type) setContentType(data.content_type);
    if (data.personalized_content || data.content) {
      setGeneratedContent({
        content: data.personalized_content || data.content,
        variations: data.variations || [],
      });
    }
    setStep('generate');
    setMessages([{
      id: 1,
      text: `**Workflow History Loaded**\n\nThis stage's content was already generated for **${data.channel || selectedChannel}** (${data.content_type || contentType}) and is shown read-only below.`,
      sender: 'agent',
      timestamp: new Date().toISOString(),
      format: 'markdown',
    }]);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [isHistoryView, stageData]);

  // Handle mode change side effects
  useEffect(() => {
    if (prevModeRef.current && !isDemoMode) {
      // Switching from demo to live: reset content
      setGeneratedContent(null);
      setMessages([{
        id: 1,
        text: "Welcome to the Content Marketing Agent! I'll help you create marketing content across all channels using your documents and knowledge graphs.",
        sender: 'agent',
        timestamp: new Date().toISOString(),
        format: 'markdown'
      }]);
    }
    prevModeRef.current = isDemoMode;
  }, [isDemoMode]);

  // ============= FILE UPLOAD =============
  const handleFileSelect = async (e) => {
    const files = Array.from(e.target.files);
    if (!files.length) return;

    if (!cmProjectId) {
      addMessage('Still setting up this project - please try again in a moment.', 'agent');
      return;
    }

    setIsLoading(true);
    const formData = new FormData();
    formData.append('project_id', cmProjectId);
    
    files.forEach(file => {
      formData.append('files', file);
    });

    try {
      const response = await fetch(`${API_CONFIG.API_URL}/api/content-marketing/documents/upload`, {
        method: 'POST',
        body: formData
      });

      const data = await response.json();
      if (data.success) {
        setUploadedFiles(prev => [...prev, ...data.document_ids]);
        setDomainContext(data.domain_specialization);
        setKnowledgeGraph({
          id: data.knowledge_graph_id,
          ...data
        });

        addMessage(
          `Uploaded ${data.uploaded_files} files and created knowledge graph!\n\n` +
          `**Domain Context:**\n` +
          `- Industry: ${data.domain_specialization.industry}\n` +
          `- Sector: ${data.domain_specialization.sector}\n` +
          `- Function: ${data.domain_specialization.function}\n` +
          `- Key Themes: ${data.domain_specialization.key_themes.join(', ')}\n\n` +
          `Ready to generate marketing content!`,
          'agent'
        );
        
        setStep('generate');
      } else {
        addMessage(`Error: ${data.error}`, 'agent');
      }
    } catch (error) {
      addMessage(`Upload error: ${error.message}`, 'agent');
    } finally {
      setIsLoading(false);
      if (fileInputRef.current) fileInputRef.current.value = '';
    }
  };

  // ============= CONTENT GENERATION =============
  const handleGenerateContent = async () => {
    if (!selectedChannel || !contentType) {
      showToast('Please select channel and content type', 'warning');
      return;
    }

    // Demo mode: use pre-generated content
    if (isDemoMode) {
      const demoContent = DEMO_GENERATED_CONTENT[selectedChannel]?.[contentType] ||
        DEMO_GENERATED_CONTENT[selectedChannel]?.post ||
        DEMO_GENERATED_CONTENT.linkedin.post;

      const demoData = {
        content: demoContent,
        variations: [
          'Variation 1: A more casual tone version',
          'Variation 2: A formal business version',
          'Variation 3: A storytelling approach'
        ]
      };

      setGeneratedContent(demoData);
      addMessage(
        `Content generated for **${selectedChannel}** (${contentType})! (Demo Mode)\n\n` +
        `---\n\n` +
        `${demoData.content}\n\n` +
        `---\n\n` +
        `I also generated ${demoData.variations.length} variations. Type 'show variations' to see them.`,
        'agent'
      );
      if (isInWorkflow) {
        saveStageData({
          personalized_content: demoData.content,
          channel: selectedChannel,
          content_type: contentType,
          variations: demoData.variations,
        });
      }
      return;
    }

    if (!cmProjectId) {
      showToast('Still setting up this project - please try again in a moment.', 'warning');
      return;
    }

    setIsLoading(true);
    try {
      const response = await fetch(`${API_CONFIG.API_URL}/api/content-marketing/generate-content`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          project_id: cmProjectId,
          channel: selectedChannel,
          content_type: contentType,
          context: userContext
        })
      });

      const data = await response.json();
      if (data.success) {
        setGeneratedContent(data);
        addMessage(
          `Content generated for **${selectedChannel}** (${contentType})!\n\n` +
          `---\n\n` +
          `${data.content}\n\n` +
          `---\n\n` +
          `I also generated ${data.variations.length} variations. Type 'show variations' to see them.`,
          'agent'
        );
        if (isInWorkflow) {
          saveStageData({
            personalized_content: data.content,
            channel: selectedChannel,
            content_type: contentType,
            variations: data.variations,
          });
        }
      } else {
        addMessage(`Error: ${data.error}`, 'agent');
      }
    } catch (error) {
      addMessage(`Generation error: ${error.message}`, 'agent');
    } finally {
      setIsLoading(false);
    }
  };

  // ============= CHAT FUNCTIONALITY =============
  const addMessage = (text, sender, format = 'markdown') => {
    const newMessage = {
      id: Date.now(),
      text,
      sender,
      timestamp: new Date().toISOString(),
      format
    };
    setMessages(prev => [...prev, newMessage]);
  };

  const handleChatSubmit = async (e) => {
    e.preventDefault();
    if (!inputMessage.trim()) return;

    addMessage(inputMessage, 'user');
    setInputMessage('');
    setIsLoading(true);

    try {
      const response = await fetch(`${API_CONFIG.API_URL}/api/content-marketing/chat`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          project_id: cmProjectId,
          message: inputMessage
        })
      });

      const data = await response.json();
      if (data.success) {
        addMessage(data.response, 'agent');
      } else {
        addMessage(`Error: ${data.error}`, 'agent');
      }
    } catch (error) {
      addMessage(`Error: ${error.message}`, 'agent');
    } finally {
      setIsLoading(false);
    }
  };

  // ============= RENDER FUNCTIONS =============
  const MessageContent = ({ message }) => {
    // Always use ReactMarkdown for safe rendering - no dangerouslySetInnerHTML
    return (
      <div className="message-text">
        <ReactMarkdown>{message.text}</ReactMarkdown>
      </div>
    );
  };

  // Combined view - no more tabs

  const renderWorkspace = () => (
    <div className="cma-workspace">
      {/* Left: Configuration */}
      <div className="cma-config-panel">
        <h3>Content Settings</h3>

        {isHistoryView && (
          <div className="workflow-history-banner">
            🔒 Viewing completed stage - read-only
          </div>
        )}

        {/* Channel Selection */}
        <div className="config-section">
          <label className="config-label">Channel</label>
          <div className="pill-group">
            {['LinkedIn', 'Email', 'Social', 'Ads'].map(channel => (
              <button
                key={channel}
                type="button"
                className={`config-pill ${selectedChannel === channel.toLowerCase() ? 'selected' : ''}`}
                onClick={() => setSelectedChannel(channel.toLowerCase())}
                disabled={isHistoryView}
              >
                {channel}
              </button>
            ))}
          </div>
        </div>

        {/* Content Type */}
        <div className="config-section">
          <label className="config-label">Type</label>
          <div className="pill-group">
            {['Post', 'Article', 'Ad Copy', 'Email'].map(type => (
              <button
                key={type}
                type="button"
                className={`config-pill ${contentType === type.toLowerCase().replace(' ', '_') ? 'selected' : ''}`}
                onClick={() => setContentType(type.toLowerCase().replace(' ', '_'))}
                disabled={isHistoryView}
              >
                {type}
              </button>
            ))}
          </div>
        </div>

        {/* Knowledge Base - Documents */}
        <div className="config-section">
          <div className="docs-header-inline">
            <label className="config-label">Knowledge Base</label>
            <span className="docs-count">{uploadedFiles.length} docs</span>
          </div>

          <div
            className="docs-upload-inline"
            onClick={() => !isHistoryView && fileInputRef.current?.click()}
            style={isHistoryView ? { opacity: 0.5, cursor: 'not-allowed' } : {}}
          >
            <img src="/assets/icons/document.png" alt="" className="docs-upload-icon" />
            <span>Add documents</span>
          </div>

          <input
            ref={fileInputRef}
            type="file"
            multiple
            accept=".pdf,.docx,.txt,.xlsx,.html,.md"
            onChange={handleFileSelect}
            disabled={isHistoryView}
            style={{ display: 'none' }}
          />

          {uploadedFiles.length > 0 && (
            <div className="docs-list">
              {uploadedFiles.map((fileId, idx) => (
                <div key={fileId} className="doc-chip">
                  Document {idx + 1}
                </div>
              ))}
            </div>
          )}
        </div>

        <button
          className="generate-btn"
          onClick={handleGenerateContent}
          disabled={isLoading || isHistoryView}
        >
          {isLoading ? 'Generating...' : 'Generate Content'}
        </button>
      </div>

      {/* Right: Context & Output */}
      <div className="cma-output-panel">
        {/* Context Section - expandable */}
        <div className="context-section">
          <label className="config-label">Context / Instructions</label>
          <Textarea
            placeholder="Describe your target audience, key messages, tone, campaign goals, specific talking points..."
            value={userContext}
            onChange={(e) => setUserContext(e.target.value)}
            rows={6}
            className="context-textarea-large"
            disabled={isHistoryView}
          />
        </div>

        {generatedContent && (
          <div className="content-preview">
            <h3>Generated Content</h3>
            <div className="content-box">
              <ReactMarkdown>{generatedContent.content}</ReactMarkdown>
            </div>
            <div className="content-actions">
              {generatedContent.variations.length > 0 && (
                <p className="hint">{generatedContent.variations.length} variations available in chat</p>
              )}
              <button
                className="btn btn-send-campaign"
                onClick={() => setShowCampaignModal(true)}
              >
                Send to Email Campaign
              </button>
            </div>
          </div>
        )}
      </div>
    </div>
  );

  // ============= MAIN RENDER =============
  return (
    <div className="content-marketing-agent">
      <Header />

      <div className="agent-page-header">
        <div className="agent-header-left">
          <BackButton />
          <div className="agent-header-content">
            <div className="agent-title-row">
              <h1>Content Marketing</h1>
            </div>
            <p className="text-muted">
              Create engaging content for all channels using AI and your knowledge base.
            </p>
          </div>
        </div>
        <div className="agent-header-right">
          <ProjectSelector
            agentKey="contentMarketing"
            onProjectChange={(project) => {
              if (project) {
                console.log('Project selected:', project.name);
              }
            }}
          />
        </div>
      </div>

      <AgentOutcomesStrip
        items={[
          { iconSrc: '/assets/icons/copy.png', title: 'Multi-channel content', description: 'LinkedIn, email, Twitter posts from your docs.' },
          { iconSrc: '/assets/icons/retrieval.png', title: 'Knowledge-powered', description: 'Uses uploaded documents and brand context.' },
          { iconSrc: '/assets/icons/bullhorn.png', title: 'Campaign ready', description: 'Send content to Market Research campaigns.' },
        ]}
      />

      <LiveModeHint
        requireProject
        message="Choose a project from the header dropdown, or create one with + New Project. Switch to Demo to explore a sample campaign."
      />

      <ProjectGate agentLabel="Content Marketing workspace">
      <div className="cma-main-content">
        {renderWorkspace()}

        {isChatOpen && (
        <div className="cma-right-panel">
          <div className="chat-container">
            <div className="chat-panel-header">
              <h3>Content Assistant</h3>
              <button type="button" className="chat-close-btn" onClick={() => setIsChatOpen(false)} aria-label="Close chat">
                ×
              </button>
            </div>
            <div className="messages-container">
              {messages.map((msg, index) => {
                const prevMessage = messages[index - 1];
                const showDateSeparator = !prevMessage || !isSameDay(msg.timestamp, prevMessage.timestamp);
                return (
                  <React.Fragment key={msg.id}>
                    {showDateSeparator && (
                      <div className="date-separator">
                        <span>{getRelativeDateLabel(msg.timestamp)}</span>
                      </div>
                    )}
                    <div className={`message message-${msg.sender}`}>
                      <div className="message-time">{formatTime(msg.timestamp)}</div>
                      <MessageContent message={msg} />
                    </div>
                  </React.Fragment>
                );
              })}
              {isLoading && (
                <div className="message message-agent">
                  <div className="typing-indicator">
                    <span></span><span></span><span></span>
                  </div>
                </div>
              )}
              <div ref={messagesEndRef} />
            </div>

            {projectId && (
              <form className="chat-input-form" onSubmit={handleChatSubmit}>
                <input
                  type="text"
                  placeholder="Ask me to refine content, create variations, or get suggestions..."
                  value={inputMessage}
                  onChange={(e) => setInputMessage(e.target.value)}
                  disabled={isLoading}
                />
                <button type="submit" disabled={isLoading}>
                  Send
                </button>
              </form>
            )}
          </div>
        </div>
        )}

        {!isChatOpen && projectId && (
          <button type="button" className="floating-chat-btn" onClick={() => setIsChatOpen(true)} aria-label="Open chat">
            <img src="/assets/icons/chat.png" alt="" className="chat-btn-icon" />
          </button>
        )}
      </div>
      </ProjectGate>

      <Modal
        open={showCampaignModal}
        onClose={() => setShowCampaignModal(false)}
        title="Send to Email Campaign"
        footer={
          <>
            <button type="button" className="btn btn-secondary" onClick={() => setShowCampaignModal(false)}>Cancel</button>
            <button
              type="button"
              className="btn btn-primary"
              onClick={handleSendToCampaign}
              disabled={!selectedCampaignId || isSendingToCampaign}
            >
              {isSendingToCampaign ? 'Sending...' : 'Send to Campaign'}
            </button>
          </>
        }
      >
        <p className="modal-hint">Select a campaign from Market Research to use this content:</p>
        {availableCampaigns.length === 0 ? (
          <div className="no-campaigns">
            <p>No campaigns found. Create a campaign in Market Research first.</p>
          </div>
        ) : (
          <div className="campaign-list">
            {availableCampaigns.map(campaign => (
              <label
                key={campaign.id}
                className={`campaign-option ${selectedCampaignId === campaign.id ? 'selected' : ''}`}
              >
                <input
                  type="radio"
                  name="campaign"
                  value={campaign.id}
                  checked={selectedCampaignId === campaign.id}
                  onChange={(e) => setSelectedCampaignId(e.target.value)}
                />
                <div className="campaign-option-info">
                  <span className="campaign-name">{campaign.name}</span>
                  <span className="campaign-meta">{campaign.lead_count} leads · {campaign.status}</span>
                </div>
              </label>
            ))}
          </div>
        )}
      </Modal>
    </div>
  );
}

export default ContentMarketingAgent;
