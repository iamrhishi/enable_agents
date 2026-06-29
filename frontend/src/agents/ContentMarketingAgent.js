import { API_CONFIG } from '../config/apiConfig';
import React, { useState, useRef, useEffect } from 'react';
import ReactMarkdown from 'react-markdown';
import Header from '../core/Header';
import { BackButton, Input, Textarea, Select } from '../components';
import '../styles/ContentMarketingAgent.css';
import { showToast } from '../core/toast';

// Demo content templates
const DEMO_GENERATED_CONTENT = {
  linkedin: {
    post: `🚀 Excited to share our latest innovation in AI-powered business automation!

At Enable Agents, we're transforming how businesses operate with intelligent automation that adapts to your needs.

Key highlights:
✅ 50% reduction in manual tasks
✅ Real-time insights and analytics
✅ Seamless integration with existing tools

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
    post: `🤖 AI automation is changing the game for businesses.

Our latest update brings:
• 50% faster workflows
• Smart task prioritization
• Real-time collaboration

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
  // Demo mode detection with change listener
  const [isDemoMode, setIsDemoMode] = useState(() => {
    return localStorage.getItem('enableAgentsMode') !== 'live';
  });
  const [step, setStep] = useState('project'); // project, upload, generate, chat
  const [projectId, setProjectId] = useState(null);
  const [projectName, setProjectName] = useState('');
  const [industry, setIndustry] = useState('');
  const [sector, setSector] = useState('');
  
  const [uploadedFiles, setUploadedFiles] = useState([]);
  const [knowledgeGraph, setKnowledgeGraph] = useState(null);
  const [domainContext, setDomainContext] = useState(null);
  
  const [selectedChannel, setSelectedChannel] = useState('linkedin');
  const [contentType, setContentType] = useState('post');
  const [userContext, setUserContext] = useState('');
  const [generatedContent, setGeneratedContent] = useState(null);
  
  const [inputMessage, setInputMessage] = useState('');
  const [messages, setMessages] = useState([
    {
      id: 1,
      text: "Welcome to the Content Marketing Agent! I'll help you create marketing content across all channels using your documents and knowledge graphs.",
      sender: 'agent',
      timestamp: new Date().toLocaleTimeString(),
      format: 'markdown'
    }
  ]);
  
  const [isLoading, setIsLoading] = useState(false);
  const [showKGVisualization, setShowKGVisualization] = useState(false);
  const messagesEndRef = useRef(null);
  const fileInputRef = useRef(null);

  const scrollToBottom = () => {
    messagesEndRef.current?.scrollIntoView({ behavior: "smooth" });
  };

  useEffect(() => {
    scrollToBottom();
  }, [messages]);

  // Listen for mode changes and clear demo data when switching to live
  useEffect(() => {
    const handleModeChange = () => {
      const newMode = localStorage.getItem('enableAgentsMode') !== 'live';
      if (isDemoMode && !newMode) {
        // Switching to live: reset content
        setGeneratedContent(null);
        setMessages([{
          id: 1,
          text: "Welcome to the Content Marketing Agent! I'll help you create marketing content across all channels using your documents and knowledge graphs.",
          sender: 'agent',
          timestamp: new Date().toLocaleTimeString(),
          format: 'markdown'
        }]);
      }
      setIsDemoMode(newMode);
    };
    window.addEventListener('storage', handleModeChange);
    const interval = setInterval(handleModeChange, 1000);
    return () => {
      window.removeEventListener('storage', handleModeChange);
      clearInterval(interval);
    };
  }, [isDemoMode]);

  // ============= PROJECT CREATION =============
  const handleCreateProject = async () => {
    if (!projectName.trim()) {
      showToast('Please enter a project name', 'warning');
      return;
    }

    // Demo mode: simulate project creation
    if (isDemoMode) {
      setProjectId('demo-project-1');
      addMessage(
        `✅ Project "${projectName}" created successfully! (Demo Mode)\n\nYou can skip document upload and go directly to content generation, or upload sample documents.`,
        'agent'
      );
      setStep('upload');
      return;
    }

    setIsLoading(true);
    try {
      const response = await fetch(`${API_CONFIG.API_URL}/api/content-marketing/projects`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          user_id: 'user_001',
          project_name: projectName,
          industry: industry || 'General',
          sector: sector || 'Technology'
        })
      });

      const data = await response.json();
      if (data.success) {
        setProjectId(data.project_id);
        addMessage(
          `✅ Project "${projectName}" created successfully! Now let's upload your documents.`,
          'agent'
        );
        setStep('upload');
      } else {
        addMessage(`❌ Error: ${data.error}`, 'agent');
      }
    } catch (error) {
      addMessage(`❌ Connection error: ${error.message}`, 'agent');
    } finally {
      setIsLoading(false);
    }
  };

  // ============= FILE UPLOAD =============
  const handleFileSelect = async (e) => {
    const files = Array.from(e.target.files);
    if (!files.length) return;

    setIsLoading(true);
    const formData = new FormData();
    formData.append('project_id', projectId);
    
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
          `✅ Uploaded ${data.uploaded_files} files and created knowledge graph!\n\n` +
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
        addMessage(`❌ Error: ${data.error}`, 'agent');
      }
    } catch (error) {
      addMessage(`❌ Upload error: ${error.message}`, 'agent');
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
        `✅ Content generated for **${selectedChannel}** (${contentType})! (Demo Mode)\n\n` +
        `---\n\n` +
        `${demoData.content}\n\n` +
        `---\n\n` +
        `✨ I also generated ${demoData.variations.length} variations. Type 'show variations' to see them.`,
        'agent'
      );
      return;
    }

    setIsLoading(true);
    try {
      const response = await fetch(`${API_CONFIG.API_URL}/api/content-marketing/generate-content`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          project_id: projectId,
          channel: selectedChannel,
          content_type: contentType,
          context: userContext
        })
      });

      const data = await response.json();
      if (data.success) {
        setGeneratedContent(data);
        addMessage(
          `✅ Content generated for **${selectedChannel}** (${contentType})!\n\n` +
          `---\n\n` +
          `${data.content}\n\n` +
          `---\n\n` +
          `✨ I also generated ${data.variations.length} variations. Type 'show variations' to see them.`,
          'agent'
        );
      } else {
        addMessage(`❌ Error: ${data.error}`, 'agent');
      }
    } catch (error) {
      addMessage(`❌ Generation error: ${error.message}`, 'agent');
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
      timestamp: new Date().toLocaleTimeString(),
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
          project_id: projectId,
          message: inputMessage
        })
      });

      const data = await response.json();
      if (data.success) {
        addMessage(data.response, 'agent');
      } else {
        addMessage(`❌ Error: ${data.error}`, 'agent');
      }
    } catch (error) {
      addMessage(`❌ Error: ${error.message}`, 'agent');
    } finally {
      setIsLoading(false);
    }
  };

  // ============= RENDER FUNCTIONS =============
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

  const renderProjectSetup = () => (
    <div className="content-marketing-container">
      <div className="setup-panel">
        <h2>Create New Project</h2>
        <div className="form-group">
          <label>Project Name *</label>
          <Input
            placeholder="e.g., Q1 Marketing Campaign"
            value={projectName}
            onChange={(e) => setProjectName(e.target.value)}
          />
        </div>

        <div className="form-group">
          <label>Industry</label>
          <Select value={industry} onChange={(e) => setIndustry(e.target.value)} variant="outlined">
            <option value="">Select Industry</option>
            <option value="Technology">Technology</option>
            <option value="Healthcare">Healthcare</option>
            <option value="Finance">Finance</option>
            <option value="Retail">Retail & Ecommerce</option>
            <option value="Manufacturing">Manufacturing</option>
            <option value="Real Estate">Real Estate</option>
            <option value="Education">Education</option>
          </Select>
        </div>

        <div className="form-group">
          <label>Sector</label>
          <Input
            placeholder="e.g., SaaS, B2B, D2C"
            value={sector}
            onChange={(e) => setSector(e.target.value)}
          />
        </div>

        <button
          className="btn btn-primary"
          onClick={handleCreateProject}
          disabled={isLoading}
        >
          {isLoading ? 'Creating...' : 'Create Project'}
        </button>
      </div>
    </div>
  );

  const renderUploadStep = () => (
    <div className="content-marketing-container">
      <div className="upload-panel">
        <h2>📤 Upload Your Documents</h2>
        <p>Upload product docs, marketing materials, website content, sales info, etc.</p>
        
        <div className="upload-area" onClick={() => fileInputRef.current?.click()}>
          <div className="upload-icon">📁</div>
          <p>Drag files here or click to select</p>
          <p className="upload-hint">PDF, DOCX, TXT, XLSX, HTML, MD (max 50MB each)</p>
        </div>

        <input
          ref={fileInputRef}
          type="file"
          multiple
          accept=".pdf,.docx,.txt,.xlsx,.html,.md"
          onChange={handleFileSelect}
          style={{ display: 'none' }}
        />

        {uploadedFiles.length > 0 && (
          <div className="uploaded-files">
            <h3>Uploaded Files ({uploadedFiles.length})</h3>
            <div className="file-list">
              {uploadedFiles.map((fileId, idx) => (
                <div key={fileId} className="file-item">
                  ✓ Document {idx + 1}
                </div>
              ))}
            </div>
          </div>
        )}

        {knowledgeGraph && (
          <button 
            className="btn btn-secondary"
            onClick={() => setShowKGVisualization(!showKGVisualization)}
          >
            {showKGVisualization ? '🔍 Hide' : '🔍 View'} Knowledge Graph
          </button>
        )}
      </div>
    </div>
  );

  const renderGenerateStep = () => (
    <div className="content-marketing-container">
      <div className="generate-panel">
        <h2>Generate Marketing Content</h2>

        <div className="form-row">
          <div className="form-group">
            <label>Channel</label>
            <Select
              value={selectedChannel}
              onChange={(e) => setSelectedChannel(e.target.value)}
              variant="outlined"
            >
              <option value="linkedin">LinkedIn</option>
              <option value="email">Email</option>
              <option value="social">Social Media</option>
              <option value="google_ads">Google Ads</option>
            </Select>
          </div>

          <div className="form-group">
            <label>Content Type</label>
            <Select
              value={contentType}
              onChange={(e) => setContentType(e.target.value)}
              variant="outlined"
            >
              <option value="post">Post</option>
              <option value="article">Article</option>
              <option value="ad">Ad Copy</option>
              <option value="email_campaign">Email Campaign</option>
              <option value="case_study">Case Study</option>
            </Select>
          </div>
        </div>

        <div className="form-group">
          <label>Additional Context/Guidance</label>
          <Textarea
            placeholder="e.g., Focus on cost savings, target CTOs, emphasize ROI..."
            value={userContext}
            onChange={(e) => setUserContext(e.target.value)}
            rows={3}
          />
        </div>

        <button
          className="btn btn-primary"
          onClick={handleGenerateContent}
          disabled={isLoading}
        >
          {isLoading ? 'Generating...' : 'Generate Content'}
        </button>

        {generatedContent && (
          <div className="content-preview">
            <h3>📝 Generated Content</h3>
            <div className="content-box">
              <ReactMarkdown>{generatedContent.content}</ReactMarkdown>
            </div>
            {generatedContent.variations.length > 0 && (
              <div className="variations">
                <p className="hint">{generatedContent.variations.length} variations available in chat</p>
              </div>
            )}
          </div>
        )}
      </div>
    </div>
  );

  // ============= MAIN RENDER =============
  return (
    <div className="content-marketing-agent">
      <Header />
      <BackButton />

      <div className="cma-main-content">
        <div className="cma-left-panel">
          {step === 'project' && renderProjectSetup()}
          {step === 'upload' && renderUploadStep()}
          {step === 'generate' && renderGenerateStep()}
        </div>

        <div className="cma-right-panel">
          <div className="chat-container">
            <div className="messages-container">
              {messages.map((msg) => (
                <div key={msg.id} className={`message message-${msg.sender}`}>
                  <div className="message-time">{msg.timestamp}</div>
                  <MessageContent message={msg} />
                </div>
              ))}
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
      </div>
    </div>
  );
}

export default ContentMarketingAgent;
