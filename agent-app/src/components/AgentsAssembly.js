import axios from 'axios';
import React, { useState, useEffect } from 'react';
import { useNavigate } from 'react-router-dom';
import Header from './Header';
import '../styles/AgentsAssembly.css';


// Move these functions OUTSIDE the component
const analyzeBusinessRequirements = (query) => {
  const queryLower = query.toLowerCase();
  
  // Business type patterns
  const businessPatterns = {
    'food delivery': ['food delivery', 'delivery service', 'food service', 'restaurant delivery', 'meal delivery'],
    'restaurant': ['restaurant', 'dining', 'food service', 'eatery', 'cafe', 'bistro'],
    'retail': ['retail', 'store', 'shop', 'selling products', 'merchandise', 'ecommerce'],
    'ecommerce': ['ecommerce', 'online store', 'web store', 'digital commerce', 'online selling'],
    'small business': ['small business', 'startup', 'growing business', 'new business', 'entrepreneur'],
    'manufacturing': ['manufacturing', 'production', 'factory', 'assembly', 'industrial'],
    'consulting': ['consulting', 'professional services', 'advisory', 'consulting firm'],
    'technology': ['tech company', 'software', 'IT services', 'technology startup', 'app development']
  };
  
  // Need/requirement patterns
  const needPatterns = {
    'orders': ['orders', 'order management', 'process orders', 'order tracking', 'sales orders'],
    'inventory': ['inventory', 'stock', 'warehouse', 'supply chain', 'track products'],
    'invoices': ['invoices', 'billing', 'payments', 'accounts receivable', 'charge customers'],
    'hiring': ['hiring', 'recruitment', 'find employees', 'team building', 'staffing'],
    'suppliers': ['suppliers', 'vendors', 'procurement', 'supply chain', 'sourcing'],
    'reports': ['reports', 'analytics', 'insights', 'performance', 'data analysis'],
    'documents': ['documents', 'paperwork', 'file management', 'compliance', 'record keeping'],
    'travel': ['travel', 'business trips', 'client meetings', 'field work', 'remote team'],
    'performance': ['team performance', 'employee evaluation', 'productivity', 'management'],
    'market research': ['market research', 'competitor analysis', 'customer insights', 'market validation']
  };
  
  // Size indicators
  const sizePatterns = {
    'small': ['small', 'startup', 'new', 'growing', 'entrepreneur'],
    'medium': ['medium', 'expanding', 'established', 'scaling'],
    'large': ['large', 'enterprise', 'corporation', 'big company']
  };
  
  const results = {
    businessTypes: [],
    needs: [],
    size: null,
    industries: [],
    confidence: 0
  };
  
  // Analyze business types
  Object.entries(businessPatterns).forEach(([type, patterns]) => {
    patterns.forEach(pattern => {
      if (queryLower.includes(pattern)) {
        results.businessTypes.push(type);
      }
    });
  });
  
  // Analyze needs
  Object.entries(needPatterns).forEach(([need, patterns]) => {
    patterns.forEach(pattern => {
      if (queryLower.includes(pattern)) {
        results.needs.push(need);
      }
    });
  });
  
  // Analyze size
  Object.entries(sizePatterns).forEach(([size, patterns]) => {
    patterns.forEach(pattern => {
      if (queryLower.includes(pattern)) {
        results.size = size;
      }
    });
  });
  
  // Calculate confidence based on matches
  const totalMatches = results.businessTypes.length + results.needs.length + (results.size ? 1 : 0);
  results.confidence = Math.min(totalMatches * 0.2, 1.0);
  
  return results;
};

// Smart filtering function
const filterModulesByRequirements = (modules, requirements) => {
  if (!requirements || requirements.confidence < 0.2) {
    return modules; // Return all if confidence is too low
  }
  
  return modules.filter(module => {
    let score = 0;
    
    // Skip technical modules that don't have business context
    if (!module.businessContext || !module.keywords) {
      return false;
    }
    
    // Check business context match
    if (requirements.businessTypes.length > 0) {
      const contextMatch = requirements.businessTypes.some(type =>
        module.businessContext.some(context => 
          context.includes(type) || type.includes(context)
        )
      );
      if (contextMatch) score += 3;
    }
    
    // Check specific needs
    if (requirements.needs.length > 0) {
      const needMatch = requirements.needs.some(need => {
        const moduleName = module.name.toLowerCase();
        return moduleName.includes(need) || 
               module.keywords.some(keyword => keyword.includes(need)) ||
               (module.useCases && module.useCases.some(useCase => useCase.includes(need)));
      });
      if (needMatch) score += 5;
    }
    
    // Boost score for commonly needed modules in specific business types
    if (requirements.businessTypes.includes('food delivery')) {
      if (['Orders', 'Inventory', 'Invoices', 'Supplier Tracking'].includes(module.name)) {
        score += 2;
      }
    }
    
    if (requirements.businessTypes.includes('small business')) {
      if (['Invoices', 'Orders', 'Reports', 'Documents'].includes(module.name)) {
        score += 1;
      }
    }
    
    return score > 0;
  }).sort((a, b) => {
    // Calculate scores for sorting
    let scoreA = 0, scoreB = 0;
    
    requirements.needs.forEach(need => {
      if (a.name.toLowerCase().includes(need)) scoreA += 5;
      if (b.name.toLowerCase().includes(need)) scoreB += 5;
    });
    
    requirements.businessTypes.forEach(type => {
      if (a.businessContext && a.businessContext.includes(type)) scoreA += 3;
      if (b.businessContext && b.businessContext.includes(type)) scoreB += 3;
    });
    
    return scoreB - scoreA; // Sort by relevance (highest first)
  });
};

function AgentsAssembly() {
  const [searchTerm, setSearchTerm] = useState('');
  const [selectedIndustry, setSelectedIndustry] = useState('');
  const [selectedProcess, setSelectedProcess] = useState('');
  const [searchResults, setSearchResults] = useState(null);
  const [filteredModules, setFilteredModules] = useState([]);
  const [userMessage, setUserMessage] = useState('');
  const [inputValue, setInputValue] = useState('');
  const [chatState, setChatState] = useState({});
  const [chatHistory, setChatHistory] = useState([]);
  const [nextQuestion, setNextQuestion] = useState("Tell us more about your business to get agent recommendations");
  const [nextQuestionKey, setNextQuestionKey] = useState("");
  const [completed, setCompleted] = useState(false);
  const [isBuffering, setIsBuffering] = useState(false);
  const [recommendedModules, setRecommendedModules] = useState([]);
  
  const navigate = useNavigate();

  const businessModules = [
    { 
      name: 'Market Research', 
      icon: '/assets/icons/search-analysis.png', 
      price: '$29/month',
      keywords: ['market analysis', 'competitor research', 'customer insights', 'business intelligence', 'market trends'],
      businessContext: ['retail', 'ecommerce', 'startup', 'product launch', 'competitive analysis'],
      industries: ['retail', 'technology', 'healthcare', 'finance', 'manufacturing'],
      useCases: ['understanding market', 'competitive analysis', 'customer research', 'market validation']
    },
    { 
      name: 'Orders', 
      icon: '/assets/icons/orders.png', 
      price: '$35/month',
      keywords: ['order management', 'order processing', 'order tracking', 'sales orders', 'purchase orders'],
      businessContext: ['food delivery', 'ecommerce', 'retail', 'restaurant', 'online store', 'marketplace'],
      industries: ['food service', 'retail', 'ecommerce', 'manufacturing', 'wholesale'],
      useCases: ['managing orders', 'order fulfillment', 'delivery tracking', 'sales processing']
    },
    { 
      name: 'Inventory', 
      icon: '/assets/icons/inventory.png', 
      price: '$25/month',
      keywords: ['stock management', 'inventory tracking', 'warehouse management', 'stock levels', 'supply chain'],
      businessContext: ['food delivery', 'restaurant', 'retail', 'ecommerce', 'manufacturing', 'warehouse'],
      industries: ['food service', 'retail', 'manufacturing', 'wholesale', 'logistics'],
      useCases: ['tracking stock', 'inventory control', 'supply management', 'warehouse operations']
    },
    { 
      name: 'Hiring & Onboarding', 
      icon: '/assets/icons/hr.png', 
      price: '$45/month',
      keywords: ['recruitment', 'hiring process', 'employee onboarding', 'HR management', 'talent acquisition'],
      businessContext: ['growing business', 'startup', 'scaling team', 'remote work', 'human resources'],
      industries: ['all industries', 'technology', 'consulting', 'healthcare', 'finance'],
      useCases: ['hiring employees', 'team expansion', 'recruitment process', 'employee management']
    },
    { 
      name: 'Team Performance', 
      icon: '/assets/icons/performance.png', 
      price: '$39/month',
      keywords: ['performance management', 'employee evaluation', 'productivity tracking', 'team analytics'],
      businessContext: ['management', 'team leadership', 'performance review', 'productivity improvement'],
      industries: ['all industries', 'consulting', 'technology', 'finance', 'healthcare'],
      useCases: ['managing team performance', 'employee evaluation', 'productivity monitoring']
    },
    { 
      name: 'Supplier Tracking', 
      icon: '/assets/icons/suppliers.png', 
      price: '$32/month',
      keywords: ['supplier management', 'vendor tracking', 'procurement', 'supply chain', 'vendor relations'],
      businessContext: ['manufacturing', 'retail', 'food delivery', 'restaurant', 'supply chain management'],
      industries: ['manufacturing', 'retail', 'food service', 'construction', 'healthcare'],
      useCases: ['managing suppliers', 'vendor relationships', 'procurement process', 'supply chain']
    },
    { 
      name: 'Documents', 
      icon: '/assets/icons/documents.png', 
      price: '$22/month',
      keywords: ['document management', 'file storage', 'document workflow', 'paperwork automation'],
      businessContext: ['office management', 'legal compliance', 'document processing', 'administrative tasks'],
      industries: ['all industries', 'legal', 'healthcare', 'finance', 'consulting'],
      useCases: ['managing documents', 'file organization', 'document workflow', 'compliance']
    },
    { 
      name: 'Reports', 
      icon: '/assets/icons/reports.png', 
      price: '$28/month',
      keywords: ['business reporting', 'analytics', 'data visualization', 'business intelligence', 'KPI tracking'],
      businessContext: ['business analysis', 'performance monitoring', 'decision making', 'data-driven insights'],
      industries: ['all industries', 'finance', 'retail', 'manufacturing', 'technology'],
      useCases: ['business reporting', 'performance analysis', 'data insights', 'decision support']
    },
    { 
      name: 'Invoices', 
      icon: '/assets/icons/invoices.png', 
      price: '$26/month',
      keywords: ['invoice management', 'billing', 'accounts receivable', 'payment processing', 'financial management'],
      businessContext: ['food delivery', 'service business', 'freelancing', 'small business', 'accounting'],
      industries: ['all industries', 'professional services', 'retail', 'food service', 'consulting'],
      useCases: ['billing customers', 'invoice processing', 'payment tracking', 'financial management']
    },
    { 
      name: 'Travel Agent', 
      icon: '/assets/icons/travel.png', 
      price: '$42/month',
      keywords: ['travel management', 'trip planning', 'travel booking', 'expense management', 'business travel'],
      businessContext: ['business travel', 'remote work', 'consulting', 'sales team', 'client meetings'],
      industries: ['consulting', 'sales', 'technology', 'professional services', 'field service'],
      useCases: ['managing business travel', 'trip planning', 'travel expenses', 'team travel']
    },
    { 
      name: 'Community Network', 
      icon: '/assets/icons/community.png', 
      price: '$38/month',
      keywords: ['community management', 'network building', 'customer engagement', 'social platform', 'relationship management'],
      businessContext: ['customer engagement', 'brand building', 'social media', 'community building', 'customer loyalty'],
      industries: ['retail', 'technology', 'media', 'nonprofit', 'education'],
      useCases: ['building community', 'customer engagement', 'network management', 'brand loyalty']
    }
  ];

  // Enhanced technical modules with required fields
  const technicalModules = [
    { 
      name: 'Testing AI', 
      icon: '/assets/icons/checklist.png', 
      price: '$55/month',
      keywords: ['automated testing', 'quality assurance', 'test automation', 'bug detection'],
      businessContext: ['software development', 'quality control', 'testing'],
      industries: ['technology', 'software', 'development'],
      useCases: ['automated testing', 'quality assurance', 'bug detection']
    },
    { 
      name: 'LLM Benchmarking', 
      icon: '/assets/icons/bar-chart.png', 
      price: '$65/month',
      keywords: ['AI performance', 'model evaluation', 'benchmarking', 'AI testing'],
      businessContext: ['AI development', 'machine learning', 'model evaluation'],
      industries: ['technology', 'AI', 'research'],
      useCases: ['AI model evaluation', 'performance testing', 'benchmarking']
    },
    { 
      name: 'Data Discovery', 
      icon: '/assets/icons/data-discovery.png', 
      price: '$48/month',
      keywords: ['data analysis', 'data mining', 'insights', 'data exploration'],
      businessContext: ['data analysis', 'business intelligence', 'analytics'],
      industries: ['all industries', 'technology', 'finance'],
      useCases: ['data exploration', 'business insights', 'data analysis']
    },
    // Simple modules without enhanced fields
    { name: 'Users', icon: '/assets/icons/users.png', price: '$35/month' },
    { name: 'Data Security', icon: '/assets/icons/data-security.png', price: '$75/month' },
    { name: 'Alerts', icon: '/assets/icons/alerts.png', price: '$22/month' },
    { name: 'Notifications', icon: '/assets/icons/notifications.png', price: '$18/month' },
    { name: 'Dashboards', icon: '/assets/icons/dashboards.png', price: '$45/month' },
    { name: 'AI Chatbot', icon: '/assets/icons/ai-chatbots.png', price: '$52/month' },
    { name: 'Monitoring', icon: '/assets/icons/monitoring.png', price: '$38/month' },
    { name: 'Analytics', icon: '/assets/icons/analytics.png', price: '$58/month' },
    { name: 'Data Transformation', icon: '/assets/icons/data-transformation.png', price: '$68/month' },
    { name: 'Integration', icon: '/assets/icons/integration.png', price: '$62/month' },
    { name: 'Automation', icon: '/assets/icons/automation.png', price: '$55/month' }
  ];

  // FIXED: Use useEffect to handle filtering instead of calling setState during render
  useEffect(() => {
    let modules = [...businessModules, ...technicalModules];
    
    // If there's a search term, use smart filtering
    if (searchTerm.trim()) {
      const requirements = analyzeBusinessRequirements(searchTerm);
      setSearchResults(requirements); // This is now safe in useEffect
      
      if (requirements.confidence > 0.2) {
        modules = filterModulesByRequirements(modules, requirements);
      } else {
        // Fallback to simple text search
        modules = modules.filter(module =>
          module.name.toLowerCase().includes(searchTerm.toLowerCase()) ||
          (module.keywords && module.keywords.some(keyword => 
            keyword.toLowerCase().includes(searchTerm.toLowerCase())
          )) ||
          (module.businessContext && module.businessContext.some(context =>
            context.toLowerCase().includes(searchTerm.toLowerCase())
          ))
        );
      }
    } else {
      setSearchResults(null);
    }
    
    // Apply other filters
    if (selectedIndustry) {
      modules = modules.filter(module => 
        module.industries && (
          module.industries.includes(selectedIndustry.toLowerCase()) ||
          module.industries.includes('all industries')
        )
      );
    }
    
    setFilteredModules(modules);
  }, [searchTerm, selectedIndustry, selectedProcess]); // Dependencies

  // Rest of your handlers remain the same...
  const handleCardClick = (moduleName) => {
    if (moduleName === 'Data Discovery') {
      navigate('/datainsights');
    }
    else if (moduleName === 'Market Research') {
      navigate('/requirements');
    }
    else if (moduleName === 'AI Chatbot') {
      navigate('/aichatbot');
    }
    else if (moduleName === 'Community Network') {
      navigate('/community-network');
    }
    else if (moduleName === 'Travel Agent') {
      navigate('/travel-agent');
    }
  };

  const handleTryModule = (moduleName) => {
    // console.log('Trying module:', module.name);
    // alert(`Starting free trial for ${module.name}!\n\nDuration: 14 days\nPrice after trial: ${module.price}\n\nClick OK to begin your trial.`);
    if (moduleName === 'Data Discovery') {
      navigate('/datainsights');
    }
    else if (moduleName === 'Market Research') {
      navigate('/requirements');
    }
    else if (moduleName === 'AI Chatbot') {
      navigate('/aichatbot');
    }
    else if (moduleName === 'Community Network') {
      navigate('/community-network');
    }
    else if (moduleName === 'Travel Agent') {
      navigate('/travel-agent');
    }
  
  };

  const handleFileChange = (e) => {
    const file = e.target.files[0];
    if (file) {
      // Handle file upload logic here (show preview, send to backend, etc.)
    }
  };

  const handleBuyModule = (module) => {
    console.log('Buying module:', module.name);
    const confirmPurchase = window.confirm(
      `Purchase ${module.name}?\n\nPrice: ${module.price}\nBilling: Monthly subscription\n\nClick OK to proceed to checkout.`
    );
    
    if (confirmPurchase) {
      alert(`Redirecting to checkout for ${module.name}...`);
    }
  };


const handleEnterpriseChat = async (userInput) => {
  let localChatState = { ...chatState };
  let lastAnswer = userInput;
  let lastQuestionKey = nextQuestionKey;

  // Store previous chat history to append user answers after each system prompt
  let updatedChatHistory = [...chatHistory];

  setIsBuffering(true);
  while (!completed) {
    try {
      // Show buffering message
      setChatHistory(prev => [
        ...prev,
        { type: 'buffer', text: 'Thinking...' }
      ]);

      const res = await axios.post('http://localhost:5000/enterprise_chat', {
        chat_state: localChatState,
        last_answer: lastAnswer,
        last_question_key: lastQuestionKey
      });

      const data = res.data;
      setChatState(data.chat_state || {});
      setCompleted(data.completed);

      // Remove buffering message
      setChatHistory(prev => prev.filter(msg => msg.type !== 'buffer'));

      // If there was a system prompt before, add the user reply after it
      if (updatedChatHistory.length > 0 && lastAnswer) {
        let lastSystemIdx = updatedChatHistory.map(msg => msg.type).lastIndexOf('system');
        if (lastSystemIdx !== -1) {
          updatedChatHistory.splice(lastSystemIdx + 1, 0, { type: 'user', text: lastAnswer });
        } else {
          updatedChatHistory.push({ type: 'user', text: lastAnswer });
        }
      } else if (lastAnswer) {
        updatedChatHistory.push({ type: 'user', text: lastAnswer });
      }

      // Add next system prompt if not completed
      if (data.next_question && !data.completed) {
        updatedChatHistory.push({ type: 'system', text: data.next_question });
      }

      // Add summary if completed
      if (data.completed && data.search_summary) {
        updatedChatHistory.push({ type: 'system', text: data.search_summary });
      }

      setChatHistory(updatedChatHistory);

      if (data.completed) {
        setNextQuestion("Thank you! Here is the summary of your business context.");

        // Show buffering for recommendations
        setChatHistory(prev => [
          ...prev,
          { type: 'buffer', text: 'Finding recommendations...' }
        ]);

        try {
          const recRes = await axios.post('http://localhost:5000/recommend_agents', data.chat_state);
          const recData = recRes.data;

          // Remove buffering message
          setChatHistory(prev => prev.filter(msg => msg.type !== 'buffer'));

          let toolNames = [];
          if (
            recData &&
            recData.recommendations &&
            Array.isArray(recData.recommendations.recommended_tools)
          ) {
            toolNames = recData.recommendations.recommended_tools.map(tool => tool.name);
          }

          setRecommendedModules(toolNames);
        } catch (recErr) {
          setRecommendedModules([]);
        }

        setIsBuffering(false);
        break;
      } else {
        setNextQuestion(data.next_question);
        setNextQuestionKey(data.next_question_key);
        lastAnswer = ""; // Wait for next user input
        setIsBuffering(false);
        break; // Exit loop, wait for next user input
      }
    } catch (err) {
      setChatHistory(prev => [...prev.filter(msg => msg.type !== 'buffer'), { type: 'system', text: "Error contacting chat API." }]);
      setIsBuffering(false);
      break;
    }
  }
};


  const handleChatInput = async (e) => {
    if (e.key === 'Enter' && inputValue.trim()) {
      await handleEnterpriseChat(inputValue.trim());
      setInputValue('');
    }
  };

  return (
    <div className="agents-page">
      <Header />
      <div className="agents-assembly">
        <h2>Agents Assembly</h2>
        
        <div className="chatbot-section">
          <div className="chatbot-container">
            <div className="chat-bubble">
              {chatHistory.length === 0 && (
                <div className="chat-bubble system">
                  {nextQuestion}
                </div>
              )}
              {chatHistory.map((msg, idx) => (
                <div key={idx} className={`chat-bubble ${msg.type}`}>
                  {msg.type === 'buffer' ? (
                    <span>
                      <span className="loading-dots">
                        <span>.</span><span>.</span><span>.</span>
                      </span> {msg.text}
                    </span>
                  ) : (
                    msg.text
                  )}
                </div>
              ))}
            </div>
          </div>
          <div className="chatbot-input-card">
            <input
              id="chat-file-input"
              type="file"
              style={{ display: 'none' }}
              accept="image/*,.pdf,.doc,.docx,.xlsx,.ppt,.pptx"
              onChange={handleFileChange}
            />
            <img
              src="/assets/icons/plus.png"
              alt="Attach file"
              className="chat-plus-btn"
              onClick={() => document.getElementById('chat-file-input').click()}
              title="Attach file"
              tabIndex={0}
              role="button"
              style={{ cursor: 'pointer' }}
            />
            <input
              type="text"
              className="chat-input"
              placeholder={completed ? "Business context complete!" : isBuffering ? "Waiting for response..." : "Talk to us!"}
              value={inputValue}
              onChange={e => setInputValue(e.target.value)}
              onKeyDown={handleChatInput}
              disabled={completed || isBuffering}
            />
          </div>
        </div>
        {/* Show recommended modules as cards matching business/technical modules, with a 'Recommended' tag */}
        {recommendedModules.length > 0 && (
          <div className="recommended-modules enhanced">
            <h3>
              <span role="img" aria-label="star" style={{color: '#fbbf24', marginRight: '8px'}}>★</span>
              Recommended Agentic Modules
            </h3>
            <div className="modules-container recommended">
              {recommendedModules.map((name, idx) => {
                // Find module details from businessModules or technicalModules
                const module = businessModules.find(m => m.name === name) || technicalModules.find(m => m.name === name);
                if (!module) return null;
                return (
                  <div
                    key={idx}
                    className={`module-card recommended-card ${businessModules.some(b => b.name === name) ? 'business-module' : 'technical-module'}`}
                    style={{ cursor: 'pointer', position: 'relative' }}
                  >
                    <img src={module.icon} alt={module.name} />
                    <p>{module.name}</p>
                    <span className="recommended-tag">Recommended</span>
                    <div className="card-buttons">
                      <button 
                        className="try-button"
                        onClick={(e) => {
                          e.stopPropagation();
                          handleTryModule(module.name);
                        }}
                        title={`Try ${module.name} for free`}
                      >
                        Try
                      </button>
                      <button 
                        className="buy-button"
                        onClick={(e) => {
                          e.stopPropagation();
                          handleBuyModule(module);
                        }}
                        title={`Buy ${module.name} - ${module.price}`}
                      >
                        Buy
                      </button>
                    </div>
                  </div>
                );
              })}
            </div>
          </div>
        )}

        {/* Show search insights */}
        {searchResults && searchResults.confidence > 0.2 && (
          <div className="search-insights">
            <div className="insights-content">
              {searchResults.businessTypes.length > 0 && (
                <p><strong>Business Type:</strong> {searchResults.businessTypes.join(', ')}</p>
              )}
              {searchResults.needs.length > 0 && (
                <p><strong>Identified Needs:</strong> {searchResults.needs.join(', ')}</p>
              )}
              <p><strong>Showing {filteredModules.length} relevant Agentic modules</strong></p>
            </div>
          </div>
        )}

        <div className="dropdown-container">
          <select
            value={selectedIndustry}
            onChange={(e) => setSelectedIndustry(e.target.value)}
            className="dropdown"
          >
            <option value="">Select Industry</option>
            <option value="Retail">Retail</option>
            <option value="Food Service">Food Service</option>
            <option value="Manufacturing">Manufacturing</option>
            <option value="Healthcare">Healthcare</option>
            <option value="Finance">Finance</option>
            <option value="Technology">Technology</option>
            <option value="Consulting">Consulting</option>
          </select>
          <select
            value={selectedProcess}
            onChange={(e) => setSelectedProcess(e.target.value)}
            className="dropdown"
          >
            <option value="">Select Process</option>
            <option value="Sales">Sales</option>
            <option value="Procurement">Procurement</option>
            <option value="HR">HR</option>
            <option value="Operations">Operations</option>
            <option value="Finance">Finance</option>
            <option value="Customer Service">Customer Service</option>
          </select>
        </div>

        {/* Modules Section */}
        <div className="modules-container">
          {filteredModules.length > 0 ? (
            filteredModules.map((module, index) => (
              <div
                key={index}
                className={`module-card ${
                  businessModules.some((b) => b.name === module.name)
                    ? 'business-module'
                    : 'technical-module'
                }`}
                onClick={() => handleCardClick(module.name)}
                style={{ cursor: 'pointer' }}
              >
                <img src={module.icon} alt={module.name} />
                <p>{module.name}</p>
                
                <div className="card-buttons">
                  <button 
                    className="try-button"
                    onClick={(e) => {
                      e.stopPropagation();
                      handleTryModule(module.name);
                    }}
                    title={`Try ${module.name} for free`}
                  >
                    Try
                  </button>
                  <button 
                    className="buy-button"
                    onClick={(e) => {
                      e.stopPropagation();
                      handleBuyModule(module);
                    }}
                    title={`Buy ${module.name} - ${module.price}`}
                  >
                    Buy
                  </button>
                </div>
              </div>
            ))
          ) : (
            <div className="no-results">
              <h3>No modules found</h3>
              <p>Try adjusting your search or browse all available modules.</p>
              <button onClick={() => setSearchTerm('')}>Clear Search</button>
            </div>
          )}
        </div>
      </div>
    </div>
  );
}

export default AgentsAssembly;