import axios from 'axios';
import React, { useState, useEffect, useRef } from 'react';
import { useNavigate } from 'react-router-dom';
import Header from '../core/Header';
import '../styles/AgentsAssembly.css';
import { API_CONFIG } from '../config/apiConfig';
import { fetchAgents } from '../agents/agentRegistry';




// Utility: Analyze requirements from a query string (stub)
const analyzeBusinessRequirements = (query) => {
  // This should return an object like { confidence: 1, needs: [], businessTypes: [] }
  return { confidence: 1, needs: [], businessTypes: [] };
};

// Utility: Filter modules by requirements (basic implementation)
function filterModulesByRequirements(modules, requirements) {
  // Example: filter by needs and businessTypes if present
  if (!requirements) return modules;
  let filtered = modules;
  if (requirements.needs && requirements.needs.length > 0) {
    filtered = filtered.filter(module =>
      requirements.needs.some(need =>
        module.name.toLowerCase().includes(need.toLowerCase()) ||
        (module.keywords && module.keywords.some(k => k.toLowerCase().includes(need.toLowerCase())))
      )
    );
  }
  if (requirements.businessTypes && requirements.businessTypes.length > 0) {
    filtered = filtered.filter(module =>
      module.businessContext && requirements.businessTypes.some(type =>
        module.businessContext.includes(type.toLowerCase())
      )
    );
  }
  return filtered;
}


function AgentsAssembly() {
  // Department/function options for second question
  const departmentOptions = [
    'Sales', 'Marketing', 'Finance', 'Operations', 'HR', 'Customer Service', 'Product', 'IT', 'Legal', 'Procurement', 'R&D', 'Strategy', 'Supply Chain', 'Admin', 'Executive'
  ];
  const [departmentPrompted, setDepartmentPrompted] = useState(false);
  // Industry options for initial selection
  const industryOptions = [
    'Retail', 'Food Service', 'Manufacturing', 'Healthcare', 'Finance', 'Technology', 'Consulting',
    'Education', 'Transportation', 'Hospitality', 'Real Estate', 'Media', 'Nonprofit', 'Legal', 'Construction'
  ];
  const [industryPrompted, setIndustryPrompted] = useState(false);
  const [showDetailedReport, setShowDetailedReport] = useState(false);
  const [detailedReportData, setDetailedReportData] = useState(null);
  const [searchTerm, setSearchTerm] = useState('');
  const [selectedIndustry, setSelectedIndustry] = useState('');
  const [selectedProcess, setSelectedProcess] = useState('');
  const [searchResults, setSearchResults] = useState(null);
  const [businessPage, setBusinessPage] = useState(1);
  const [businessesPerPage] = useState(50); // Show 50 per page
  const [allBusinesses, setAllBusinesses] = useState([]);
  const [filteredModules, setFilteredModules] = useState([]);
  const [userMessage, setUserMessage] = useState('');
  const [inputValue, setInputValue] = useState('');
  const [inputHighlighted, setInputHighlighted] = useState(false);
  const [chatState, setChatState] = useState({});
  const [chatHistory, setChatHistory] = useState([]);
  const [nextQuestion, setNextQuestion] = useState("Tell us more about your business to get agent recommendations");
  const [nextQuestionKey, setNextQuestionKey] = useState("");
  const [completed, setCompleted] = useState(false);
  const [isBuffering, setIsBuffering] = useState(false);
  const [recommendedModules, setRecommendedModules] = useState([]);
  const [moduleTab, setModuleTab] = useState('business'); // 'business' or 'technical'
  const [showChatbot, setShowChatbot] = useState(false);
  const [registryAgents, setRegistryAgents] = useState([]);

  const navigate = useNavigate();

  // Load enabled agents from the backend registry on mount
  useEffect(() => {
    fetchAgents().then((agents) => {
      if (agents.length > 0) setRegistryAgents(agents);
    });
  }, []);

  const businessModules = [
    {
      name: 'Executive Assistant Agent',
      icon: '/assets/icons/networking.png',
      price: 'Free',
      status: 'in-progress',
      keywords: ['executive assistant', 'task management', 'reminders', 'whatsapp', 'stakeholder updates'],
      businessContext: ['executive', 'management', 'personal productivity', 'team coordination'],
      industries: ['all industries'],
      useCases: ['task reminders', 'stakeholder follow-up', 'whatsapp integration', 'executive support']
    },
    { 
      name: 'Market Research', 
      icon: '/assets/icons/search-analysis.png', 
      price: '$29/month',
      status: 'ready',
      keywords: ['market analysis', 'competitor research', 'customer insights', 'business intelligence', 'market trends'],
      businessContext: ['retail', 'ecommerce', 'startup', 'product launch', 'competitive analysis'],
      industries: ['retail', 'technology', 'healthcare', 'finance', 'manufacturing'],
      useCases: ['understanding market', 'competitive analysis', 'customer research', 'market validation']
    },
    {
      name: 'Sales Helper Agent',
      icon: '/assets/icons/increase.png',
      price: '$45/month',
      status: 'in-progress',
      keywords: ['sales', 'sales enablement', 'CRM', 'lead management', 'sales strategy'],
      businessContext: ['sales', 'lead generation', 'customer acquisition', 'sales optimization', 'business growth'],
      industries: ['retail', 'technology', 'ecommerce', 'services', 'consulting'],
      useCases: ['sales optimization', 'lead management', 'sales strategy', 'revenue growth']
    },
    {
      name: 'Content Marketing Agent',
      icon: '/assets/icons/bullhorn.png',
      price: '$49/month',
      status: 'in-progress',
      keywords: ['content marketing', 'content creation', 'marketing strategy', 'brand content', 'SEO'],
      businessContext: ['content marketing', 'brand building', 'digital marketing', 'social media', 'marketing strategy'],
      industries: ['retail', 'technology', 'media', 'education', 'ecommerce'],
      useCases: ['creating marketing content', 'content strategy', 'brand engagement', 'digital marketing']
    },
    { 
      name: 'Hiring & Onboarding', 
      icon: '/assets/icons/hr.png', 
      price: '$45/month',
      status: 'in-progress',
      keywords: ['recruitment', 'hiring process', 'employee onboarding', 'HR management', 'talent acquisition'],
      businessContext: ['growing business', 'startup', 'scaling team', 'remote work', 'human resources'],
      industries: ['all industries', 'technology', 'consulting', 'healthcare', 'finance'],
      useCases: ['hiring employees', 'team expansion', 'recruitment process', 'employee management']
    },
    { 
      name: 'Documents', 
      icon: '/assets/icons/document.png', 
      price: '$22/month',
      status: 'in-progress',
      keywords: ['document management', 'file storage', 'document workflow', 'paperwork automation'],
      businessContext: ['office management', 'legal compliance', 'document processing', 'administrative tasks'],
      industries: ['all industries', 'legal', 'healthcare', 'finance', 'consulting'],
      useCases: ['managing documents', 'file organization', 'document workflow', 'compliance']
    },
    { 
      name: 'Supplier Tracking', 
      icon: '/assets/icons/agreement.png', 
      price: '$32/month',
      status: 'in-progress',
      keywords: ['supplier management', 'vendor tracking', 'procurement', 'supply chain', 'vendor relations'],
      businessContext: ['manufacturing', 'retail', 'food delivery', 'restaurant', 'supply chain management'],
      industries: ['manufacturing', 'retail', 'food service', 'construction', 'healthcare'],
      useCases: ['managing suppliers', 'vendor relationships', 'procurement process', 'supply chain']
    },
    { 
      name: 'Invoices', 
      icon: '/assets/icons/invoices.png', 
      price: '$26/month',
      status: 'in-progress',
      keywords: ['invoice management', 'billing', 'accounts receivable', 'payment processing', 'financial management'],
      businessContext: ['food delivery', 'service business', 'freelancing', 'small business', 'accounting'],
      industries: ['all industries', 'professional services', 'retail', 'food service', 'consulting'],
      useCases: ['billing customers', 'invoice processing', 'payment tracking', 'financial management']
    },
    {
      name: 'Supply Chain Agent',
      icon: '/assets/icons/supply-chain-management.png',
      price: 'Custom',
      status: 'in-progress',
      keywords: ['supply chain', 'impact analysis', 'dashboard', 'visualization', 'logistics'],
      businessContext: ['supply chain', 'logistics', 'operations', 'risk management'],
      industries: ['manufacturing', 'retail', 'logistics', 'operations'],
      useCases: ['visualize supply chain impact', 'event impact analysis', 'dashboard visualization']
    },
    { 
      name: 'Inventory', 
      icon: '/assets/icons/inventory.png', 
      price: '$25/month',
      status: 'in-progress',
      keywords: ['stock management', 'inventory tracking', 'warehouse management', 'stock levels', 'supply chain'],
      businessContext: ['food delivery', 'restaurant', 'retail', 'ecommerce', 'manufacturing', 'warehouse'],
      industries: ['food service', 'retail', 'manufacturing', 'wholesale', 'logistics'],
      useCases: ['tracking stock', 'inventory control', 'supply management', 'warehouse operations']
    },
    { 
      name: 'Orders', 
      icon: '/assets/icons/orders.png', 
      price: '$35/month',
      status: 'in-progress',
      keywords: ['order management', 'order processing', 'order tracking', 'sales orders', 'purchase orders'],
      businessContext: ['food delivery', 'ecommerce', 'retail', 'restaurant', 'online store', 'marketplace'],
      industries: ['food service', 'retail', 'ecommerce', 'manufacturing', 'wholesale'],
      useCases: ['managing orders', 'order fulfillment', 'delivery tracking', 'sales processing']
    },
    { 
      name: 'Travel Agent', 
      icon: '/assets/icons/travel.png', 
      price: '$42/month',
      status: 'in-progress',
      keywords: ['travel management', 'trip planning', 'travel booking', 'expense management', 'business travel'],
      businessContext: ['business travel', 'remote work', 'consulting', 'sales team', 'client meetings'],
      industries: ['consulting', 'sales', 'technology', 'professional services', 'field service'],
      useCases: ['managing business travel', 'trip planning', 'travel expenses', 'team travel']
    },
    { 
      name: 'Community Network', 
      icon: '/assets/icons/community.png', 
      price: '$38/month',
      status: 'in-progress',
      keywords: ['community management', 'network building', 'customer engagement', 'social platform', 'relationship management'],
      businessContext: ['customer engagement', 'brand building', 'social media', 'community building', 'customer loyalty'],
      industries: ['retail', 'technology', 'media', 'nonprofit', 'education'],
      useCases: ['building community', 'customer engagement', 'network management', 'brand loyalty']
    },
    {
      name: 'Invest Agent',
      icon: '/assets/icons/save-money.png',
      price: 'Custom',
      status: 'in-progress',
      keywords: ['investment', 'financial instruments', 'assessment', 'parameters', 'finance'],
      businessContext: ['finance', 'investment', 'portfolio management', 'financial analysis'],
      industries: ['finance', 'investment', 'banking', 'wealth management'],
      useCases: ['assess financial instruments', 'investment analysis', 'parameter dashboard']
    },
    { 
      name: 'Reports', 
      icon: '/assets/icons/reports.png', 
      price: '$28/month',
      status: 'in-progress',
      keywords: ['business reporting', 'analytics', 'data visualization', 'business intelligence', 'KPI tracking'],
      businessContext: ['business analysis', 'performance monitoring', 'decision making', 'data-driven insights'],
      industries: ['all industries', 'finance', 'retail', 'manufacturing', 'technology'],
      useCases: ['business reporting', 'performance analysis', 'data insights', 'decision support']
    },
    { 
      name: 'Team Performance', 
      icon: '/assets/icons/performance.png', 
      price: '$39/month',
      status: 'in-progress',
      keywords: ['performance management', 'employee evaluation', 'productivity tracking', 'team analytics'],
      businessContext: ['management', 'team leadership', 'performance review', 'productivity improvement'],
      industries: ['all industries', 'consulting', 'technology', 'finance', 'healthcare'],
      useCases: ['managing team performance', 'employee evaluation', 'productivity monitoring']
    }
  ];

  // Enhanced technical modules with required fields
  const technicalModules = [
    { 
      name: 'Testing AI', 
      icon: '/assets/icons/checklist.png', 
      price: '$55/month',
      status: 'in-progress',
      keywords: ['automated testing', 'quality assurance', 'test automation', 'bug detection'],
      businessContext: ['software development', 'quality control', 'testing'],
      industries: ['technology', 'software', 'development'],
      useCases: ['automated testing', 'quality assurance', 'bug detection']
    },
    { 
      name: 'LLM Benchmarking', 
      icon: '/assets/icons/bar-chart.png', 
      price: '$65/month',
      status: 'in-progress',
      keywords: ['AI performance', 'model evaluation', 'benchmarking', 'AI testing'],
      businessContext: ['AI development', 'machine learning', 'model evaluation'],
      industries: ['technology', 'AI', 'research'],
      useCases: ['AI model evaluation', 'performance testing', 'benchmarking']
    },
    { 
      name: 'Data Discovery', 
      icon: '/assets/icons/data-discovery.png', 
      price: '$48/month',
      status: 'ready',
      keywords: ['data analysis', 'data mining', 'insights', 'data exploration'],
      businessContext: ['data analysis', 'business intelligence', 'analytics'],
      industries: ['all industries', 'technology', 'finance'],
      useCases: ['data exploration', 'business insights', 'data analysis']
    },
    // Simple modules without enhanced fields
    { name: 'Users', icon: '/assets/icons/users.png', price: '$35/month', status: 'in-progress' },
    { name: 'Data Security', icon: '/assets/icons/data-security.png', price: '$75/month', status: 'in-progress' },
    { name: 'Alerts', icon: '/assets/icons/alerts.png', price: '$22/month', status: 'in-progress' },
    { name: 'Notifications', icon: '/assets/icons/notifications.png', price: '$18/month', status: 'in-progress' },
    { name: 'Dashboards', icon: '/assets/icons/dashboards.png', price: '$45/month', status: 'in-progress' },
    { name: 'AI Chatbot', icon: '/assets/icons/ai-chatbots.png', price: '$52/month', status: 'in-progress' },
    { name: 'Monitoring', icon: '/assets/icons/monitoring.png', price: '$38/month', status: 'in-progress' },
    { name: 'Analytics', icon: '/assets/icons/analytics.png', price: '$58/month', status: 'in-progress' },
    { name: 'Data Transformation', icon: '/assets/icons/data-transformation.png', price: '$68/month', status: 'in-progress' },
    { name: 'Integration', icon: '/assets/icons/integration.png', price: '$62/month', status: 'in-progress' },
    { name: 'Automation', icon: '/assets/icons/automation.png', price: '$55/month', status: 'in-progress' }
  ];

  // FIXED: Use useEffect to handle filtering instead of calling setState during render
  useEffect(() => {
    // Fetch businesses from backend with pagination
    const fetchBusinesses = async () => {
      try {
        const res = await axios.get(`${API_CONFIG.API_URL}/search_businesses`, {
          params: {
            query: searchTerm,
            location: selectedIndustry,
            max_results: 500,
            page: businessPage,
            per_page: businessesPerPage
          }
        });
        setAllBusinesses(res.data.businesses || []);
      } catch (err) {
        setAllBusinesses([]);
      }
    };
    if (searchTerm.trim() || selectedIndustry) {
      fetchBusinesses();
    }
    // Filtering modules logic remains unchanged
    let modules = [...businessModules, ...technicalModules];
    if (searchTerm.trim()) {
      const requirements = analyzeBusinessRequirements(searchTerm);
      setSearchResults(requirements);
      if (requirements.confidence > 0.2) {
        modules = filterModulesByRequirements(modules, requirements);
      } else {
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
    } else if (selectedIndustry) {
      modules = modules.filter(module => 
        module.industries && (
          module.industries.includes(selectedIndustry.toLowerCase()) ||
          module.industries.includes('all industries')
        )
      );
    } else {
      modules = [...businessModules, ...technicalModules];
    }
    setFilteredModules(modules);
  }, [searchTerm, selectedIndustry, selectedProcess, businessPage]);
        {/* Businesses Pagination Section */}
        {allBusinesses.length > 0 && (
          <div className="businesses-pagination">
            <h3>Businesses ({allBusinesses.length} found)</h3>
            <ul>
              {allBusinesses.slice((businessPage-1)*businessesPerPage, businessPage*businessesPerPage).map((biz, idx) => (
                <li key={biz.id || idx}>
                  <strong>{biz.name}</strong> - {biz.address} {biz.rating ? `(Rating: ${biz.rating})` : ''}
                </li>
              ))}
            </ul>
            <div className="pagination-controls">
              <button disabled={businessPage === 1} onClick={() => setBusinessPage(businessPage-1)}>Prev</button>
              <span>Page {businessPage} of {Math.ceil(allBusinesses.length/businessesPerPage)}</span>
              <button disabled={businessPage === Math.ceil(allBusinesses.length/businessesPerPage)} onClick={() => setBusinessPage(businessPage+1)}>Next</button>
            </div>
          </div>
        )}

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
    else if (moduleName === 'Event Networking Agent') {
      navigate('/event-networking-agent');
    }
    else if (moduleName === 'Travel Agent') {
      navigate('/travel-agent');
    }
    else if (moduleName === 'Content Marketing Agent') {
      navigate('/content-marketing');
    }
    else if (moduleName === 'Sales Helper Agent') {
      navigate('/sales-helper');
    }
    else if (moduleName === 'Executive Assistant Agent') {
      navigate('/executive-assistant');
    }
    else if (moduleName === 'Invest Agent') {
      navigate('/invest-agent');
    }
    else if (moduleName === 'Supply Chain Agent') {
      navigate('/supply-chain-agent');
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
    else if (moduleName === 'Content Marketing Agent') {
      navigate('/content-marketing');
    }
    else if (moduleName === 'Sales Helper Agent') {
      navigate('/sales-helper');
    }
    else if (moduleName === 'Event Networking Agent') {
      navigate('/event-networking-agent');
    }
    else if (moduleName === 'Invest Agent') {
      navigate('/invest-agent');
    }
    else if (moduleName === 'Supply Chain Agent') {
      navigate('/supply-chain-agent');
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

      const res = await axios.post(`${API_CONFIG.API_URL}/enterprise_chat`, {
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
        // Insert department options marker only if the question asks about role or department
        if (
          (data.next_question.toLowerCase().includes('role') || data.next_question.toLowerCase().includes('department')) &&
          !updatedChatHistory.some(msg => msg.text === '__DEPARTMENT_OPTIONS__')
        ) {
          updatedChatHistory.push({ type: 'system', text: '__DEPARTMENT_OPTIONS__' });
        }
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
          const recRes = await axios.post(`${API_CONFIG.API_URL}/recommend_agents`, data.chat_state);
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
          setDetailedReportData(recData);
        } catch (recErr) {
          setRecommendedModules([]);
          setDetailedReportData(null);
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






  // Process Map Popup State
  const [showProcessMap, setShowProcessMap] = useState(false);
  const [processMapTab, setProcessMapTab] = useState('visual');
  const [processMapData, setProcessMapData] = useState(null);

  // Dummy process map generator
  const generateProcessMapData = () => {
    return {
      industry: selectedIndustry || 'Generic',
      department: chatState.department || 'General',
      responsibilities: chatState.responsibilities || ['Planning', 'Execution', 'Reporting'],
      steps: [
        { name: 'Initiate', description: 'Start the process', owner: 'Manager' },
        { name: 'Plan', description: 'Plan activities', owner: 'Team Lead' },
        { name: 'Execute', description: 'Carry out tasks', owner: 'Staff' },
        { name: 'Report', description: 'Report outcomes', owner: 'Analyst' }
      ]
    };
  };

  // Handler for process icon click
  const handleProcessClick = () => {
    setProcessMapData(generateProcessMapData());
    setShowProcessMap(true);
    setProcessMapTab('visual');
  };

  // Close popup
  const handleCloseProcessMap = () => {
    setShowProcessMap(false);
  };

  return (
    <div className="agents-page">
      <Header onProcessClick={handleProcessClick} />
      <div className="agents-assembly">
        {/* Process Map Popup */}
        {showProcessMap && (
          <div className="modal-overlay">
            <div className="modal-content" style={{ width: '650px', maxWidth: '98vw', minHeight: '420px', borderRadius: '18px', zIndex: 999 }}>
              <div className="detailed-report-tool-header" style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
                <h2 style={{ fontSize: '1.25em', fontWeight: 600 }}>Process Map</h2>
                <div>
                  <button className={processMapTab === 'visual' ? 'active-tab' : ''} onClick={() => setProcessMapTab('visual')}>Visual</button>
                  <button className={processMapTab === 'json' ? 'active-tab' : ''} onClick={() => setProcessMapTab('json')}>JSON</button>
                  <button className="modal-close" onClick={handleCloseProcessMap} style={{ marginLeft: '18px', fontSize: '1.5em' }}>×</button>
                </div>
              </div>
              <div className="process-map-content" style={{ marginTop: '18px' }}>
                {processMapTab === 'visual' ? (
                  <div style={{ padding: '12px' }}>
                    <div><strong>Industry:</strong> {processMapData.industry}</div>
                    <div><strong>Department:</strong> {processMapData.department}</div>
                    <div><strong>Responsibilities:</strong> {Array.isArray(processMapData.responsibilities) ? processMapData.responsibilities.join(', ') : processMapData.responsibilities}</div>
                    <div style={{ marginTop: '18px' }}>
                      <h4>Process Steps</h4>
                      <ol style={{ paddingLeft: '18px' }}>
                        {processMapData.steps.map((step, idx) => (
                          <li key={idx} style={{ marginBottom: '8px' }}>
                            <strong>{step.name}</strong>: {step.description} <span style={{ color: '#64748b', fontSize: '0.95em' }}>({step.owner})</span>
                          </li>
                        ))}
                      </ol>
                    </div>
                  </div>
                ) : (
                  <pre style={{ background: '#f8fafc', borderRadius: '8px', padding: '18px', fontSize: '1em', color: '#334155', border: '1px solid #e2e8f0', maxHeight: '320px', overflow: 'auto' }}>{JSON.stringify(processMapData, null, 2)}</pre>
                )}
              </div>
            </div>
          </div>
        )}
        <h2>Agents Assembly</h2>
        
        <div style={{ display: 'flex', alignItems: 'center', marginBottom: '24px', gap: '12px' }}>
          <span style={{ fontSize: '0.95rem', fontWeight: 600, color: '#1f2937', letterSpacing: '0.01em' }}>Agent Stage</span>
          <button
            className={`modern-toggle-button ${showChatbot ? 'active' : 'inactive'}`}
            onClick={() => setShowChatbot(!showChatbot)}
            title={showChatbot ? 'Agent Stage: Active' : 'Agent Stage: Inactive'}
          >
            <span className="toggle-circle"></span>
            <span className="toggle-label">{showChatbot ? 'Active' : 'Inactive'}</span>
          </button>
        </div>

        {showChatbot && (
        <div className="chatbot-section">
          <div className="chatbot-container enhanced-chatbot unified-chat">
            <div style={{ background: 'linear-gradient(135deg, #1E3A5F 0%, #2c5282 100%)', padding: '14px 28px', borderBottom: '1px solid rgba(255,255,255,0.1)', display: 'flex', alignItems: 'center', gap: '12px' }}>
              <div>
                <h3 style={{ color: '#ffffff', fontSize: '0.95rem', fontWeight: '600', margin: 0 }}>Business Intelligence Assistant</h3>
              </div>
            </div>
            <div className="chat-history" style={{ maxHeight: 'calc(100vh - 500px)', minHeight: '320px', overflowY: 'auto', paddingBottom: '8px' }}>
              {chatHistory.length === 0 && (
                <>
                  <div className="chat-row system">
                    <div className="chat-avatar system-avatar" />
                    <div className="chat-message system">
                      <span>{nextQuestion}</span>
                    </div>
                  </div>
                  {!industryPrompted && (
                    <div className="chat-row system">
                      <div className="chat-avatar system-avatar" />
                      <div className="chat-message system">
                        <span>
                          <div className="industry-options-list">
                            {industryOptions.map((option, idx) => (
                              <button
                                key={option}
                                className="industry-option-btn"
                                onClick={() => {
                                  setInputValue(`We are in the ${option} industry`);
                                  setIndustryPrompted(true);
                                  setInputHighlighted(true);
                                }}
                              >
                                {option}
                              </button>
                            ))}
                          </div>
                        </span>
                      </div>
                    </div>
                  )}
                </>
              )}
              {chatHistory.map((msg, idx) => {
                if (msg.text === '__DEPARTMENT_OPTIONS__') {
                  // Render department options as a separate chat row
                  return (
                    <div key={idx} className="chat-row system">
                      <div className="chat-avatar system-avatar" />
                      <div className="chat-message system">
                        <span>
                          Please select your department or business function:
                          <div className="industry-options-list">
                            {departmentOptions.map((option, didx) => (
                              <button
                                key={option}
                                className="industry-option-btn"
                                onClick={() => {
                                  setInputValue(`I am in the ${option}`);
                                  setInputHighlighted(true);
                                }}
                              >
                                {option}
                              </button>
                            ))}
                          </div>
                        </span>
                      </div>
                    </div>
                  );
                } else if (msg.text === '' && msg.type === 'system') {
                  // Skip empty system messages (if any)
                  return null;
                } else {
                  // Render all other chat messages as before
                  return (
                    <div key={idx} className={`chat-row ${msg.type}`}> 
                      <div className={`chat-avatar ${msg.type}-avatar`} />
                      <div className={`chat-message ${msg.type}`}> 
                        {msg.type === 'buffer' ? (
                          <span className="buffering">
                            <span className="loading-dots">
                              <span>.</span><span>.</span><span>.</span>
                            </span> {msg.text}
                          </span>
                        ) : (
                          <span>{msg.text}</span>
                        )}
                      </div>
                    </div>
                  );
                }
              })}
              {isBuffering && (
                <div className="chat-row buffer-row">
                  <div className="chat-avatar system-avatar" />
                  <div className="chat-message system">
                    <span className="buffering">
                      <span className="loading-dots">
                        <span>.</span><span>.</span><span>.</span>
                      </span> Thinking ...
                    </span>
                  </div>
                </div>
              )}
            </div>
            <div className="chatbot-input-card enhanced-input unified-input" style={{ borderTop: '1px solid #e2e8f0', background: '#f8fafc', display: 'flex', alignItems: 'center', gap: '7px', padding: '11px 12px' }}>
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
                style={{ cursor: 'pointer', width: '14px', height: '14px', flexShrink: 0 }}
              />
              <input
                type="text"
                className={`chat-input enhanced${inputHighlighted ? ' highlighted' : ''}`}
                placeholder={completed ? "Business context complete!" : isBuffering ? "Waiting for response..." : "Talk to us!"}
                value={inputValue}
                onChange={e => {
                  setInputValue(e.target.value);
                  if (inputHighlighted) setInputHighlighted(false);
                }}
                onKeyDown={e => {
                  if (inputHighlighted) setInputHighlighted(false);
                  if (e.key === 'Enter' && !completed && !isBuffering && inputValue.trim()) {
                    e.preventDefault();
                    handleEnterpriseChat(inputValue);
                    setInputValue('');
                  }
                }}
                disabled={completed || isBuffering}
                autoFocus
                aria-label="Type your message"
                style={{ flex: '1 1 auto', minWidth: 0, minHeight: '28px', padding: '7px 11px', fontSize: '0.81rem', maxWidth: 'calc(100% - 60px)' }}
              />
              <button
                onClick={() => {
                  if (inputValue.trim()) {
                    handleEnterpriseChat(inputValue);
                    setInputValue('');
                  }
                }}
                disabled={completed || isBuffering || !inputValue.trim()}
                className="chat-send-btn"
                title="Send message"
                aria-label="Send message"
                style={{
                  width: '32px',
                  height: '32px',
                  minWidth: '32px',
                  flexShrink: 0,
                  display: 'flex',
                  alignItems: 'center',
                  justifyContent: 'center',
                  background: (inputValue.trim() && !completed && !isBuffering) ? 'linear-gradient(135deg, #C2410C 0%, #B45309 100%)' : '#e5e7eb',
                  color: '#ffffff',
                  border: 'none',
                  borderRadius: '6px',
                  cursor: (inputValue.trim() && !completed && !isBuffering) ? 'pointer' : 'not-allowed',
                  fontSize: '0.75rem',
                  transition: 'all 0.2s ease',
                  fontWeight: '600',
                  boxShadow: (inputValue.trim() && !completed && !isBuffering) ? '0 2px 8px rgba(194, 65, 12, 0.12)' : 'none'
                }}
              >
                →
              </button>
            </div>
          </div>
        </div>
        )}

        {/* Show recommended modules as cards matching business/technical modules, with a 'Recommended' tag and Detailed Report */}
        {recommendedModules.length > 0 && (
          <div className="recommended-modules enhanced">
            <h3 style={{display: 'flex', justifyContent: 'space-between', alignItems: 'center'}}>
              <span>
                <span role="img" aria-label="star" style={{color: '#fbbf24', marginRight: '8px'}}>★</span>
                Recommended Agentic Modules
              </span>
              <span
                className="detailed-report-tag"
                style={{cursor: 'pointer', fontSize: '0.95rem', color: '#2563eb', background: '#e0e7ef', borderRadius: '8px', padding: '4px 12px', marginLeft: '12px', fontWeight: 500}}
                onClick={() => setShowDetailedReport(true)}
              >
                Detailed Report
              </span>
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
            {/* Detailed Report Popup - Visual Tabular Format */}
            {showDetailedReport && detailedReportData && (
              <div className="detailed-report-popup">
                <div className="detailed-report-modal">
                  <button className="detailed-report-close" onClick={() => setShowDetailedReport(false)}>Close</button>
                  <h2 className="detailed-report-title">Detailed Recommendation Report</h2>
                  {/* Recommended Tools Section - Card Style */}
                  <div className="detailed-report-section corporate-section">
                    <div className="corporate-header">
                      <span className="corporate-title">Top Recommended Agentic Tools</span>
                      <span className="corporate-divider" />
                    </div>
                    <div className="detailed-report-tool-list">
                      {(detailedReportData.recommendations?.recommended_tools || []).length === 0 ? (
                        <div className="detailed-report-empty">No recommended tools found.</div>
                      ) : (
                        detailedReportData.recommendations.recommended_tools.map((tool, idx) => (
                          <div key={idx} className="detailed-report-tool-card corporate-card">
                            <div className="detailed-report-tool-header">
                              <span className="detailed-report-tool-name">{tool.name}</span>
                              {tool.relevance && <span className="detailed-report-badge">{tool.relevance}</span>}
                            </div>
                            {tool.description && <div className="detailed-report-tool-desc">{tool.description}</div>}
                          </div>
                        ))
                      )}
                    </div>
                  </div>
                  {/* Potential Integration Section - Card Style */}
                  <div className="detailed-report-section corporate-section">
                    <div className="corporate-header">
                      <span className="corporate-title">Smart Integration Opportunities</span>
                      <span className="corporate-divider" />
                    </div>
                    <div className="detailed-report-integration-list">
                      {(detailedReportData.recommendations?.integration_pairs || []).length === 0 ? (
                        <div className="detailed-report-empty">No integration pairs found.</div>
                      ) : (
                        detailedReportData.recommendations.integration_pairs.map((pairObj, idx) => {
                          let tool1 = '-';
                          let tool2 = '-';
                          let dataShared = '-';
                          let description = '-';
                          if (pairObj.pair && Array.isArray(pairObj.pair)) {
                            tool1 = pairObj.pair[0] || '-';
                            tool2 = pairObj.pair[1] || '-';
                          } else {
                            tool1 = pairObj.tool_1 || '-';
                            tool2 = pairObj.tool_2 || '-';
                          }
                          dataShared = pairObj.data_shared || '-';
                          description = pairObj.integration_description || pairObj.integration_type || '-';
                          return (
                            <div key={idx} className="detailed-report-integration-card corporate-card">
                              <div className="detailed-report-integration-tools">
                                <span className="detailed-report-integration-tool">{tool1}</span>
                                <span className="detailed-report-integration-sep">+</span>
                                <span className="detailed-report-integration-tool">{tool2}</span>
                              </div>
                              <div className="detailed-report-integration-data">
                                <span className="detailed-report-badge">{dataShared}</span>
                              </div>
                              <div className="detailed-report-integration-desc">{description}</div>
                            </div>
                          );
                        })
                      )}
                    </div>
                  </div>
                  {/* Additional Tools Section - Card Style with Companies */}
                  <div className="detailed-report-section corporate-section">
                    <div className="corporate-header">
                      <span className="corporate-title">Other Useful Agentic Tools & Providers</span>
                      <span className="corporate-divider" />
                    </div>
                    <div className="detailed-report-tool-list">
                      {(detailedReportData.recommendations?.additional_tools || []).length === 0 ? (
                        <div className="detailed-report-empty">No additional tools found.</div>
                      ) : (
                        detailedReportData.recommendations.additional_tools.map((tool, idx) => {
                          let companies = '-';
                          if (Array.isArray(tool.companies_offering) && tool.companies_offering.length > 0) {
                            companies = tool.companies_offering.join(', ');
                          } else if (tool.company) {
                            companies = tool.company;
                          }
                          return (
                            <div key={idx} className="detailed-report-tool-card corporate-card">
                              <div className="detailed-report-tool-header">
                                <span className="detailed-report-tool-name">{tool.name}</span>
                              </div>
                              {tool.description && <div className="detailed-report-tool-desc">{tool.description}</div>}
                              <div className="detailed-report-tool-companies">
                                {Array.isArray(tool.companies_offering) && tool.companies_offering.length > 0 ? (
                                  <div className="company-pill-list">
                                    {tool.companies_offering.map((company, cidx) => (
                                      <span key={cidx} className="company-pill" title="Third-party tool suggestion">{company}</span>
                                    ))}
                                  </div>
                                ) : (
                                  <span className="company-pill" title="Third-party tool suggestion">{companies}</span>
                                )}
                              </div>
                            </div>
                          );
                        })
                      )}
                    </div>
                  </div>
                </div>
              </div>
            )}
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

        {/* Module Tabs - Modern Professional */}
        <div className="module-tabs" style={{ display: 'flex', gap: '0', marginBottom: '32px', borderBottom: '1px solid #e8ecf1', paddingBottom: '0' }}>
          <button
            onClick={() => setModuleTab('business')}
            style={{
              padding: '14px 28px',
              borderBottom: moduleTab === 'business' ? '3px solid #c2410c' : '3px solid transparent',
              background: 'transparent',
              cursor: 'pointer',
              fontSize: '0.95rem',
              fontWeight: moduleTab === 'business' ? '700' : '600',
              color: moduleTab === 'business' ? '#c2410c' : '#475569',
              transition: 'all 0.3s cubic-bezier(0.4, 0, 0.2, 1)',
              letterSpacing: '-0.2px',
              marginRight: '24px'
            }}
          >
            Business Modules ({businessModules.length})
          </button>
          <button
            onClick={() => setModuleTab('technical')}
            style={{
              padding: '14px 28px',
              borderBottom: moduleTab === 'technical' ? '3px solid #475569' : '3px solid transparent',
              background: 'transparent',
              cursor: 'pointer',
              fontSize: '0.95rem',
              fontWeight: moduleTab === 'technical' ? '700' : '600',
              color: moduleTab === 'technical' ? '#475569' : '#1e3a5f',
              transition: 'all 0.3s cubic-bezier(0.4, 0, 0.2, 1)',
              letterSpacing: '-0.2px'
            }}
          >
            Technical Modules ({technicalModules.length})
          </button>
        </div>

        {/* Modules Section */}
        <div className="modules-container">
          {filteredModules.length > 0 ? (
            filteredModules
              .filter(module => {
                if (moduleTab === 'business') {
                  return businessModules.some(b => b.name === module.name);
                } else {
                  return technicalModules.some(t => t.name === module.name);
                }
              })
              .sort((a, b) => {
                // Sort ready agents first, then in-progress
                if (a.status === 'ready' && b.status !== 'ready') return -1;
                if (a.status !== 'ready' && b.status === 'ready') return 1;
                return 0;
              })
              .map((module, index) => (
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
                  {module.status && (
                    <div className="status-badge" style={{
                      position: 'absolute',
                      top: '12px',
                      right: '12px',
                      padding: '6px 12px',
                      borderRadius: '6px',
                      fontSize: '0.7rem',
                      fontWeight: '700',
                      backgroundColor: module.status === 'ready' ? '#10b981' : '#f97316',
                      color: '#ffffff',
                      boxShadow: module.status === 'ready' ? 
                        '0 4px 12px rgba(16, 185, 129, 0.25)' : 
                        '0 4px 12px rgba(249, 115, 22, 0.25)',
                      textTransform: 'uppercase',
                      letterSpacing: '0.5px',
                      border: 'none',
                      zIndex: 3
                    }}>
                      {module.status === 'ready' ? '✓ Ready' : '⚡ In Progress'}
                    </div>
                  )}
                  <div className="card-header">
                    <img src={module.icon} alt={module.name} />
                    <p>{module.name}</p>
                  </div>
                  
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