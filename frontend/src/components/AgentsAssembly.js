import axios from 'axios';
import React, { useState, useEffect, useRef } from 'react';
import { useNavigate } from 'react-router-dom';
import Header from '../core/Header';
import '../styles/AgentsAssembly.css';
import { API_CONFIG } from '../config/apiConfig';
import { fetchAgents } from '../agents/agentRegistry';
import { showConfirm, showAlert } from './ConfirmDialog';
import { Modal, ModalTabs } from './Modal';
import { CardGrid, StatusIndicator } from './Card';
import Select from './Select';




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

  // Live/Demo mode - read from localStorage (synced with Header and Settings)
  const [isLiveMode, setIsLiveMode] = useState(() => {
    const stored = localStorage.getItem('enableAgentsMode');
    return stored === 'live';
  });

  // Listen for mode changes from Header toggle
  useEffect(() => {
    const handleStorageChange = () => {
      const stored = localStorage.getItem('enableAgentsMode');
      setIsLiveMode(stored === 'live');
    };
    window.addEventListener('storage', handleStorageChange);
    // Also poll for changes (for same-tab updates)
    const interval = setInterval(handleStorageChange, 1000);
    return () => {
      window.removeEventListener('storage', handleStorageChange);
      clearInterval(interval);
    };
  }, []);

  const navigate = useNavigate();
  const chatHistoryRef = useRef(null);
  const carouselRef = useRef(null);

  // Carousel state
  const [carouselIndex, setCarouselIndex] = useState(0);
  const cardsPerView = 4; // Number of cards visible at once

  // Load enabled agents from the backend registry on mount
  useEffect(() => {
    fetchAgents().then((agents) => {
      if (agents.length > 0) setRegistryAgents(agents);
    });
  }, []);

  // Auto-scroll chat history to bottom when new messages arrive
  useEffect(() => {
    if (chatHistoryRef.current) {
      // Use setTimeout to ensure DOM has updated before scrolling
      setTimeout(() => {
        chatHistoryRef.current.scrollTop = chatHistoryRef.current.scrollHeight;
      }, 0);
    }
  }, [chatHistory, isBuffering]);

  // Auto-scroll chat history to bottom when new messages arrive
  useEffect(() => {
    if (chatHistoryRef.current) {
      // Use setTimeout to ensure DOM has updated before scrolling
      setTimeout(() => {
        chatHistoryRef.current.scrollTop = chatHistoryRef.current.scrollHeight;
      }, 0);
    }
  }, [chatHistory, isBuffering]);

  const businessModules = [
    {
      name: 'Executive Assistant Agent',
      icon: '/assets/icons/networking.png',
      price: 'Free',
      status: 'in-progress',
      description: 'Your AI-powered executive assistant for task management, reminders, and stakeholder coordination via WhatsApp.',
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
      description: 'Discover market trends, analyze competitors, and gather customer insights to make data-driven decisions.',
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
      description: 'Supercharge your sales with lead management, CRM integration, and intelligent sales strategy recommendations.',
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
      description: 'Create compelling content, manage campaigns, and boost your brand presence with AI-powered marketing.',
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
      description: 'Streamline recruitment, automate onboarding workflows, and build your dream team efficiently.',
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
      description: 'Organize, search, and manage documents intelligently with AI-powered document understanding.',
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
      description: 'Track vendors, manage procurement, and optimize supplier relationships for better supply chain efficiency.',
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
      description: 'Automate billing, track payments, and manage accounts receivable with intelligent invoice processing.',
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
      description: 'Visualize supply chain impact, analyze disruptions, and optimize logistics with real-time dashboards.',
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
      description: 'Track stock levels, manage warehouses, and prevent stockouts with intelligent inventory management.',
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
      description: 'Process orders efficiently, track deliveries, and manage fulfillment from a single dashboard.',
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
      description: 'Plan business trips, manage travel expenses, and coordinate team travel with AI assistance.',
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
      description: 'Build and engage your community, manage relationships, and grow customer loyalty organically.',
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
      description: 'Assess financial instruments, analyze investment opportunities, and track portfolio performance.',
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
      description: 'Generate insightful reports, visualize KPIs, and make data-driven decisions with ease.',
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
      description: 'Track team productivity, evaluate performance, and identify areas for improvement with analytics.',
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
      description: 'Automate your testing workflows with AI-powered test generation and intelligent bug detection.',
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
      description: 'Evaluate and compare LLM models with comprehensive benchmarking and performance metrics.',
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
      description: 'Explore your data, uncover hidden patterns, and generate actionable business insights.',
      keywords: ['data analysis', 'data mining', 'insights', 'data exploration'],
      businessContext: ['data analysis', 'business intelligence', 'analytics'],
      industries: ['all industries', 'technology', 'finance'],
      useCases: ['data exploration', 'business insights', 'data analysis']
    },
    { name: 'Users', icon: '/assets/icons/users.png', price: '$35/month', status: 'in-progress', description: 'Manage user accounts, permissions, and access control across your organization.' },
    { name: 'Data Security', icon: '/assets/icons/data-security.png', price: '$75/month', status: 'in-progress', description: 'Protect sensitive data with encryption, access controls, and compliance monitoring.' },
    { name: 'Alerts', icon: '/assets/icons/alerts.png', price: '$22/month', status: 'in-progress', description: 'Set up intelligent alerts for critical events and stay informed in real-time.' },
    { name: 'Notifications', icon: '/assets/icons/notifications.png', price: '$18/month', status: 'in-progress', description: 'Deliver timely notifications across email, SMS, and push channels.' },
    { name: 'Dashboards', icon: '/assets/icons/dashboards.png', price: '$45/month', status: 'in-progress', description: 'Create custom dashboards to visualize metrics and track KPIs at a glance.' },
    { name: 'AI Chatbot', icon: '/assets/icons/ai-chatbots.png', price: '$52/month', status: 'in-progress', description: 'Deploy intelligent chatbots for customer support and internal assistance.' },
    { name: 'Monitoring', icon: '/assets/icons/monitoring.png', price: '$38/month', status: 'in-progress', description: 'Monitor systems, applications, and infrastructure with real-time observability.' },
    { name: 'Analytics', icon: '/assets/icons/analytics.png', price: '$58/month', status: 'in-progress', description: 'Analyze business data with advanced analytics and machine learning insights.' },
    { name: 'Data Transformation', icon: '/assets/icons/data-transformation.png', price: '$68/month', status: 'in-progress', description: 'Transform and prepare data for analysis with automated ETL pipelines.' },
    { name: 'Integration', icon: '/assets/icons/integration.png', price: '$62/month', status: 'in-progress', description: 'Connect your tools and systems with pre-built integrations and APIs.' },
    { name: 'Automation', icon: '/assets/icons/automation.png', price: '$55/month', status: 'in-progress', description: 'Automate repetitive tasks and workflows to boost productivity.' }
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
    setCarouselIndex(0); // Reset carousel to start when filters change
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

  const handleBuyModule = async (module) => {
    console.log('Buying module:', module.name);
    const confirmPurchase = await showConfirm({
      title: `Purchase ${module.name}?`,
      message: `Price: ${module.price}\nBilling: Monthly subscription`,
      confirmLabel: 'Proceed to Checkout',
      cancelLabel: 'Cancel',
      variant: 'primary',
    });

    if (confirmPurchase) {
      await showAlert(`Redirecting to checkout for ${module.name}...`, 'Processing');
      // TODO: Implement actual checkout redirect
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
            toolNames = recData.recommendations.recommended_tools
              .map((tool) => tool.name || tool.tool_name)
              .filter(Boolean);
          }

          setRecommendedModules(toolNames);
          setDetailedReportData(recData);

          if (!toolNames.length) {
            setChatHistory((prev) => [
              ...prev,
              { type: 'system', text: 'We could not identify recommended modules from the response. Please try refining your answers.' }
            ]);
          }
        } catch (recErr) {
          setRecommendedModules([]);
          setDetailedReportData(null);
          setChatHistory((prev) => [
            ...prev.filter(msg => msg.type !== 'buffer'),
            { type: 'system', text: 'Recommendation service is currently unavailable. Please try again shortly.' }
          ]);
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
        {/* Process Map Modal */}
        <Modal
          open={showProcessMap}
          onClose={handleCloseProcessMap}
          title="Process Map"
          size="lg"
        >
          <ModalTabs
            tabs={[
              {
                id: 'visual',
                label: 'Visual',
                content: processMapData && (
                  <div className="process-map-visual">
                    <div className="process-map-info">
                      <p><strong>Industry:</strong> {processMapData.industry}</p>
                      <p><strong>Department:</strong> {processMapData.department}</p>
                      <p><strong>Responsibilities:</strong> {Array.isArray(processMapData.responsibilities) ? processMapData.responsibilities.join(', ') : processMapData.responsibilities}</p>
                    </div>
                    <div className="process-map-steps">
                      <h4>Process Steps</h4>
                      <ol>
                        {processMapData.steps.map((step, idx) => (
                          <li key={idx}>
                            <strong>{step.name}</strong>: {step.description}
                            <span className="process-step-owner">({step.owner})</span>
                          </li>
                        ))}
                      </ol>
                    </div>
                  </div>
                )
              },
              {
                id: 'json',
                label: 'JSON',
                content: (
                  <pre className="process-map-json">{JSON.stringify(processMapData, null, 2)}</pre>
                )
              }
            ]}
            activeTab={processMapTab}
            onTabChange={setProcessMapTab}
          />
        </Modal>
        <div className="page-header-row">
          <h2>Agents Assembly</h2>
          <button
            className={`stage-badge ${showChatbot ? 'stage-badge--active' : 'stage-badge--inactive'}`}
            onClick={() => setShowChatbot(!showChatbot)}
            title={showChatbot ? 'AI Assistant: Active - Click to hide' : 'AI Assistant: Inactive - Click to activate'}
            aria-pressed={showChatbot}
            aria-label={`AI Assistant ${showChatbot ? 'Active' : 'Inactive'}`}
          >
            <span className="stage-badge-icon">{showChatbot ? '🤖' : '💤'}</span>
            <span className="stage-badge-label">{showChatbot ? 'AI Active' : 'AI Off'}</span>
          </button>
        </div>

        <div className={`chatbot-section ${showChatbot ? 'chatbot-section--open' : 'chatbot-section--closed'}`}>
          <div className="chatbot-container enhanced-chatbot unified-chat">
            <div className="chat-header">
              <div className="chat-header-content">
                <span className="chat-header-icon">✨</span>
                <span className="chat-header-title">AI Assistant</span>
              </div>
              <button
                className="chat-minimize-btn"
                onClick={() => setShowChatbot(false)}
                aria-label="Minimize chat"
              >
                −
              </button>
            </div>
            <div ref={chatHistoryRef} className="chat-history" role="log" aria-live="polite" aria-label="Chat messages" style={{ maxHeight: 'calc(100vh - 500px)', minHeight: '320px', overflowY: 'auto', paddingBottom: '12px', scrollBehavior: 'smooth' }}>
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
            <div className="enhanced-input">
              <input
                id="chat-file-input"
                type="file"
                style={{ display: 'none' }}
                accept="image/*,.pdf,.doc,.docx,.xlsx,.ppt,.pptx"
                onChange={handleFileChange}
              />
              <button
                className="chat-attach-btn"
                onClick={() => document.getElementById('chat-file-input').click()}
                title="Attach file"
                aria-label="Attach file"
              >
                +
              </button>
              <input
                type="text"
                className={`chat-input enhanced${inputHighlighted ? ' highlighted' : ''}`}
                placeholder={completed ? "Conversation complete" : isBuffering ? "Thinking..." : "Message AI Assistant..."}
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
              />
              <button
                onClick={() => {
                  if (inputValue.trim()) {
                    handleEnterpriseChat(inputValue);
                    setInputValue('');
                  }
                }}
                disabled={completed || isBuffering || !inputValue.trim()}
                className={`chat-send-btn ${inputValue.trim() && !completed && !isBuffering ? 'chat-send-btn--active' : ''}`}
                title="Send message"
                aria-label="Send message"
              >
                ↑
              </button>
            </div>
          </div>
        </div>

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
            <CardGrid columns="auto" gap="md" className="modules-container recommended">
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
            </CardGrid>
            {/* Detailed Report Popup - Visual Tabular Format */}
            {showDetailedReport && detailedReportData && (
              <div className="detailed-report-popup" role="dialog" aria-modal="true" aria-labelledby="detailed-report-title">
                <div className="detailed-report-modal">
                  <button className="detailed-report-close" onClick={() => setShowDetailedReport(false)} aria-label="Close detailed report">Close</button>
                  <h2 id="detailed-report-title" className="detailed-report-title">Detailed Recommendation Report</h2>
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

        {/* Module Tabs + Filter Chips Row */}
        <div className="tabs-filters-row">
          <div className="module-tabs" role="tablist" aria-label="Module categories">
            <button
              role="tab"
              className={`module-tab module-tab--business ${moduleTab === 'business' ? 'module-tab--active' : ''}`}
              aria-selected={moduleTab === 'business'}
              aria-controls="business-modules-panel"
              onClick={() => { setModuleTab('business'); setCarouselIndex(0); }}
            >
              Business ({businessModules.length})
            </button>
            <button
              role="tab"
              className={`module-tab module-tab--technical ${moduleTab === 'technical' ? 'module-tab--active' : ''}`}
              aria-selected={moduleTab === 'technical'}
              aria-controls="technical-modules-panel"
              onClick={() => { setModuleTab('technical'); setCarouselIndex(0); }}
            >
              Technical ({technicalModules.length})
            </button>
          </div>
          <div className="filter-chips" role="group" aria-label="Filter modules">
            <Select
              value={selectedIndustry}
              onChange={(e) => setSelectedIndustry(e.target.value)}
              aria-label="Select industry"
            >
              <option value="">All Industries</option>
              <option value="Retail">Retail</option>
              <option value="Food Service">Food Service</option>
              <option value="Manufacturing">Manufacturing</option>
              <option value="Healthcare">Healthcare</option>
              <option value="Finance">Finance</option>
              <option value="Technology">Technology</option>
              <option value="Consulting">Consulting</option>
            </Select>
            <Select
              value={selectedProcess}
              onChange={(e) => setSelectedProcess(e.target.value)}
              aria-label="Select process"
            >
              <option value="">All Processes</option>
              <option value="Sales">Sales</option>
              <option value="Procurement">Procurement</option>
              <option value="HR">HR</option>
              <option value="Operations">Operations</option>
              <option value="Finance">Finance</option>
              <option value="Customer Service">Customer Service</option>
            </Select>
          </div>
        </div>

        {/* Modules Section - Infinite 3D Carousel */}
        {(() => {
          const displayModules = filteredModules
            .filter(module => {
              if (moduleTab === 'business') {
                return businessModules.some(b => b.name === module.name);
              } else {
                return technicalModules.some(t => t.name === module.name);
              }
            })
            .sort((a, b) => {
              if (a.status === 'ready' && b.status !== 'ready') return -1;
              if (a.status !== 'ready' && b.status === 'ready') return 1;
              return 0;
            });

          const total = displayModules.length;
          if (total === 0) {
            return (
              <div className="no-results">
                <h3>No modules found</h3>
                <p>Try adjusting your search or browse all available modules.</p>
                <button onClick={() => setSearchTerm('')}>Clear Search</button>
              </div>
            );
          }

          // Infinite circular navigation
          const scrollCarousel = (direction) => {
            if (direction === 'left') {
              setCarouselIndex(prev => (prev - 1 + total) % total);
            } else {
              setCarouselIndex(prev => (prev + 1) % total);
            }
          };

          // Get circular offset from center (-2, -1, 0, 1, 2)
          const getCircularOffset = (index) => {
            const activeIndex = carouselIndex;
            let offset = index - activeIndex;
            // Wrap around for circular effect
            if (offset > total / 2) offset -= total;
            if (offset < -total / 2) offset += total;
            return offset;
          };

          // Calculate 3D card style based on offset from center
          const getCardStyle = (index) => {
            const offset = getCircularOffset(index);
            const absOffset = Math.abs(offset);

            // Only show 5 cards: -2, -1, 0, 1, 2
            if (absOffset > 2) {
              return { visible: false };
            }

            // Scale: center = 1, ±1 = 0.8, ±2 = 0.65
            const scale = absOffset === 0 ? 1 : absOffset === 1 ? 0.8 : 0.65;

            // Opacity: center = 1, ±1 = 0.6, ±2 = 0.3
            const opacity = absOffset === 0 ? 1 : absOffset === 1 ? 0.6 : 0.3;

            // Z-index: center highest
            const zIndex = 100 - absOffset * 10;

            // X translation: spread cards out from center (responsive)
            const cardWidth = Math.min(380, window.innerWidth * 0.22);
            const translateX = offset * (cardWidth + 40);

            // Slight Y offset for depth
            const translateY = absOffset * 10;

            return { visible: true, scale, opacity, zIndex, translateX, translateY, offset };
          };

          return (
            <div className="carousel-3d-container">
              <button
                className="carousel-nav carousel-nav--left"
                onClick={() => scrollCarousel('left')}
                aria-label="Previous agent"
              >
                ‹
              </button>

              <div className="carousel-3d-viewport">
                <div className="carousel-3d-stage">
                  {displayModules.map((module, index) => {
                    const style = getCardStyle(index);
                    if (!style.visible) return null;

                    const isReady = module.status === 'ready';
                    const isLocked = isLiveMode && !isReady;
                    const isActive = style.offset === 0;

                    return (
                      <div
                        key={module.name}
                        className={`carousel-3d-card ${isActive ? 'carousel-3d-card--active' : ''}`}
                        onClick={() => {
                          if (isActive) {
                            handleCardClick(module.name);
                          } else {
                            setCarouselIndex(index);
                          }
                        }}
                        style={{
                          transform: `translateX(${style.translateX}px) translateY(${style.translateY}px) scale(${style.scale})`,
                          opacity: style.opacity,
                          zIndex: style.zIndex,
                        }}
                      >
                        <div className="card-inner">
                          <div className="card-header">
                            <img src={module.icon} alt={module.name} />
                            <StatusIndicator
                              status={isLocked ? 'unavailable' : (isReady ? 'ready' : 'in-progress')}
                            />
                          </div>
                          <p className="card-title">{module.name}</p>
                          <p className="card-description">
                            {module.description || 'AI-powered agent to help automate and optimize your workflows.'}
                          </p>
                          <div className="card-price">{module.price}</div>
                          <div className="card-buttons">
                            <button
                              className="try-button"
                              onClick={(e) => {
                                e.stopPropagation();
                                if (!isLocked && isActive) handleTryModule(module.name);
                              }}
                              disabled={isLocked || !isActive}
                            >
                              Try Free
                            </button>
                            <button
                              className="buy-button"
                              onClick={(e) => {
                                e.stopPropagation();
                                if (isActive) handleBuyModule(module);
                              }}
                              disabled={!isActive}
                            >
                              Buy
                            </button>
                          </div>
                        </div>
                      </div>
                    );
                  })}
                </div>
              </div>

              <button
                className="carousel-nav carousel-nav--right"
                onClick={() => scrollCarousel('right')}
                aria-label="Next agent"
              >
                ›
              </button>

              {/* Carousel dots - show max 7 with smart pagination */}
              <div className="carousel-dots">
                {(() => {
                  const maxDots = 7;
                  if (total <= maxDots) {
                    return displayModules.map((_, i) => (
                      <button
                        key={i}
                        className={`carousel-dot ${i === carouselIndex ? 'carousel-dot--active' : ''}`}
                        onClick={() => setCarouselIndex(i)}
                        aria-label={`Go to agent ${i + 1}`}
                      />
                    ));
                  }
                  // Show: first, ..., current-1, current, current+1, ..., last
                  const dots = [];
                  const current = carouselIndex;
                  const showIndices = new Set([0, current - 1, current, current + 1, total - 1].filter(i => i >= 0 && i < total));
                  let lastShown = -2;

                  for (let i = 0; i < total; i++) {
                    if (showIndices.has(i)) {
                      if (lastShown < i - 1) {
                        dots.push(<span key={`ellipsis-${i}`} className="carousel-dot-ellipsis">•••</span>);
                      }
                      dots.push(
                        <button
                          key={i}
                          className={`carousel-dot ${i === carouselIndex ? 'carousel-dot--active' : ''}`}
                          onClick={() => setCarouselIndex(i)}
                          aria-label={`Go to agent ${i + 1}`}
                        />
                      );
                      lastShown = i;
                    }
                  }
                  return dots;
                })()}
              </div>
            </div>
          );
        })()}
      </div>
    </div>
  );
}

export default AgentsAssembly;