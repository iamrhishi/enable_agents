/**
 * Centralized Agents/Modules Configuration
 *
 * Single source of truth for all agent metadata.
 * Update this file to add/modify agents - no need to change multiple files.
 */

export const AGENT_CATEGORIES = {
  BUSINESS: 'business',
  TECHNICAL: 'technical',
};

export const AGENT_STATUS = {
  READY: 'ready',
  COMING_SOON: 'coming_soon',
  IN_PROGRESS: 'in_progress',
};

/**
 * All agents configuration
 *
 * Each agent has:
 * - id: Unique identifier (used as key)
 * - name: Display name
 * - route: URL path for navigation
 * - icon: Icon filename or emoji
 * - description: Short description (2-3 lines)
 * - category: 'business' or 'technical'
 * - status: 'ready', 'coming_soon', or 'in_progress'
 * - price: Pricing display string
 * - features: Array of key features (optional)
 */
export const AGENTS = {
  marketResearch: {
    id: 'marketResearch',
    name: 'Market Research',
    route: '/market-research',
    icon: 'market-research.png',
    description: 'Discover market trends, analyze competitors, and gather customer insights to make data-driven decisions.',
    category: AGENT_CATEGORIES.BUSINESS,
    status: AGENT_STATUS.READY,
    price: '$29/month',
    features: ['Lead Generation', 'Competitor Analysis', 'Market Trends'],
  },

  salesHelper: {
    id: 'salesHelper',
    name: 'Sales Helper',
    route: '/sales-helper',
    icon: 'sales-helper.png',
    description: 'Supercharge your sales with CRM integration, and intelligent recommendations.',
    category: AGENT_CATEGORIES.BUSINESS,
    status: AGENT_STATUS.READY,
    price: '$45/month',
    features: ['CRM Integration', 'Lead Scoring', 'Sales Analytics'],
  },

  contentMarketing: {
    id: 'contentMarketing',
    name: 'Content Marketing',
    route: '/content-marketing',
    icon: 'content-marketing.png',
    description: 'Create engaging content, manage campaigns, and analyze performance across channels.',
    category: AGENT_CATEGORIES.BUSINESS,
    status: AGENT_STATUS.READY,
    price: '$35/month',
    features: ['Content Generation', 'Campaign Management', 'Analytics'],
  },

  communityNetwork: {
    id: 'communityNetwork',
    name: 'Community Network',
    route: '/community-network',
    icon: 'community-network.png',
    description: 'Build and engage your community with AI-powered networking and relationship management.',
    category: AGENT_CATEGORIES.BUSINESS,
    status: AGENT_STATUS.READY,
    price: '$25/month',
    features: ['Community Building', 'Engagement Tools', 'Network Analysis'],
  },

  eventNetworking: {
    id: 'eventNetworking',
    name: 'Event Networking',
    route: '/event-networking',
    icon: 'event-networking.png',
    description: 'Maximize event ROI with smart attendee matching and follow-up automation.',
    category: AGENT_CATEGORIES.BUSINESS,
    status: AGENT_STATUS.READY,
    price: '$30/month',
    features: ['Attendee Matching', 'Follow-up Automation', 'Event Analytics'],
  },

  executiveAssistant: {
    id: 'executiveAssistant',
    name: 'Executive Assistant',
    route: '/executive-assistant',
    icon: 'executive-assistant.png',
    description: 'Your AI-powered executive assistant for task management, reminders, and stakeholder coordination via WhatsApp.',
    category: AGENT_CATEGORIES.BUSINESS,
    status: AGENT_STATUS.READY,
    price: 'Free',
    features: ['Task Management', 'WhatsApp Integration', 'Scheduling'],
  },

  dataInsights: {
    id: 'dataInsights',
    name: 'Data Insights',
    route: '/data-insights',
    icon: 'data-insights.png',
    description: 'Transform raw data into actionable insights with AI-powered analytics and visualization.',
    category: AGENT_CATEGORIES.TECHNICAL,
    status: AGENT_STATUS.READY,
    price: '$49/month',
    features: ['Data Visualization', 'Predictive Analytics', 'Reports'],
  },

  // Hidden - placeholder only
  supplyChain: {
    id: 'supplyChain',
    name: 'Supply Chain',
    route: '/supply-chain',
    icon: 'supply-chain.png',
    description: 'Optimize your supply chain with demand forecasting and vendor management.',
    category: AGENT_CATEGORIES.TECHNICAL,
    status: AGENT_STATUS.COMING_SOON,
    hidden: true,
    price: '$55/month',
    features: ['Demand Forecasting', 'Vendor Management', 'Inventory'],
  },

  // Hidden - placeholder only
  investAgent: {
    id: 'investAgent',
    name: 'Investment Agent',
    route: '/invest-agent',
    icon: 'invest-agent.png',
    description: 'Make smarter investment decisions with AI-powered market analysis and portfolio recommendations.',
    category: AGENT_CATEGORIES.TECHNICAL,
    status: AGENT_STATUS.COMING_SOON,
    hidden: true,
    price: '$65/month',
    features: ['Market Analysis', 'Portfolio Tracking', 'Risk Assessment'],
  },

  // Hidden - placeholder only
  teamPerformance: {
    id: 'teamPerformance',
    name: 'Team Performance',
    route: '/team-performance',
    icon: 'team-performance.png',
    description: 'Track team productivity, evaluate performance, and identify areas for improvement with analytics.',
    category: AGENT_CATEGORIES.TECHNICAL,
    status: AGENT_STATUS.COMING_SOON,
    hidden: true,
    price: '$39/month',
    features: ['Performance Tracking', 'Team Analytics', 'Goal Management'],
  },

  // Hidden - placeholder only
  aiChatbot: {
    id: 'aiChatbot',
    name: 'AI Chatbot',
    route: '/ai-chatbot',
    icon: 'ai-chatbot.png',
    description: 'Deploy intelligent chatbots for customer support and lead qualification.',
    category: AGENT_CATEGORIES.TECHNICAL,
    status: AGENT_STATUS.COMING_SOON,
    hidden: true,
    price: '$40/month',
    features: ['Customer Support', 'Lead Qualification', 'Multi-channel'],
  },
};

/**
 * Helper functions
 */

// Get agent by route path
export const getAgentByRoute = (route) => {
  return Object.values(AGENTS).find(agent => agent.route === route);
};

// Get agent by name
export const getAgentByName = (name) => {
  return Object.values(AGENTS).find(
    agent => agent.name.toLowerCase() === name.toLowerCase()
  );
};

// Get agent by ID
export const getAgentById = (id) => {
  return AGENTS[id];
};

// Get all agents as array (excluding hidden)
export const getAllAgents = () => {
  return Object.values(AGENTS).filter(agent => !agent.hidden);
};

// Get agents by category (excluding hidden)
export const getAgentsByCategory = (category) => {
  return Object.values(AGENTS).filter(agent => agent.category === category && !agent.hidden);
};

// Get business modules
export const getBusinessAgents = () => {
  return getAgentsByCategory(AGENT_CATEGORIES.BUSINESS);
};

// Get technical modules
export const getTechnicalAgents = () => {
  return getAgentsByCategory(AGENT_CATEGORIES.TECHNICAL);
};

// Get ready agents only (excluding hidden)
export const getReadyAgents = () => {
  return Object.values(AGENTS).filter(agent => agent.status === AGENT_STATUS.READY && !agent.hidden);
};

// Check if agent is ready
export const isAgentReady = (agentId) => {
  const agent = AGENTS[agentId];
  return agent?.status === AGENT_STATUS.READY;
};

// Route to agent name mapping (for Header breadcrumbs)
export const ROUTE_TO_AGENT_NAME = Object.values(AGENTS).reduce((acc, agent) => {
  acc[agent.route] = agent.name;
  return acc;
}, {});

// Legacy route redirects (old path -> new path)
export const ROUTE_REDIRECTS = {
  '/requirements': '/market-research',
  '/requirements-gathering': '/market-research',
  '/campaign-dashboard': '/market-research/campaigns',
  '/datainsights': '/data-insights',
  '/aichatbot': '/ai-chatbot',
  '/event-networking-agent': '/event-networking',
  '/supply-chain-agent': '/supply-chain',
};

/**
 * Module display name to route mapping
 * Used for navigation from module cards
 * Maps the exact display names used in the UI to their routes
 */
export const MODULE_NAME_TO_ROUTE = {
  // Business modules
  'Market Research': '/market-research',
  'Executive Assistant Agent': '/executive-assistant',
  'Sales Helper Agent': '/sales-helper',
  'Content Marketing Agent': '/content-marketing',
  'Community Network': '/community-network',
  'Event Networking Agent': '/event-networking',
  'Travel Agent': '/travel-agent',
  'Invest Agent': '/invest-agent',
  'Supply Chain Agent': '/supply-chain',
  'Hiring & Onboarding': '/hiring',
  'Documents': '/documents',
  'Supplier Tracking': '/supplier-tracking',
  'Invoices': '/invoices',
  'Inventory': '/inventory',
  'Orders': '/orders',
  'Reports': '/reports',
  'Team Performance': '/team-performance',

  // Technical modules
  'Data Discovery': '/data-insights',
  'AI Chatbot': '/ai-chatbot',
  'Testing AI': '/testing-ai',
  'LLM Benchmarking': '/llm-benchmarking',
  'Data Insights': '/data-insights',
};

/**
 * Get route for a module by its display name
 * @param {string} moduleName - The display name of the module
 * @returns {string|null} - The route path or null if not found
 */
export const getRouteByModuleName = (moduleName) => {
  return MODULE_NAME_TO_ROUTE[moduleName] || null;
};

/**
 * Navigate to a module by its display name
 * Use with react-router's navigate function:
 *
 * const navigate = useNavigate();
 * const route = getRouteByModuleName(moduleName);
 * if (route) navigate(route);
 */

export default AGENTS;
