/**
 * API Configuration
 * Centralizes all API URLs for the application
 * Uses environment variables for remote/local deployment switching
 */

// API Base URL - uses environment variable with fallback to localhost
const API_URL = process.env.REACT_APP_API_URL || 'http://localhost:5000';

// Helper function to get API URL (useful for components not using export)
export const getApiUrl = () => API_URL;

export const API_CONFIG = {
  API_URL,
  
  // Auth endpoints
  GOOGLE_AUTH_START: `${API_URL}/auth/google/start`,
  GOOGLE_AUTH_CALLBACK: `${API_URL}/auth/google/callback`,
  LOGIN: `${API_URL}/login`,
  REGISTER: `${API_URL}/register`,
  
  // Business search
  SEARCH_BUSINESSES: `${API_URL}/search_businesses`,
  
  // Chat endpoints
  ENTERPRISE_CHAT: `${API_URL}/enterprise_chat`,
  CHAT_API: `${API_URL}/chat_api`,
  RECOMMEND_AGENTS: `${API_URL}/recommend_agents`,
  
  // File operations
  CHECK_EXISTING_FILE: `${API_URL}/check_existing_file`,
  SAVE_JSON_FILE: `${API_URL}/save_json_file`,
  LOAD_JSON_FILE: `${API_URL}/load_json_file`,
  FILE_TO_JSON_CONVERT: `${API_URL}/file_to_json_convert`,
  SAVE_USER_FAVORITE: `${API_URL}/save_user_favorite`,
  GET_USER_FAVORITES: `${API_URL}/get_user_favorites`,
  REMOVE_USER_FAVORITE: `${API_URL}/remove_user_favorite`,
  
  // Search
  SIMPLE_SEARCH: `${API_URL}/simple_search`,
  SEARCH_SUGGESTIONS: `${API_URL}/search_suggestions`,
  
  // Browser
  CHROME_HISTORY: `${API_URL}/chrome_history`,
  
  // Upload
  UPLOAD: `${API_URL}/upload`,
  
  // Connection
  TEST_CONNECTION: `${API_URL}/test-connection`,
  
  // Tools
  GET_TOOLS_LANDSCAPE: `${API_URL}/get_tools_landscape`,
  
  // Content Marketing
  CONTENT_MARKETING_PROJECTS: `${API_URL}/api/content-marketing/projects`,
  CONTENT_MARKETING_UPLOAD: `${API_URL}/api/content-marketing/documents/upload`,
  CONTENT_MARKETING_GENERATE: `${API_URL}/api/content-marketing/generate-content`,
  CONTENT_MARKETING_CHAT: `${API_URL}/api/content-marketing/chat`,
  
  // WhatsApp
  SEND_WHATSAPP_REMINDER: `${API_URL}/send-whatsapp-reminder`,
  
  // RAG
  RAG_TEST: `${API_URL}/rag_test`,
  
  // Requirements Gathering
  GENERATE_REQUIREMENTS: `${API_URL}/generate-requirements`,
  SEARCH_GOOGLE_BUSINESSES: `${API_URL}/search-google-businesses`,
  ENRICH_BUSINESSES_WITH_EMAILS: `${API_URL}/enrich-businesses-with-emails`,
  ENRICH_BUSINESSES_WITH_LINKEDIN: `${API_URL}/enrich-businesses-with-linkedin`,
  GET_CAMPAIGNS: `${API_URL}/get-campaigns`,
  GENERATE_EMAIL: `${API_URL}/api/generate-email`,
  GET_GOOGLE_BUSINESS_DATA: `${API_URL}/get-google-business-data`,
  PREVIOUS_PROMPTS: `${API_URL}/previous-prompts`,
  GET_GOOGLE_CREDENTIALS: `${API_URL}/get-google-credentials`,
  EMAIL_EXTRACTION_USAGE: `${API_URL}/email-extraction-usage`,
  CONNECT_GOOGLE_BUSINESS: `${API_URL}/connect-google-business`,
  SEND_BULK_EMAILS: `${API_URL}/send-bulk-emails`,
  GET_CAMPAIGNS_STATS: `${API_URL}/api/campaigns/stats`,
  GET_CAMPAIGN_RECIPIENTS: `${API_URL}/api/campaigns/{campaignId}/recipients`,
  GET_SAVED_PROJECTS: `${API_URL}/api/saved-projects`,
  GET_SAVED_PROJECT_LEADS: `${API_URL}/api/saved-projects`,
  
  // Executive Assistant API
  EXECUTIVE_ASSISTANT_BASE: `${API_URL}/api`,
  APPEND_PROJECT: `${API_URL}/api/append-project`,
  SAVE_PROJECT: `${API_URL}/api/save-project`,
};

export default API_CONFIG;
