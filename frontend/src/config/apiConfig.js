/**
 * API Configuration
 * Centralizes all API URLs for the application
 * Uses environment variables for remote/local deployment switching
 */

const API_URL = process.env.REACT_APP_API_URL || 'http://localhost:5000';

export const API_CONFIG = {
  API_URL,
  
  // Auth endpoints
  GOOGLE_AUTH_START: `${API_URL}/auth/google/start`,
  GOOGLE_AUTH_CALLBACK: `${API_URL}/auth/google/callback`,
  
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
  
  // Search
  SIMPLE_SEARCH: `${API_URL}/simple_search`,

  // Search suggestions
  SEARCH_SUGGESTIONS: `${API_URL}/search_suggestions`,

  // RAG
  RAG_TEST: `${API_URL}/rag_test`,
  
  // Browser
  CHROME_HISTORY: `${API_URL}/chrome_history`,
  
  // Upload
  UPLOAD: `${API_URL}/upload`,
  
  // Connection
  TEST_CONNECTION: `${API_URL}/test-connection`,
  
  // Tools
  GET_TOOLS_LANDSCAPE: `${API_URL}/get_tools_landscape`,

  // Requirements / market research
  GENERATE_REQUIREMENTS: `${API_URL}/generate-requirements`,
  PREVIOUS_PROMPTS: `${API_URL}/previous-prompts`,
  SEARCH_GOOGLE_BUSINESSES: `${API_URL}/api/search-google-businesses`,
  ENRICH_BUSINESSES_WITH_EMAILS: `${API_URL}/api/enrich-businesses-with-emails`,
  ENRICH_BUSINESSES_WITH_LINKEDIN: `${API_URL}/api/enrich-businesses-with-linkedin`,
  EMAIL_EXTRACTION_USAGE: `${API_URL}/api/email-extraction-usage`,
  CONNECT_GOOGLE_BUSINESS: `${API_URL}/api/connect-google-business`,
  GET_GOOGLE_BUSINESS_DATA: `${API_URL}/api/get-google-business-data`,
  GET_GOOGLE_CREDENTIALS: `${API_URL}/api/get-google-credentials`,

  // Campaigns / emails
  GENERATE_EMAIL: `${API_URL}/api/generate-email`,
  SEND_BULK_EMAILS: `${API_URL}/api/send-bulk-emails`,
  GET_CAMPAIGNS: `${API_URL}/api/campaigns`,
  GET_CAMPAIGNS_STATS: `${API_URL}/api/campaigns/stats`,
  GET_CAMPAIGN_RECIPIENTS: `${API_URL}/api/campaigns/{campaignId}/recipients`,

  // Saved leads
  SAVE_PROJECT: `${API_URL}/api/save-project`,
  APPEND_PROJECT: `${API_URL}/api/append-project`,
  SCORE_LEADS: `${API_URL}/api/score-leads`,
  GET_SAVED_PROJECTS: `${API_URL}/api/get-saved-projects`,
  GET_SAVED_PROJECT_LEADS: `${API_URL}/api/get-saved-project-leads`,
  DELETE_SAVED_PROJECT: `${API_URL}/api/delete-saved-project`,
  
  // Executive Assistant API
  EXECUTIVE_ASSISTANT_BASE: `${API_URL}/api`,
};

export default API_CONFIG;
