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
  
  // Executive Assistant API
  EXECUTIVE_ASSISTANT_BASE: `${API_URL}/api`,
};

export default API_CONFIG;
