/**
 * API Configuration
 * Centralizes all API URLs for the application
 * Uses environment variables for remote/local deployment switching
 */

// API Base URL - must be set via REACT_APP_API_URL environment variable
const API_URL = process.env.REACT_APP_API_URL;

if (!API_URL) {
  throw new Error('REACT_APP_API_URL environment variable is not set. Please configure it in .env file.');
}

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
  
  // Browser
  CHROME_HISTORY: `${API_URL}/chrome_history`,
  
  // Upload
  UPLOAD: `${API_URL}/upload`,
  
  // Connection
  TEST_CONNECTION: `${API_URL}/test-connection`,
  
  // Tools
  GET_TOOLS_LANDSCAPE: `${API_URL}/get_tools_landscape`,
  
  // Executive Assistant API
  EXECUTIVE_ASSISTANT_BASE: `${API_URL}/api`,
};

export default API_CONFIG;
