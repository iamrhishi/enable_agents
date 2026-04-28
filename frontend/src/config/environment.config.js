/**
 * Frontend Environment Configuration
 * 
 * This module helps identify the current environment and provides
 * environment-specific URLs for the React application.
 * 
 * The REACT_APP_API_URL is set via environment variable during build:
 * - Local workflow: generated into agent-app/.env by ./scripts/run.sh local
 * - Docker workflow: provided via docker build args / .env.docker
 * 
 * Important: React bakes environment variables into the build at compile time.
 * Changes to .env require rebuilding the app with: npm run build
 */

/**
 * Get the current API URL
 * This is set via REACT_APP_API_URL environment variable
 */
export const API_URL = process.env.REACT_APP_API_URL;

/**
 * Detect current environment
 */
export const ENVIRONMENT = process.env.NODE_ENV || 'development';

export const IS_DEVELOPMENT = ENVIRONMENT === 'development';
export const IS_PRODUCTION = ENVIRONMENT === 'production';

/**
 * Validate that API_URL is configured
 */
if (!API_URL) {
  throw new Error(
    'REACT_APP_API_URL environment variable is not set!\n' +
    'Please ensure .env file contains: REACT_APP_API_URL=<your_backend_url>\n' +
    'Examples:\n' +
    '  Development: REACT_APP_API_URL=http://localhost:5000\n' +
    '  Production: REACT_APP_API_URL=http://agents.enableyou.co:5000\n' +
    'Then rebuild with: npm run build'
  );
}

/**
 * Environment configuration object
 */
export const EnvironmentConfig = {
  API_URL,
  ENVIRONMENT,
  IS_DEVELOPMENT,
  IS_PRODUCTION,
  
  // Helper to construct auth URLs
  getAuthUrl: (provider) => {
    switch (provider) {
      case 'google':
        return `${API_URL}/auth/google/start`;
      case 'linkedin':
        return `${API_URL}/auth/linkedin/start`;
      default:
        throw new Error(`Unknown auth provider: ${provider}`);
    }
  },
  
  // Print config for debugging
  printConfig: () => {
    console.log('='.repeat(70));
    console.log('FRONTEND ENVIRONMENT CONFIGURATION');
    console.log('='.repeat(70));
    console.log(`Environment: ${ENVIRONMENT}`);
    console.log(`Backend API URL: ${API_URL}`);
    console.log(`Google Auth: ${EnvironmentConfig.getAuthUrl('google')}`);
    console.log('='.repeat(70));
  },
};

export default EnvironmentConfig;
