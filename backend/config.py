"""
Centralized configuration management for backend
Reads from environment variables and provides validated config
"""
import os
from dotenv import load_dotenv

# Load environment variables from .env file
ENV_FILE = os.path.join(os.path.dirname(__file__), '.env')
load_dotenv(ENV_FILE, override=True)


class Config:
    """Base configuration"""
    
    # Deployment environment
    ENVIRONMENT = os.environ.get('ENVIRONMENT', 'development')
    IS_DEVELOPMENT = ENVIRONMENT == 'development'
    IS_PRODUCTION = ENVIRONMENT == 'production'
    
    # Public URL - where this instance is accessible (for OAuth redirects)
    PUBLIC_URL = os.environ.get('PUBLIC_URL')
    
    if not PUBLIC_URL:
        raise ValueError(
            'PUBLIC_URL environment variable not set!\n'
            'For localhost: http://localhost:5000\n'
            'For remote: http://agents.enableyou.co:5000'
        )
    
    # Ensure PUBLIC_URL has correct protocol
    if not (PUBLIC_URL.startswith('http://') or PUBLIC_URL.startswith('https://')):
        raise ValueError('PUBLIC_URL must start with http:// or https://')
    
    # OAuth configuration
    GOOGLE_CLIENT_ID = os.environ.get('GOOGLE_CLIENT_ID')
    GOOGLE_CLIENT_SECRET = os.environ.get('GOOGLE_CLIENT_SECRET')
    GOOGLE_REDIRECT_URI = os.environ.get(
        'GOOGLE_REDIRECT_URI',
        f'{PUBLIC_URL}/auth/google/callback'
    )
    
    if not GOOGLE_CLIENT_ID or not GOOGLE_CLIENT_SECRET:
        raise ValueError(
            'GOOGLE_CLIENT_ID and GOOGLE_CLIENT_SECRET not set!\n'
            'Get these from: https://console.developers.google.com/'
        )
    
    # LinkedIn OAuth (optional)
    LINKEDIN_CLIENT_ID = os.environ.get('LINKEDIN_CLIENT_ID')
    LINKEDIN_CLIENT_SECRET = os.environ.get('LINKEDIN_CLIENT_SECRET')
    LINKEDIN_REDIRECT_URI = os.environ.get(
        'LINKEDIN_REDIRECT_URI',
        f'{PUBLIC_URL}/auth/linkedin/callback'
    )
    
    # Twilio (optional)
    TWILIO_ACCOUNT_SID = os.environ.get('TWILIO_ACCOUNT_SID')
    TWILIO_AUTH_TOKEN = os.environ.get('TWILIO_AUTH_TOKEN')
    
    # OpenAI
    OPENAI_API_KEY = os.environ.get('OPENAI_API_KEY')
    if not OPENAI_API_KEY:
        raise ValueError('OPENAI_API_KEY environment variable not set!')
    
    # OAuth security setting (allow HTTP in development only)
    if IS_DEVELOPMENT:
        os.environ['OAUTHLIB_INSECURE_TRANSPORT'] = '1'
    else:
        os.environ['OAUTHLIB_INSECURE_TRANSPORT'] = '0'
    
    # Database - PostgreSQL required
    DATABASE_URL = os.environ.get('DATABASE_URL') or os.environ.get('DATABASE_URI')
    if not DATABASE_URL:
        raise ValueError("DATABASE_URL or DATABASE_URI environment variable is required.")
    
    @classmethod
    def print_config(cls):
        """Print current configuration (without secrets)"""
        print("=" * 70)
        print("BACKEND CONFIGURATION")
        print("=" * 70)
        print(f"Environment: {cls.ENVIRONMENT}")
        print(f"Public URL: {cls.PUBLIC_URL}")
        print(f"Google OAuth Redirect: {cls.GOOGLE_REDIRECT_URI}")
        print(f"Database: {cls.DATABASE_URL}")
        print(f"OAUTHLIB_INSECURE_TRANSPORT: {os.environ.get('OAUTHLIB_INSECURE_TRANSPORT')}")
        print("=" * 70)


# Export configuration
config = Config()
