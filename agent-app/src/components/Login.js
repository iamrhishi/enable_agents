import React, { useState, useEffect } from 'react';
import '../styles/Login.css';
import { useNavigate, useLocation } from 'react-router-dom';

function Login() {
  const [email, setEmail] = useState('');
  const [password, setPassword] = useState('');
  const [step, setStep] = useState('email');
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState('');
  const navigate = useNavigate();
  const location = useLocation();

  const [bgIndex, setBgIndex] = useState(0);

  const bgImages = [
    `${process.env.PUBLIC_URL}/assets/background_images/pexels-googledeepmind-17483867.jpg`,
    `${process.env.PUBLIC_URL}/assets/background_images/pexels-googledeepmind-17483868.jpg`,
    `${process.env.PUBLIC_URL}/assets/background_images/pexels-googledeepmind-17483873.jpg`,
    `${process.env.PUBLIC_URL}/assets/background_images/pexels-googledeepmind-17483874.jpg`
  ];

  useEffect(() => {
    // Check if coming back from Google OAuth
    const searchParams = new URLSearchParams(location.search);
    if (searchParams.get('google_auth') === 'success') {
      const googleEmail = searchParams.get('email');
      localStorage.setItem('userEmail', googleEmail);
      localStorage.setItem('firstName', googleEmail.split('@')[0]);
      navigate('/agents');
    }
  }, [location, navigate]);

  useEffect(() => {
    const interval = setInterval(() => {
      setBgIndex(prev => (prev + 1) % bgImages.length);
    }, 6000);
    return () => clearInterval(interval);
  }, [bgImages.length]);

  const handleEmailSubmit = (e) => {
    e.preventDefault();
    if (email.trim()) {
      setStep('password');
    }
  };

  const handlePasswordSubmit = async (e) => {
    e.preventDefault();
    if (!password.trim()) return;
    setLoading(true);
    setError('');
    
    try {
      const apiUrl = process.env.REACT_APP_API_URL;
      console.log('[Login] API URL:', apiUrl);
      console.log('[Login] Attempting login for:', email);
      
      const response = await fetch(`${apiUrl}/login`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ email, password }),
      });
      
      console.log('[Login] Response status:', response.status);
      const data = await response.json();
      console.log('[Login] Response data:', data);
      
      if (response.ok) {
        console.log('[Login] Login successful');
        localStorage.setItem('firstName', data.username || data.first_name || 'User');
        localStorage.setItem('userEmail', email);
        navigate('/agents');
      } else {
        const errorMsg = data.error || 'Login failed';
        console.error('[Login] Error:', errorMsg);
        setError(errorMsg);
        alert(errorMsg);
      }
    } catch (error) {
      console.error('[Login] Error:', error.message);
      const errorMsg = `Connection failed: ${error.message}`;
      setError(errorMsg);
      alert(errorMsg);
    } finally {
      setLoading(false);
    }
  };

  const handleGoogleLogin = async () => {
    setLoading(true);
    setError('');
    
    try {
      const apiUrl = process.env.REACT_APP_API_URL;
      console.log('[Google Login] API URL:', apiUrl);
      console.log('[Google Login] Full endpoint:', `${apiUrl}/auth/google/start`);
      
      const response = await fetch(`${apiUrl}/auth/google/start`, {
        method: 'GET',
        headers: {
          'Accept': 'application/json'
        }
      });
      
      console.log('[Google Login] Response status:', response.status);
      console.log('[Google Login] Response ok:', response.ok);
      
      if (!response.ok) {
        throw new Error(`HTTP Error: ${response.status} ${response.statusText}`);
      }
      
      const data = await response.json();
      console.log('[Google Login] Response data:', data);
      
      if (data.auth_url) {
        console.log('[Google Login] Redirecting to:', data.auth_url);
        window.location.href = data.auth_url;
      } else {
        setError('No auth URL received from server');
        console.error('[Google Login] No auth_url in response');
      }
    } catch (error) {
      console.error('[Google Login] Error:', error.message);
      console.error('[Google Login] Full error:', error);
      const errorMsg = `Backend connection failed: ${error.message}. Make sure the backend is running on ${process.env.REACT_APP_API_URL}`;
      setError(errorMsg);
      alert(errorMsg);
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="login-page">
      <div className="login-card-container">
        <div className="login-header">
          <img 
            src={`${process.env.PUBLIC_URL}/logo192.svg`} 
            alt="Enable Logo" 
            className="login-logo"
          />
        </div>
        
        {step === 'email' && (
          <form onSubmit={handleEmailSubmit} className="login-form">
            {error && <div style={{color: '#ff6b6b', marginBottom: '10px', fontSize: '14px'}}>{error}</div>}
            <input
              type="email"
              placeholder="Email Address"
              value={email}
              onChange={e => setEmail(e.target.value)}
              required
              className="styled-input"
              autoFocus
            />
            <button type="submit" className="primary-button">
              Continue
            </button>
            <div className="form-divider">OR</div>
            <button 
              type="button" 
              onClick={handleGoogleLogin} 
              className="google-button"
              disabled={loading}
              style={{opacity: loading ? 0.6 : 1}}
            >
              <img src="/assets/icons/google.png" alt="Google" className="google-icon" />
              {loading ? 'Connecting...' : 'Continue with Google'}
            </button>
            <p className="login-footer-text">
              New to Enable? <span className="new-user-link" onClick={() => navigate('/register')}>Create account</span>
            </p>
          </form>
        )}

        {step === 'password' && (
          <form onSubmit={handlePasswordSubmit} className="login-form">
            {error && <div style={{color: '#ff6b6b', marginBottom: '10px', fontSize: '14px'}}>{error}</div>}
            <input
              type="password"
              placeholder="Password"
              value={password}
              onChange={e => setPassword(e.target.value)}
              required
              className="styled-input"
              autoFocus
            />
            <button type="submit" className="primary-button" disabled={loading}>
              Sign In
            </button>
            <button type="button" className="secondary-button" onClick={() => {setStep('email'); setError('');}} disabled={loading}>
              Back to Email
            </button>
          </form>
        )}
      </div>

      <div className="footer-text">
        <a href="https://enableyou.co/" target="_blank" rel="noopener noreferrer">enableyou.co</a>
      </div>
    </div>
  );
}

export default Login;
