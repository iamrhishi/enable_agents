import React, { useState, useEffect } from 'react';
import '../styles/Login.css';
import { useNavigate, useLocation } from 'react-router-dom';
import { API_CONFIG } from '../config/apiConfig';

function Login() {
  const [email, setEmail] = useState('');
  const [password, setPassword] = useState('');
  const [step, setStep] = useState('email');
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
    try {
      const response = await fetch(`${API_CONFIG.API_URL}/login`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ email, password }),
      });
      const data = await response.json();
      if (response.ok) {
        localStorage.setItem('firstName', data.username || data.first_name || 'User');
        localStorage.setItem('userEmail', email);
        navigate('/agents');
      } else {
        alert(data.error || 'Login failed');
      }
    } catch (error) {
      alert('Login failed');
    }
  };

  const handleGoogleLogin = async () => {
    try {
      const response = await fetch(`${API_CONFIG.API_URL}/auth/google/start`, {
        method: 'GET'
      });
      const data = await response.json();
      if (data.auth_url) {
        window.location.href = data.auth_url;
      }
    } catch (error) {
      alert('Failed to initiate Google Login');
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
            <button type="button" onClick={handleGoogleLogin} className="google-button">
              <img src="/assets/icons/google.png" alt="Google" className="google-icon" />
              Continue with Google
            </button>
            <p className="login-footer-text">
              New to Enable? <span className="new-user-link" onClick={() => navigate('/register')}>Create account</span>
            </p>
          </form>
        )}

        {step === 'password' && (
          <form onSubmit={handlePasswordSubmit} className="login-form">
            <input
              type="password"
              placeholder="Password"
              value={password}
              onChange={e => setPassword(e.target.value)}
              required
              className="styled-input"
              autoFocus
            />
            <button type="submit" className="primary-button">
              Sign In
            </button>
            <button type="button" className="secondary-button" onClick={() => setStep('email')}>
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
