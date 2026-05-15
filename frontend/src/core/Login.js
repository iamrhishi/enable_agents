import React, { useState, useEffect } from 'react';
import '../styles/Login.css';
import { useNavigate, useLocation } from 'react-router-dom';
import { API_CONFIG } from '../config/apiConfig';
import { showToast } from './toast';

function Login() {
  const [email, setEmail] = useState('');
  const [password, setPassword] = useState('');
  const [step, setStep] = useState('email');
  const [loading, setLoading] = useState(false);
  const navigate = useNavigate();
  const location = useLocation();

  const bgImages = [
    `${process.env.PUBLIC_URL}/assets/background_images/pexels-googledeepmind-17483867.jpg`,
    `${process.env.PUBLIC_URL}/assets/background_images/pexels-googledeepmind-17483868.jpg`,
    `${process.env.PUBLIC_URL}/assets/background_images/pexels-googledeepmind-17483873.jpg`,
    `${process.env.PUBLIC_URL}/assets/background_images/pexels-googledeepmind-17483874.jpg`,
  ];

  const [bgIndex, setBgIndex] = useState(0);

  useEffect(() => {
    const searchParams = new URLSearchParams(location.search);
    const oauthError = searchParams.get('error');
    if (oauthError) {
      showToast(`Google sign-in failed: ${oauthError}`, 'error');
    }
    if (searchParams.get('google_auth') === 'success') {
      const googleEmail = searchParams.get('email');
      const sess = searchParams.get('session_token');
      if (googleEmail) {
        localStorage.setItem('userEmail', googleEmail);
        localStorage.setItem('firstName', googleEmail.split('@')[0]);
      }
      if (sess) {
        localStorage.setItem('sessionToken', sess);
      }
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
    if (email.trim()) setStep('password');
  };

  const handlePasswordSubmit = async (e) => {
    e.preventDefault();
    if (!password.trim()) return;
    setLoading(true);
    try {
      const response = await fetch(`${API_CONFIG.API_URL}/login`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ email, password }),
      });
      const data = await response.json();
      if (response.ok) {
        if (data.session_token) {
          localStorage.setItem('sessionToken', data.session_token);
        }
        localStorage.setItem('firstName', data.username || data.first_name || 'User');
        localStorage.setItem('userEmail', email);
        navigate('/agents');
      } else {
        showToast(data.error || 'Login failed. Please check your credentials.', 'error');
      }
    } catch {
      showToast('Could not reach the server. Please try again.', 'error');
    } finally {
      setLoading(false);
    }
  };

  const handleGoogleLogin = async () => {
    setLoading(true);
    try {
      const response = await fetch(`${API_CONFIG.API_URL}/auth/google/start`);
      if (!response.ok) {
        const err = await response.json().catch(() => ({}));
        showToast(err.error || 'Google sign-in is not configured.', 'error');
        return;
      }
      const data = await response.json();
      if (data.auth_url) {
        window.location.href = data.auth_url;
      } else {
        showToast('Google sign-in is not available right now.', 'warning');
      }
    } catch {
      showToast('Could not reach the server. Please try again.', 'error');
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
            <input
              type="email"
              placeholder="Email Address"
              value={email}
              onChange={e => setEmail(e.target.value)}
              required
              className="styled-input"
              autoFocus
              disabled={loading}
            />
            <button type="submit" className="primary-button" disabled={loading}>
              Continue
            </button>
            <div className="form-divider">OR</div>
            <button
              type="button"
              onClick={handleGoogleLogin}
              className="google-button"
              disabled={loading}
            >
              <img src="/assets/icons/google.png" alt="Google" className="google-icon" />
              {loading ? 'Redirecting…' : 'Continue with Google'}
            </button>
            <p className="login-footer-text">
              New to Enable?{' '}
              <span className="new-user-link" onClick={() => navigate('/register')}>
                Create account
              </span>
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
              disabled={loading}
            />
            <button type="submit" className="primary-button" disabled={loading}>
              {loading ? 'Signing in…' : 'Sign In'}
            </button>
            <button
              type="button"
              className="secondary-button"
              onClick={() => setStep('email')}
              disabled={loading}
            >
              Back to Email
            </button>
          </form>
        )}
      </div>

      <div className="footer-text">
        <a href="https://enableyou.co/" target="_blank" rel="noopener noreferrer">
          enableyou.co
        </a>
      </div>
    </div>
  );
}

export default Login;
