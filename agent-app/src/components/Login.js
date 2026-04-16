import React, { useState, useEffect } from 'react';
import '../styles/Login.css';
import { useNavigate, useLocation } from 'react-router-dom';

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
      const response = await fetch('http://localhost:5000/login', {
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
      const response = await fetch('http://localhost:5000/auth/google/start', {
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
    <div className="login-page" style={{
      backgroundImage: `url(${bgImages[bgIndex]})`,
      backgroundSize: 'cover',
      backgroundPosition: 'center',
      minHeight: '100vh',
      display: 'flex',
      flexDirection: 'column',
      alignItems: 'center',
      justifyContent: 'center',
      position: 'relative'
    }}>
      <div className="login-card-container">
        <div className="enable-logo">
          Enable<span className="dot">.</span>
        </div>
        
        {step === 'email' && (
          <form onSubmit={handleEmailSubmit} className="login-form">
            <input
              type="text"
              placeholder="Email Address"
              value={email}
              onChange={e => setEmail(e.target.value)}
              required
              className="styled-input centered-placeholder"
              autoFocus
            />
            <div className="new-user-link" onClick={() => navigate('/register')}>
              New User?
            </div>
            
            <div style={{ margin: '20px 0', color: '#999', fontSize: '14px', width: '100%', textAlign: 'center' }}>
              OR
            </div>
            <button type="button" onClick={handleGoogleLogin} className="styled-input" style={{ backgroundColor: '#fff', color: '#333', cursor: 'pointer', display: 'flex', alignItems: 'center', justifyContent: 'center', gap: '10px' }}>
              <img src="https://upload.wikimedia.org/wikipedia/commons/c/c1/Google_%22G%22_logo.svg" alt="Google logo" style={{width: '20px'}}/>
              Sign in with Google
            </button>
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
              className="styled-input centered-placeholder"
              autoFocus
            />
            <button type="submit" className="login-button" style={{marginTop: '15px', display: 'none'}}>Login</button>
            <div className="new-user-link" onClick={() => setStep('email')}>
              Back
            </div>
          </form>
        )}
      </div>

      <div className="footer-text">enableyou.co</div>
    </div>
  );
}

export default Login;
