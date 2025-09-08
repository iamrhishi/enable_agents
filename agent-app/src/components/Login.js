import React, { useState, useEffect } from 'react';
import '../styles/Login.css';
import { useNavigate } from 'react-router-dom';

function Login() {
  const [username, setUsername] = useState('');
  const [password, setPassword] = useState('');
  const [bgIndex, setBgIndex] = useState(0);
  const [step, setStep] = useState('username');
  const navigate = useNavigate();

  // List of background images
  const bgImages = [
    `${process.env.PUBLIC_URL}/assets/background_images/pexels-googledeepmind-17483867.jpg`,
    `${process.env.PUBLIC_URL}/assets/background_images/pexels-googledeepmind-17483868.jpg`,
    `${process.env.PUBLIC_URL}/assets/background_images/pexels-googledeepmind-17483873.jpg`,
    `${process.env.PUBLIC_URL}/assets/background_images/pexels-googledeepmind-17483874.jpg`
  ];

  // Cycle background images every 6 seconds
  useEffect(() => {
    const interval = setInterval(() => {
      setBgIndex(prev => (prev + 1) % bgImages.length);
    }, 6000);
    return () => clearInterval(interval);
  }, [bgImages.length]);

  const handleUsernameSubmit = (e) => {
    e.preventDefault();
    if (username.trim()) {
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
        body: JSON.stringify({ username, password }),
      });
      const data = await response.json();
      if (response.ok) {
        localStorage.setItem('firstName', data.username);
        navigate('/agents');
      } else {
        alert(data.error || 'Login failed');
      }
    } catch (error) {
      alert('Login failed');
    }
  };

  return (
    <div className="login-page" style={{
      backgroundImage: `url(${bgImages[bgIndex]})`,
      transition: 'background-image 2s ease-in-out',
      backgroundSize: 'cover',
      backgroundPosition: 'center',
      minHeight: '100vh',
      display: 'flex',
      alignItems: 'center',
      justifyContent: 'center',
      position: 'relative'
    }}>
      <div className="login-bg-overlay" />
      <div className="login-card enhanced-card" style={{ minWidth: 340, maxWidth: 400, width: '100%', padding: '40px 32px 32px 32px', display: 'flex', flexDirection: 'column', alignItems: 'center', gap: '18px' }}>
        <div className="logo-wrapper" style={{ marginBottom: '18px' }}>
          <img
            src={`${process.env.PUBLIC_URL}/assets/images/enable_logo.png`}
            alt="Enable Logo"
            className="logo"
            style={{ height: '56px' }}
          />
        </div>
  {/* <div className="platform-title">Agents Assembly Platform</div> */}
        {step === 'username' && (
          <form onSubmit={handleUsernameSubmit} style={{ width: '100%' }}>
            <input
              type="text"
              placeholder="Username"
              value={username}
              onChange={e => setUsername(e.target.value)}
              required
              className="login-input"
              style={{ fontSize: '1.08rem', padding: '14px', borderRadius: '10px', width: '100%' }}
              autoFocus
            />
            <div className="register-link" style={{ textAlign: 'center', marginTop: '8px' }}>
              <span style={{ color: '#2a5298', fontWeight: 500, cursor: 'pointer' }}>New User?</span>
            </div>
          </form>
        )}
        {step === 'password' && (
          <form onSubmit={handlePasswordSubmit} style={{ width: '100%' }}>
            <input
              type="password"
              placeholder="Password"
              value={password}
              onChange={e => setPassword(e.target.value)}
              required
              className="login-input"
              style={{ fontSize: '1.08rem', padding: '14px', borderRadius: '10px', width: '100%' }}
              autoFocus
            />
            <div className="register-link" style={{ textAlign: 'center', marginTop: '8px' }}>
              <button
                type="button"
                onClick={() => { setStep('username'); setPassword(''); }}
                style={{
                  background: 'none',
                  border: 'none',
                  color: '#2a5298',
                  fontWeight: 500,
                  cursor: 'pointer',
                  fontSize: '1rem',
                  padding: 0
                }}
              >Back</button>
            </div>
          </form>
        )}
      </div>
      <div style={{ position: 'absolute', bottom: 24, left: 0, right: 0, textAlign: 'center', zIndex: 2 }}>
        <a href="https://enableyou.co" target="_blank" rel="noopener noreferrer" style={{ color: '#191a1bff', fontWeight: 400, fontSize: '1.05rem', textDecoration: 'none', letterSpacing: '0.02em' }}>enableyou.co</a>
      </div>
    </div>
  );
}

export default Login;