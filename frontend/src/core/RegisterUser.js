import React, { useState, useEffect } from 'react';
import '../styles/Login.css';
import { useNavigate } from 'react-router-dom';
import { API_CONFIG } from '../config/apiConfig';
import { showToast } from './toast';

function RegisterUser() {
  const [email, setEmail] = useState('');
  const [password, setPassword] = useState('');
  const [firstName, setFirstName] = useState('');
  const [lastName, setLastName] = useState('');
  const [company, setCompany] = useState('');
  const [linkedin, setLinkedin] = useState('');
  const [shortIntro, setShortIntro] = useState('');
  const [companyIntro, setCompanyIntro] = useState('');
  const [loading, setLoading] = useState(false);
  const navigate = useNavigate();

  const bgImages = [
    `${process.env.PUBLIC_URL}/assets/background_images/pexels-googledeepmind-17483867.jpg`,
    `${process.env.PUBLIC_URL}/assets/background_images/pexels-googledeepmind-17483868.jpg`,
    `${process.env.PUBLIC_URL}/assets/background_images/pexels-googledeepmind-17483873.jpg`,
    `${process.env.PUBLIC_URL}/assets/background_images/pexels-googledeepmind-17483874.jpg`,
  ];

  const [bgIndex, setBgIndex] = useState(0);

  useEffect(() => {
    const interval = setInterval(() => {
      setBgIndex(prev => (prev + 1) % bgImages.length);
    }, 6000);
    return () => clearInterval(interval);
  }, [bgImages.length]);

  const handleRegister = async (e) => {
    e.preventDefault();
    if (!email.trim() || !password.trim()) {
      showToast('Email and password are required.', 'warning');
      return;
    }
    setLoading(true);
    try {
      const response = await fetch(`${API_CONFIG.API_URL}/register`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          password,
          first_name: firstName,
          last_name: lastName,
          email,
          company,
          linkedin,
          short_intro: shortIntro,
          company_intro: companyIntro,
        }),
      });
      const data = await response.json();
      if (response.ok) {
        showToast('Account created! Please sign in.', 'success');
        navigate('/login');
      } else {
        showToast(data.error || 'Registration failed. Please try again.', 'error');
      }
    } catch {
      showToast('Could not reach the server. Please try again.', 'error');
    } finally {
      setLoading(false);
    }
  };

  return (
    <div
      className="login-page"
      style={{
        backgroundImage: `url(${bgImages[bgIndex]})`,
        backgroundSize: 'cover',
        backgroundPosition: 'center',
        minHeight: '100vh',
        display: 'flex',
        flexDirection: 'column',
        alignItems: 'center',
        justifyContent: 'center',
        position: 'relative',
      }}
    >
      <div
        className="login-card-container"
        style={{ maxWidth: '600px', padding: '30px 40px', maxHeight: '85vh', overflowY: 'auto' }}
      >
        <div className="enable-logo" style={{ marginBottom: '20px' }}>
          Enable<span className="dot">.</span>
        </div>

        <form onSubmit={handleRegister} className="login-form">
          <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '15px', width: '100%' }}>
            <input type="text" placeholder="First Name" value={firstName} onChange={e => setFirstName(e.target.value)} required className="styled-input centered-placeholder" disabled={loading} />
            <input type="text" placeholder="Last Name" value={lastName} onChange={e => setLastName(e.target.value)} required className="styled-input centered-placeholder" disabled={loading} />
          </div>

          <input type="email" placeholder="Email Address" value={email} onChange={e => setEmail(e.target.value)} required className="styled-input centered-placeholder" disabled={loading} />
          <input type="password" placeholder="Password" value={password} onChange={e => setPassword(e.target.value)} required className="styled-input centered-placeholder" disabled={loading} />

          <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '15px', width: '100%' }}>
            <input type="text" placeholder="Company Name" value={company} onChange={e => setCompany(e.target.value)} className="styled-input centered-placeholder" disabled={loading} />
            <input type="text" placeholder="LinkedIn Profile" value={linkedin} onChange={e => setLinkedin(e.target.value)} className="styled-input centered-placeholder" disabled={loading} />
          </div>

          <input type="text" placeholder="Short Personal Intro" value={shortIntro} onChange={e => setShortIntro(e.target.value)} className="styled-input centered-placeholder" disabled={loading} />
          <textarea placeholder="Company Intro" value={companyIntro} onChange={e => setCompanyIntro(e.target.value)} className="styled-input centered-placeholder" style={{ minHeight: '80px', resize: 'vertical' }} disabled={loading} />

          <button
            type="submit"
            disabled={loading}
            style={{
              marginTop: '15px',
              width: '100%',
              padding: '14px',
              background: loading ? '#6b9fff' : '#0066FF',
              color: 'white',
              border: 'none',
              borderRadius: '12px',
              fontSize: '1.1rem',
              cursor: loading ? 'not-allowed' : 'pointer',
            }}
          >
            {loading ? 'Creating account…' : 'Create Account'}
          </button>

          <div className="new-user-link" onClick={() => !loading && navigate('/login')}>
            Already have an account? Login
          </div>
        </form>
      </div>

      <div className="footer-text">enableyou.co</div>
    </div>
  );
}

export default RegisterUser;
