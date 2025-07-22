import React from 'react';
import '../styles/Login.css';
import { useNavigate, Link } from 'react-router-dom';
import { useState } from 'react';

function Login() {
  const [username, setUsername] = useState('');
  const [password, setPassword] = useState('');
  const navigate = useNavigate();
  
  const handleLogin = async (e) => {
    e.preventDefault();
    try {
      const response = await fetch('http://localhost:5000/login', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ username, password }),
      });
      const data = await response.json();
      if (response.ok) {
        // alert('Login successful!');
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
    <div className="login-page">
      
      <div className="login">
        <div className="logo-wrapper">
          <img
            src={`${process.env.PUBLIC_URL}/assets/images/enable_logo.png`}
            alt="Enable Logo"
            className="logo"
          />
        </div>
        <h1 className="platform-title">Agents Assembly</h1>
        <form onSubmit={handleLogin}>
          <div>
            <input
              type="text"
              placeholder="Username"
              value={username}
              onChange={e => setUsername(e.target.value)}
              required
            />
          </div>
          <div>
            <input
              type="password"
              placeholder="Password"
              value={password}
              onChange={e => setPassword(e.target.value)}
              required
            />
          </div>
          <button type="submit">Login</button>
          <div style={{ marginTop: '16px' }}>
            <Link to="/register">Register User</Link>
          </div>
        </form>
      </div>
    </div>
  );
}

export default Login;