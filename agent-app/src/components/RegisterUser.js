import React, { useState } from 'react';
import { useNavigate } from 'react-router-dom';

function RegisterUser() {
  const [username, setUsername] = useState('');
  const [password, setPassword] = useState('');
  const [firstName, setFirstName] = useState('');
  const [lastName, setLastName] = useState('');
  const [email, setEmail] = useState('');
  const [company, setCompany] = useState('');
  const navigate = useNavigate();

  const handleRegister = async (e) => {
    e.preventDefault();
    // Call your backend API to register the user
    try {
      const response = await fetch('http://localhost:5000/register', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ username, password, first_name: firstName, last_name: lastName, email, company }),
      });
      const data = await response.json();
      if (response.ok) {
        alert('User registered successfully!');
        navigate('/'); // Redirect to login page
      } else {
        alert(data.error || 'Registration failed');
      }
    } catch (error) {
      alert('Registration failed');
    }
  };

  return (
    <div className="login-page">
      <div className="login">
        <div className="logo-wrapper">
          <img
            src={`${process.env.PUBLIC_URL}/assets/images/enable_logo.jpg`}
            alt="Enable Logo"
            className="logo"
          />
        </div>
        <form onSubmit={handleRegister}>
          <div>
            <input
              type="text"
              placeholder="First Name"
              value={firstName}
              onChange={e => setFirstName(e.target.value)}
              required
            />
          </div>
          <div>
            <input
              type="text"
              placeholder="Last Name"
              value={lastName}
              onChange={e => setLastName(e.target.value)}
              required
            />
          </div>
          <div>
            <input
              type="email"
              placeholder="Email"
              value={email}
              onChange={e => setEmail(e.target.value)}
              required
            />
          </div>
          <div>
            <input
              type="text"
              placeholder="Company"
              value={company}
              onChange={e => setCompany(e.target.value)}
              required
            />
          </div>
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
          <button type="submit">Register</button>
        </form>
      </div>
    </div>
  );
}

export default RegisterUser;