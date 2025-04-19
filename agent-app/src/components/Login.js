import React from 'react';
import Header from './Header';
import '../styles/Login.css';

function Login() {
  const handleLogin = (e) => {
    e.preventDefault();
    console.log('Login submitted');
  };

  return (
    <div className="login-page">
      <Header />
      <div className="login">
        <h2>Login</h2>
        <form onSubmit={handleLogin}>
          <div>
            {/* <label>Username:</label> */}
            <input type="text" placeholder="Username" required />
          </div>
          <div>
            {/* <label>Password:</label> */}
            <input type="password" placeholder="Password" required />
          </div>
          <button type="submit">Login</button>
        </form>
      </div>
    </div>
  );
}

export default Login;