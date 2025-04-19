import React from 'react';
import { Link } from 'react-router-dom';
import '../styles/Header.css';

function Header() {
  return (
    <header className="header">
      <Link to="/">
        <img
          src={`${process.env.PUBLIC_URL}/assets/images/enable_logo.jpg`}
          alt="Enable Logo"
          className="logo"
        />
      </Link>
    </header>
  );
}

export default Header;