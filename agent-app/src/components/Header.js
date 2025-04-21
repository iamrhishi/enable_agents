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
      <div className="header-icons">
        <img src="/assets/icons/user.png" alt="User" className="icon" />
        <img src="/assets/icons/cart.png" alt="Cart" className="icon" />
      </div>
    </header>
  );
}

export default Header;