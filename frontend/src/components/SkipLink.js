import React from 'react';
import './SkipLink.css';

/**
 * SkipLink Component
 *
 * Provides a skip-to-content link for keyboard users.
 * Hidden until focused, appears at top of page.
 *
 * Usage:
 *   // In App.js or layout component
 *   <SkipLink href="#main-content" />
 *
 *   // On main content
 *   <main id="main-content" tabIndex="-1">
 *     ...
 *   </main>
 */

function SkipLink({ href = '#main-content', children = 'Skip to main content' }) {
  const handleClick = (e) => {
    e.preventDefault();
    const target = document.querySelector(href);
    if (target) {
      target.focus();
      target.scrollIntoView({ behavior: 'smooth' });
    }
  };

  return (
    <a
      href={href}
      className="skip-link"
      onClick={handleClick}
    >
      {children}
    </a>
  );
}

/**
 * SkipLinks Component
 *
 * Multiple skip links for complex pages.
 *
 * Usage:
 *   <SkipLinks links={[
 *     { href: '#main-content', label: 'Skip to main content' },
 *     { href: '#navigation', label: 'Skip to navigation' },
 *     { href: '#search', label: 'Skip to search' }
 *   ]} />
 */
function SkipLinks({ links }) {
  return (
    <nav className="skip-links" aria-label="Skip links">
      {links.map((link, index) => (
        <SkipLink key={index} href={link.href}>
          {link.label}
        </SkipLink>
      ))}
    </nav>
  );
}

SkipLink.Multiple = SkipLinks;

export default SkipLink;
