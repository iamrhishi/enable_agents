import React from 'react';
import './SkeletonLoader.css';

/**
 * Skeleton Loader Component
 *
 * Usage:
 *   <Skeleton.Text />
 *   <Skeleton.Text width="60%" />
 *   <Skeleton.Paragraph lines={3} />
 *   <Skeleton.Avatar size="lg" />
 *   <Skeleton.Card />
 *   <Skeleton.Table rows={5} cols={4} />
 */

function SkeletonBase({ className, style, ...props }) {
  return (
    <div
      className={`skeleton ${className || ''}`}
      style={style}
      aria-hidden="true"
      {...props}
    />
  );
}

function Text({ width = '100%', height = '1em' }) {
  return (
    <SkeletonBase
      className="skeleton-text"
      style={{ width, height }}
    />
  );
}

function Paragraph({ lines = 3, gap = 'var(--space-2)' }) {
  const widths = ['100%', '95%', '80%', '90%', '70%'];

  return (
    <div className="skeleton-paragraph" style={{ gap }}>
      {Array.from({ length: lines }).map((_, i) => (
        <Text key={i} width={widths[i % widths.length]} />
      ))}
    </div>
  );
}

function Avatar({ size = 'md' }) {
  const sizes = {
    sm: '32px',
    md: '40px',
    lg: '56px',
    xl: '80px'
  };

  return (
    <SkeletonBase
      className="skeleton-avatar"
      style={{
        width: sizes[size],
        height: sizes[size]
      }}
    />
  );
}

function Card({ height = '120px' }) {
  return (
    <SkeletonBase
      className="skeleton-card"
      style={{ height }}
    />
  );
}

function Table({ rows = 5, cols = 4 }) {
  return (
    <div className="skeleton-table">
      {/* Header */}
      <div className="skeleton-table-row skeleton-table-header">
        {Array.from({ length: cols }).map((_, i) => (
          <SkeletonBase key={i} className="skeleton-table-cell" />
        ))}
      </div>
      {/* Rows */}
      {Array.from({ length: rows }).map((_, rowIndex) => (
        <div key={rowIndex} className="skeleton-table-row">
          {Array.from({ length: cols }).map((_, colIndex) => (
            <SkeletonBase key={colIndex} className="skeleton-table-cell" />
          ))}
        </div>
      ))}
    </div>
  );
}

function Button({ width = '120px' }) {
  return (
    <SkeletonBase
      className="skeleton-button"
      style={{ width }}
    />
  );
}

function Input() {
  return (
    <SkeletonBase className="skeleton-input" />
  );
}

// Composite for common patterns
function CardWithContent() {
  return (
    <div className="skeleton-card-content">
      <div className="skeleton-card-header">
        <Avatar size="md" />
        <div className="skeleton-card-meta">
          <Text width="60%" />
          <Text width="40%" height="0.8em" />
        </div>
      </div>
      <Paragraph lines={2} />
    </div>
  );
}

// Export as namespace
const Skeleton = {
  Text,
  Paragraph,
  Avatar,
  Card,
  Table,
  Button,
  Input,
  CardWithContent
};

export default Skeleton;
