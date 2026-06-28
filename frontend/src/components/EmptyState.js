import React from 'react';
import './EmptyState.css';

/**
 * Empty State Component
 *
 * Usage:
 *   <EmptyState
 *     icon={<IconComponent />}
 *     title="No results found"
 *     description="Try adjusting your search or filters"
 *     action={{ label: "Clear filters", onClick: handleClear }}
 *   />
 */

// Default icons
const defaultIcons = {
  empty: (
    <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.5">
      <path strokeLinecap="round" strokeLinejoin="round" d="M20 13V6a2 2 0 00-2-2H6a2 2 0 00-2 2v7m16 0v5a2 2 0 01-2 2H6a2 2 0 01-2-2v-5m16 0h-2.586a1 1 0 00-.707.293l-2.414 2.414a1 1 0 01-.707.293h-3.172a1 1 0 01-.707-.293l-2.414-2.414A1 1 0 006.586 13H4" />
    </svg>
  ),
  search: (
    <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.5">
      <path strokeLinecap="round" strokeLinejoin="round" d="M21 21l-6-6m2-5a7 7 0 11-14 0 7 7 0 0114 0z" />
    </svg>
  ),
  document: (
    <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.5">
      <path strokeLinecap="round" strokeLinejoin="round" d="M9 12h6m-6 4h6m2 5H7a2 2 0 01-2-2V5a2 2 0 012-2h5.586a1 1 0 01.707.293l5.414 5.414a1 1 0 01.293.707V19a2 2 0 01-2 2z" />
    </svg>
  ),
  message: (
    <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.5">
      <path strokeLinecap="round" strokeLinejoin="round" d="M8 12h.01M12 12h.01M16 12h.01M21 12c0 4.418-4.03 8-9 8a9.863 9.863 0 01-4.255-.949L3 20l1.395-3.72C3.512 15.042 3 13.574 3 12c0-4.418 4.03-8 9-8s9 3.582 9 8z" />
    </svg>
  ),
  data: (
    <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.5">
      <path strokeLinecap="round" strokeLinejoin="round" d="M4 7v10c0 2.21 3.582 4 8 4s8-1.79 8-4V7M4 7c0 2.21 3.582 4 8 4s8-1.79 8-4M4 7c0-2.21 3.582-4 8-4s8 1.79 8 4m0 5c0 2.21-3.582 4-8 4s-8-1.79-8-4" />
    </svg>
  )
};

function EmptyState({
  icon,
  iconType = 'empty',
  title = 'No data yet',
  description,
  action,
  size = 'md'
}) {
  const iconElement = icon || defaultIcons[iconType] || defaultIcons.empty;

  return (
    <div className={`empty-state empty-state-${size}`}>
      <div className="empty-state-icon">
        {iconElement}
      </div>
      <h3 className="empty-state-title">{title}</h3>
      {description && (
        <p className="empty-state-description">{description}</p>
      )}
      {action && (
        <button
          className="empty-state-action btn btn-primary"
          onClick={action.onClick}
        >
          {action.label}
        </button>
      )}
    </div>
  );
}

// Preset variants
EmptyState.NoResults = (props) => (
  <EmptyState
    iconType="search"
    title="No results found"
    description="Try adjusting your search or filters"
    {...props}
  />
);

EmptyState.NoDocuments = (props) => (
  <EmptyState
    iconType="document"
    title="No documents"
    description="Upload a document to get started"
    {...props}
  />
);

EmptyState.NoMessages = (props) => (
  <EmptyState
    iconType="message"
    title="No messages yet"
    description="Start a conversation to see messages here"
    {...props}
  />
);

EmptyState.NoData = (props) => (
  <EmptyState
    iconType="data"
    title="No data available"
    description="Data will appear here once available"
    {...props}
  />
);

export default EmptyState;
