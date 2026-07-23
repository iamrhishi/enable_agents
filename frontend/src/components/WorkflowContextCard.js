/**
 * WorkflowContextCard Component
 *
 * Displays data from previous workflow stages to show data lineage and flow.
 * Shows users how current stage builds on previous work.
 */

import React from 'react';
import './WorkflowContextCard.css';

function WorkflowContextCard({ context, currentStageId }) {
  if (!context || Object.keys(context).length === 0) {
    return null;
  }

  // Stage-specific context rendering
  const renderContextForStage = () => {
    switch (currentStageId) {
      case 'document_analysis':
        return (
          <div className="wf-context-content">
            <div className="wf-context-header">
              <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
                <path d="M13 2L3 14h9l-1 8 10-12h-9l1-8z" />
              </svg>
              <h4>Context from Supplier Discovery</h4>
            </div>
            <div className="wf-context-grid">
              {context.client_name && (
                <div className="wf-context-item">
                  <span className="wf-context-label">Client:</span>
                  <span className="wf-context-value">{context.client_name}</span>
                </div>
              )}
              {context.component_type && (
                <div className="wf-context-item">
                  <span className="wf-context-label">Looking for:</span>
                  <span className="wf-context-value">{context.component_type}</span>
                </div>
              )}
              {context.location && (
                <div className="wf-context-item">
                  <span className="wf-context-label">Location:</span>
                  <span className="wf-context-value">{context.location}</span>
                </div>
              )}
              {context.industry && (
                <div className="wf-context-item">
                  <span className="wf-context-label">Industry:</span>
                  <span className="wf-context-value">{context.industry}</span>
                </div>
              )}
              {context.businesses_found && (
                <div className="wf-context-item">
                  <span className="wf-context-label">Suppliers Found:</span>
                  <span className="wf-context-value wf-badge-success">{context.businesses_found} suppliers</span>
                </div>
              )}
            </div>
            {context.top_businesses && context.top_businesses.length > 0 && (
              <div className="wf-context-businesses">
                <span className="wf-context-label">Top Suppliers:</span>
                <div className="wf-business-chips">
                  {context.top_businesses.map((business, idx) => (
                    <span key={idx} className="wf-business-chip">{business}</span>
                  ))}
                </div>
              </div>
            )}
          </div>
        );

      case 'rfq_outreach':
        return (
          <div className="wf-context-content">
            <div className="wf-context-header">
              <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
                <path d="M13 2L3 14h9l-1 8 10-12h-9l1-8z" />
              </svg>
              <h4>Context from Previous Stages</h4>
            </div>
            <div className="wf-context-grid">
              {context.businesses_found && (
                <div className="wf-context-item">
                  <span className="wf-context-label">Suppliers Identified:</span>
                  <span className="wf-context-value wf-badge-success">{context.businesses_found}</span>
                </div>
              )}
              {context.documents_analyzed && (
                <div className="wf-context-item">
                  <span className="wf-context-label">Documents Analyzed:</span>
                  <span className="wf-context-value wf-badge-info">{context.documents_analyzed}</span>
                </div>
              )}
              {context.key_findings && context.key_findings.length > 0 && (
                <div className="wf-context-item wf-context-full">
                  <span className="wf-context-label">Key Findings:</span>
                  <ul className="wf-context-list">
                    {context.key_findings.slice(0, 3).map((finding, idx) => (
                      <li key={idx}>{finding}</li>
                    ))}
                  </ul>
                </div>
              )}
            </div>
          </div>
        );

      case 'response_analysis':
        return (
          <div className="wf-context-content">
            <div className="wf-context-header">
              <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
                <path d="M13 2L3 14h9l-1 8 10-12h-9l1-8z" />
              </svg>
              <h4>Context from RFQ Outreach</h4>
            </div>
            <div className="wf-context-grid">
              {context.emails_sent && (
                <div className="wf-context-item">
                  <span className="wf-context-label">Emails Sent:</span>
                  <span className="wf-context-value wf-badge-success">{context.emails_sent}</span>
                </div>
              )}
              {context.total_recipients && (
                <div className="wf-context-item">
                  <span className="wf-context-label">Recipients:</span>
                  <span className="wf-context-value">{context.total_recipients} suppliers</span>
                </div>
              )}
              {context.email_subject && (
                <div className="wf-context-item wf-context-full">
                  <span className="wf-context-label">Subject:</span>
                  <span className="wf-context-value">{context.email_subject}</span>
                </div>
              )}
            </div>
          </div>
        );

      case 'qualification_audit':
        return (
          <div className="wf-context-content">
            <div className="wf-context-header">
              <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
                <path d="M13 2L3 14h9l-1 8 10-12h-9l1-8z" />
              </svg>
              <h4>Context from Response Analysis</h4>
            </div>
            <div className="wf-context-grid">
              {context.leads_analyzed && (
                <div className="wf-context-item">
                  <span className="wf-context-label">Responses Analyzed:</span>
                  <span className="wf-context-value wf-badge-info">{context.leads_analyzed}</span>
                </div>
              )}
              {context.matched_prospects && (
                <div className="wf-context-item">
                  <span className="wf-context-label">Qualified Suppliers:</span>
                  <span className="wf-context-value wf-badge-success">{context.matched_prospects}</span>
                </div>
              )}
              {context.top_match && (
                <div className="wf-context-item">
                  <span className="wf-context-label">Top Match:</span>
                  <span className="wf-context-value">{context.top_match}</span>
                </div>
              )}
            </div>
          </div>
        );

      case 'selection_tasks':
        return (
          <div className="wf-context-content">
            <div className="wf-context-header">
              <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
                <path d="M13 2L3 14h9l-1 8 10-12h-9l1-8z" />
              </svg>
              <h4>Context from Qualification Audit</h4>
            </div>
            <div className="wf-context-grid">
              {context.total_audited && (
                <div className="wf-context-item">
                  <span className="wf-context-label">Suppliers Audited:</span>
                  <span className="wf-context-value wf-badge-info">{context.total_audited}</span>
                </div>
              )}
              {context.passed_count && (
                <div className="wf-context-item">
                  <span className="wf-context-label">Passed Audit:</span>
                  <span className="wf-context-value wf-badge-success">{context.passed_count}</span>
                </div>
              )}
              {context.top_scorer && (
                <div className="wf-context-item">
                  <span className="wf-context-label">Top Performer:</span>
                  <span className="wf-context-value">{context.top_scorer}</span>
                </div>
              )}
              {context.audit_result && (
                <div className="wf-context-item wf-context-full">
                  <span className="wf-context-label">Result:</span>
                  <span className="wf-context-value">{context.audit_result}</span>
                </div>
              )}
            </div>
          </div>
        );

      default:
        // Generic context display
        return (
          <div className="wf-context-content">
            <div className="wf-context-header">
              <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
                <path d="M13 2L3 14h9l-1 8 10-12h-9l1-8z" />
              </svg>
              <h4>Context from Previous Stages</h4>
            </div>
            <div className="wf-context-grid">
              {Object.entries(context).slice(0, 6).map(([key, value]) => (
                <div key={key} className="wf-context-item">
                  <span className="wf-context-label">{key.replace(/_/g, ' ')}:</span>
                  <span className="wf-context-value">
                    {Array.isArray(value) ? value.join(', ') : String(value)}
                  </span>
                </div>
              ))}
            </div>
          </div>
        );
    }
  };

  return (
    <div className="workflow-context-card">
      {renderContextForStage()}
    </div>
  );
}

export default WorkflowContextCard;
