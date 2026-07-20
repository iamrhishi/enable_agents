import React, { useState, useEffect, useCallback } from 'react';
import { useParams, useNavigate } from 'react-router-dom';
import Header from '../core/Header';
import { BackButton, EmptyState } from '../components';
import { API_CONFIG } from '../config/apiConfig';
import { authJsonHeaders } from '../core/authHeaders';
import { showToast } from '../core/toast';
import WorkflowProgress from './WorkflowProgress';
import './WorkflowRunner.css';

// Demo instance data for showcase
const DEMO_INSTANCE = {
  id: 'demo-instance-1',
  name: 'Apex Manufacturing - Aluminum Housing Sourcing',
  templateId: 'supplier-qualification',
  templateName: 'Supplier Qualification Pipeline',
  status: 'completed',
  currentStageIndex: 6,
  totalStages: 6,
  currentStage: null,
  stageStates: {
    requirement_capture: { status: 'completed', completedAt: '2026-07-15T10:30:00Z' },
    market_research: { status: 'completed', completedAt: '2026-07-16T14:00:00Z' },
    supplier_outreach: { status: 'completed', completedAt: '2026-07-17T09:00:00Z' },
    response_tracking: { status: 'completed', completedAt: '2026-07-18T16:00:00Z' },
    qualification_audit: { status: 'completed', completedAt: '2026-07-19T11:00:00Z' },
    final_selection: { status: 'completed', completedAt: '2026-07-20T14:30:00Z' },
  },
  context: {
    client_name: 'Apex Manufacturing Inc.',
    client_location: 'Detroit, Michigan, USA',
    component_type: 'CNC Machined Aluminum Housing',
    material: '6061-T6 Aluminum',
    annual_volume: '50,000 units',
    selected_supplier: 'Bharat Precision Engineering',
    quote: '$12.80/unit',
    lead_time: '6 weeks',
    backup_supplier: 'Precision Components Pvt Ltd',
  },
  createdAt: '2026-07-15T10:00:00Z',
  completedAt: '2026-07-20T14:30:00Z',
};

function WorkflowRunner() {
  const { instanceId } = useParams();
  const navigate = useNavigate();
  const [instance, setInstance] = useState(null);
  const [loading, setLoading] = useState(true);
  const [stageData, setStageData] = useState({});
  const [completing, setCompleting] = useState(false);

  // Demo mode detection
  const isDemoMode = localStorage.getItem('enableAgentsMode') !== 'live';

  const fetchInstance = useCallback(async () => {
    // Demo mode: use demo data
    if (isDemoMode && instanceId.startsWith('demo-')) {
      setInstance(DEMO_INSTANCE);
      setLoading(false);
      return;
    }

    try {
      const res = await fetch(`${API_CONFIG.BASE_URL}/workflows/instances/${instanceId}`, {
        headers: authJsonHeaders(),
      });
      const data = await res.json();
      if (data.success) {
        setInstance(data.instance);
      } else {
        showToast(data.error || 'Workflow not found', 'error');
        navigate('/workflows');
      }
    } catch (err) {
      showToast('Error loading workflow', 'error');
      navigate('/workflows');
    } finally {
      setLoading(false);
    }
  }, [instanceId, navigate, isDemoMode]);

  useEffect(() => {
    fetchInstance();
  }, [fetchInstance]);

  const handleStart = async () => {
    try {
      const res = await fetch(`${API_CONFIG.BASE_URL}/workflows/instances/${instanceId}/start`, {
        method: 'POST',
        headers: authJsonHeaders(),
      });
      const data = await res.json();
      if (data.success) {
        setInstance(data.instance);
        showToast('Workflow started', 'success');
      } else {
        showToast(data.error || 'Failed to start', 'error');
      }
    } catch (err) {
      showToast('Error starting workflow', 'error');
    }
  };

  const handleCompleteStage = async () => {
    setCompleting(true);
    try {
      const res = await fetch(`${API_CONFIG.BASE_URL}/workflows/instances/${instanceId}/complete-stage`, {
        method: 'POST',
        headers: authJsonHeaders(),
        body: JSON.stringify({ data: stageData }),
      });
      const data = await res.json();
      if (data.success) {
        setInstance(data.instance);
        setStageData({});
        if (data.instance.status === 'completed') {
          showToast('Workflow completed!', 'success');
        } else {
          showToast('Stage completed', 'success');
        }
      } else {
        showToast(data.error || 'Failed to complete stage', 'error');
      }
    } catch (err) {
      showToast('Error completing stage', 'error');
    } finally {
      setCompleting(false);
    }
  };

  const handlePause = async () => {
    try {
      const res = await fetch(`${API_CONFIG.BASE_URL}/workflows/instances/${instanceId}/pause`, {
        method: 'POST',
        headers: authJsonHeaders(),
      });
      const data = await res.json();
      if (data.success) {
        setInstance(data.instance);
        showToast('Workflow paused', 'info');
      }
    } catch (err) {
      showToast('Error pausing workflow', 'error');
    }
  };

  if (loading) {
    return (
      <>
        <Header />
        <div className="workflow-runner">
          <div className="workflow-loading">Loading workflow...</div>
        </div>
      </>
    );
  }

  if (!instance) {
    return (
      <>
        <Header />
        <div className="workflow-runner">
          <EmptyState
            icon="alert-circle"
            title="Workflow not found"
            description="This workflow may have been deleted."
            action={{ label: 'Back to Workflows', onClick: () => navigate('/workflows') }}
          />
        </div>
      </>
    );
  }

  const currentStage = instance.currentStage;
  const isCompleted = instance.status === 'completed';
  const isPending = instance.status === 'pending';

  return (
    <>
      <Header />
      <div className="workflow-runner">
        <div className="runner-header">
          <div className="runner-header-left">
            <BackButton to="/workflows" />
            <div>
              <h1>{instance.name}</h1>
              <p className="runner-template">{instance.templateName}</p>
            </div>
          </div>
          <div className="runner-status">
            <span className={`status-badge status-${instance.status}`}>
              {instance.status}
            </span>
          </div>
        </div>

        <div className="runner-content">
          <div className="runner-main">
            <WorkflowProgress instance={instance} />

            {isPending && (
              <div className="stage-panel">
                <h2>Ready to Start</h2>
                <p>This workflow has {instance.totalStages} stages to complete.</p>
                <button className="btn btn-primary" onClick={handleStart}>
                  Start Workflow
                </button>
              </div>
            )}

            {instance.status === 'running' && currentStage && (
              <div className="stage-panel">
                <div className="stage-header">
                  <h2>Stage {instance.currentStageIndex + 1}: {currentStage.name}</h2>
                  <span className="stage-agent">Agent: {currentStage.agent}</span>
                </div>
                <p className="stage-description">{currentStage.description}</p>

                {currentStage.required_inputs && currentStage.required_inputs.length > 0 && (
                  <div className="stage-inputs">
                    <h3>Required Inputs</h3>
                    <div className="input-fields">
                      {currentStage.required_inputs.map((input) => (
                        <div key={input} className="input-field">
                          <label>{input.replace(/_/g, ' ')}</label>
                          <input
                            type="text"
                            className="input"
                            value={stageData[input] || instance.context[input] || ''}
                            onChange={(e) => setStageData({ ...stageData, [input]: e.target.value })}
                            placeholder={`Enter ${input.replace(/_/g, ' ')}`}
                          />
                        </div>
                      ))}
                    </div>
                  </div>
                )}

                <div className="stage-actions">
                  <button className="btn btn-secondary" onClick={handlePause}>
                    Pause
                  </button>
                  <button
                    className="btn btn-primary"
                    onClick={handleCompleteStage}
                    disabled={completing}
                  >
                    {completing ? 'Completing...' : 'Complete Stage'}
                  </button>
                </div>
              </div>
            )}

            {instance.status === 'paused' && (
              <div className="stage-panel">
                <h2>Workflow Paused</h2>
                <p>Resume this workflow to continue from stage {instance.currentStageIndex + 1}.</p>
                <button className="btn btn-primary" onClick={handleStart}>
                  Resume Workflow
                </button>
              </div>
            )}

            {isCompleted && (
              <div className="stage-panel completed-panel">
                <h2>Workflow Complete</h2>
                <p>All {instance.totalStages} stages have been completed.</p>
                <div className="completion-summary">
                  <h3>Collected Data</h3>
                  <pre className="context-preview">
                    {JSON.stringify(instance.context, null, 2)}
                  </pre>
                </div>
                <button className="btn btn-secondary" onClick={() => navigate('/workflows')}>
                  Back to Workflows
                </button>
              </div>
            )}
          </div>

          <div className="runner-sidebar">
            <div className="sidebar-section">
              <h3>Context</h3>
              <div className="context-data">
                {Object.keys(instance.context).length === 0 ? (
                  <p className="no-data">No data collected yet</p>
                ) : (
                  Object.entries(instance.context).map(([key, value]) => (
                    <div key={key} className="context-item">
                      <span className="context-key">{key}:</span>
                      <span className="context-value">
                        {typeof value === 'object' ? JSON.stringify(value) : String(value)}
                      </span>
                    </div>
                  ))
                )}
              </div>
            </div>
          </div>
        </div>
      </div>
    </>
  );
}

export default WorkflowRunner;
