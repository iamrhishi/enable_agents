import React, { useState, useEffect, useCallback } from 'react';
import Header from '../core/Header';
import { BackButton, EmptyState, ProjectSelector, ProjectGate } from '../components';
import { API_CONFIG } from '../config/apiConfig';
import { authJsonHeaders } from '../core/authHeaders';
import { showToast } from '../core/toast';
import { useSelectedProjectId } from '../hooks/useSelectedProjectId';
import './WorkflowsPage.css';

// Demo data for showcasing workflows without backend
const DEMO_TEMPLATES = [
  {
    id: 'supplier-qualification',
    name: 'Supplier Qualification Pipeline',
    description: 'End-to-end workflow for qualifying suppliers for OEM requirements. Covers requirement capture, supplier research, RFQ management, and final selection.',
    category: 'procurement',
    icon: 'truck',
    stageCount: 6,
    isSystem: true,
  },
  {
    id: 'lead-nurture',
    name: 'Lead Nurturing',
    description: 'Multi-touch campaign to convert leads into customers through personalized outreach.',
    category: 'marketing',
    icon: 'users',
    stageCount: 4,
    isSystem: true,
  },
  {
    id: 'market-launch',
    name: 'New Market Launch',
    description: 'Complete workflow for launching into a new market: research, content creation, and outreach.',
    category: 'marketing',
    icon: 'rocket',
    stageCount: 3,
    isSystem: true,
  },
  {
    id: 'vendor-evaluation',
    name: 'Vendor Evaluation',
    description: 'Research and evaluate potential vendors/suppliers for your business needs.',
    category: 'procurement',
    icon: 'clipboard-check',
    stageCount: 4,
    isSystem: true,
  },
];

const DEMO_INSTANCES = [
  {
    id: 'demo-instance-1',
    name: 'Apex Manufacturing - Aluminum Housing Sourcing',
    templateId: 'supplier-qualification',
    templateName: 'Supplier Qualification Pipeline',
    status: 'completed',
    currentStageIndex: 6,
    totalStages: 6,
    currentStage: null,
    context: {
      client_name: 'Apex Manufacturing Inc.',
      selected_supplier: 'Bharat Precision Engineering',
      quote: '$12.80/unit',
    },
    createdAt: '2026-07-15T10:00:00Z',
    completedAt: '2026-07-20T14:30:00Z',
  },
  {
    id: 'demo-instance-2',
    name: 'TechCorp Q3 Lead Campaign',
    templateId: 'lead-nurture',
    templateName: 'Lead Nurturing',
    status: 'running',
    currentStageIndex: 2,
    totalStages: 4,
    currentStage: { id: 'outreach', name: 'Personalized Outreach', agent: 'content_marketing' },
    context: {
      total_leads: 156,
      emails_sent: 89,
    },
    createdAt: '2026-07-18T09:00:00Z',
  },
  {
    id: 'demo-instance-3',
    name: 'Southeast Asia Market Entry',
    templateId: 'market-launch',
    templateName: 'New Market Launch',
    status: 'pending',
    currentStageIndex: 0,
    totalStages: 3,
    currentStage: { id: 'research', name: 'Market Research', agent: 'market_research' },
    context: {},
    createdAt: '2026-07-20T08:00:00Z',
  },
];

function WorkflowsPage() {
  const [templates, setTemplates] = useState([]);
  const [instances, setInstances] = useState([]);
  const [activeTab, setActiveTab] = useState('templates');
  const [loading, setLoading] = useState(true);
  const [selectedCategory, setSelectedCategory] = useState('all');
  const selectedProjectId = useSelectedProjectId();

  // Demo mode detection
  const [isDemoMode] = useState(() => {
    return localStorage.getItem('enableAgentsMode') !== 'live';
  });

  const fetchTemplates = useCallback(async () => {
    // Use demo data in demo mode
    if (isDemoMode) {
      setTemplates(DEMO_TEMPLATES);
      return;
    }

    try {
      const res = await fetch(`${API_CONFIG.BASE_URL}/workflows/templates`, {
        headers: authJsonHeaders(),
      });
      const data = await res.json();
      if (data.success) {
        setTemplates(data.templates || []);
      }
    } catch (err) {
      console.error('Error fetching templates:', err);
    }
  }, [isDemoMode]);

  const fetchInstances = useCallback(async () => {
    // Use demo data in demo mode
    if (isDemoMode) {
      setInstances(DEMO_INSTANCES);
      return;
    }

    try {
      const url = selectedProjectId
        ? `${API_CONFIG.BASE_URL}/workflows/instances?project_id=${selectedProjectId}`
        : `${API_CONFIG.BASE_URL}/workflows/instances`;
      const res = await fetch(url, { headers: authJsonHeaders() });
      const data = await res.json();
      if (data.success) {
        setInstances(data.instances || []);
      }
    } catch (err) {
      console.error('Error fetching instances:', err);
    }
  }, [selectedProjectId, isDemoMode]);

  useEffect(() => {
    const load = async () => {
      setLoading(true);
      await Promise.all([fetchTemplates(), fetchInstances()]);
      setLoading(false);
    };
    load();
  }, [fetchTemplates, fetchInstances]);

  const handleStartWorkflow = async (templateId) => {
    if (!selectedProjectId) {
      showToast('Select a project first', 'warning');
      return;
    }

    // Demo mode: simulate starting workflow
    if (isDemoMode) {
      const template = templates.find((t) => t.id === templateId);
      const newInstance = {
        id: `demo-${Date.now()}`,
        name: `${template?.name || 'New Workflow'} - Demo`,
        templateId,
        templateName: template?.name || 'Workflow',
        status: 'pending',
        currentStageIndex: 0,
        totalStages: template?.stageCount || 3,
        currentStage: { id: 'step-1', name: 'First Stage', agent: 'requirements_gathering' },
        context: {},
        createdAt: new Date().toISOString(),
      };
      setInstances((prev) => [newInstance, ...prev]);
      showToast('Demo workflow created', 'success');
      setActiveTab('active');
      return;
    }

    try {
      const res = await fetch(`${API_CONFIG.BASE_URL}/workflows/instances`, {
        method: 'POST',
        headers: authJsonHeaders(),
        body: JSON.stringify({
          templateId,
          projectId: selectedProjectId,
        }),
      });
      const data = await res.json();
      if (data.success) {
        showToast('Workflow started', 'success');
        setInstances((prev) => [data.instance, ...prev]);
        setActiveTab('active');
      } else {
        showToast(data.error || 'Failed to start workflow', 'error');
      }
    } catch (err) {
      showToast('Error starting workflow', 'error');
    }
  };

  const handleDeleteInstance = async (instanceId) => {
    // Demo mode: just remove from state
    if (isDemoMode) {
      setInstances((prev) => prev.filter((i) => i.id !== instanceId));
      showToast('Demo workflow deleted', 'success');
      return;
    }

    try {
      const res = await fetch(`${API_CONFIG.BASE_URL}/workflows/instances/${instanceId}`, {
        method: 'DELETE',
        headers: authJsonHeaders(),
      });
      if (res.ok) {
        setInstances((prev) => prev.filter((i) => i.id !== instanceId));
        showToast('Workflow deleted', 'success');
      }
    } catch (err) {
      showToast('Error deleting workflow', 'error');
    }
  };

  const categories = [...new Set(templates.map((t) => t.category))];
  const filteredTemplates =
    selectedCategory === 'all'
      ? templates
      : templates.filter((t) => t.category === selectedCategory);

  const activeInstances = instances.filter((i) => ['pending', 'running', 'paused'].includes(i.status));
  const completedInstances = instances.filter((i) => i.status === 'completed');

  if (loading) {
    return (
      <>
        <Header />
        <div className="workflows-page">
          <div className="workflows-loading">Loading workflows...</div>
        </div>
      </>
    );
  }

  return (
    <>
      <Header />
      <div className="workflows-page">
        <div className="workflows-header">
          <div className="workflows-header-left">
            <BackButton />
            <h1>Workflows</h1>
          </div>
          <ProjectSelector />
        </div>

        <ProjectGate>
          <div className="workflows-tabs">
            <button
              className={`workflow-tab ${activeTab === 'templates' ? 'active' : ''}`}
              onClick={() => setActiveTab('templates')}
            >
              Templates
            </button>
            <button
              className={`workflow-tab ${activeTab === 'active' ? 'active' : ''}`}
              onClick={() => setActiveTab('active')}
            >
              Active ({activeInstances.length})
            </button>
            <button
              className={`workflow-tab ${activeTab === 'completed' ? 'active' : ''}`}
              onClick={() => setActiveTab('completed')}
            >
              Completed ({completedInstances.length})
            </button>
          </div>

          {activeTab === 'templates' && (
            <div className="workflows-content">
              <div className="category-filter">
                <button
                  className={`category-btn ${selectedCategory === 'all' ? 'active' : ''}`}
                  onClick={() => setSelectedCategory('all')}
                >
                  All
                </button>
                {categories.map((cat) => (
                  <button
                    key={cat}
                    className={`category-btn ${selectedCategory === cat ? 'active' : ''}`}
                    onClick={() => setSelectedCategory(cat)}
                  >
                    {cat.charAt(0).toUpperCase() + cat.slice(1)}
                  </button>
                ))}
              </div>

              {filteredTemplates.length === 0 ? (
                <EmptyState
                  icon="workflow"
                  title="No templates available"
                  description="Workflow templates will appear here."
                />
              ) : (
                <div className="templates-grid">
                  {filteredTemplates.map((template) => (
                    <div key={template.id} className="template-card">
                      <div className="template-header">
                        <span className="template-icon">{template.icon || 'workflow'}</span>
                        <span className="template-category">{template.category}</span>
                      </div>
                      <h3>{template.name}</h3>
                      <p>{template.description}</p>
                      <div className="template-stages">
                        {template.stageCount} stages
                      </div>
                      <button
                        className="btn btn-primary"
                        onClick={() => handleStartWorkflow(template.id)}
                      >
                        Start Workflow
                      </button>
                    </div>
                  ))}
                </div>
              )}
            </div>
          )}

          {activeTab === 'active' && (
            <div className="workflows-content">
              {activeInstances.length === 0 ? (
                <EmptyState
                  icon="play"
                  title="No active workflows"
                  description="Start a workflow from the Templates tab."
                  action={{ label: 'Browse Templates', onClick: () => setActiveTab('templates') }}
                />
              ) : (
                <div className="instances-list">
                  {activeInstances.map((instance) => (
                    <InstanceCard
                      key={instance.id}
                      instance={instance}
                      onDelete={() => handleDeleteInstance(instance.id)}
                    />
                  ))}
                </div>
              )}
            </div>
          )}

          {activeTab === 'completed' && (
            <div className="workflows-content">
              {completedInstances.length === 0 ? (
                <EmptyState
                  icon="check-circle"
                  title="No completed workflows"
                  description="Completed workflows will appear here."
                />
              ) : (
                <div className="instances-list">
                  {completedInstances.map((instance) => (
                    <InstanceCard
                      key={instance.id}
                      instance={instance}
                      onDelete={() => handleDeleteInstance(instance.id)}
                    />
                  ))}
                </div>
              )}
            </div>
          )}
        </ProjectGate>
      </div>
    </>
  );
}

function InstanceCard({ instance, onDelete }) {
  const progress = instance.totalStages > 0
    ? Math.round((instance.currentStageIndex / instance.totalStages) * 100)
    : 0;

  const statusColors = {
    pending: 'var(--color-warning)',
    running: 'var(--color-info)',
    paused: 'var(--color-text-muted)',
    completed: 'var(--color-success)',
    failed: 'var(--color-error)',
  };

  return (
    <div className="instance-card">
      <div className="instance-header">
        <h3>{instance.name}</h3>
        <span className="instance-status" style={{ color: statusColors[instance.status] }}>
          {instance.status}
        </span>
      </div>
      <p className="instance-template">Template: {instance.templateName}</p>

      <div className="instance-progress">
        <div className="progress-bar">
          <div className="progress-fill" style={{ width: `${progress}%` }} />
        </div>
        <span className="progress-text">
          {instance.currentStageIndex} / {instance.totalStages} stages
        </span>
      </div>

      {instance.currentStage && instance.status === 'running' && (
        <div className="current-stage">
          <strong>Current:</strong> {instance.currentStage.name}
        </div>
      )}

      <div className="instance-actions">
        {instance.status === 'running' && (
          <a href={`/workflows/${instance.id}`} className="btn btn-primary btn-sm">
            Continue
          </a>
        )}
        {instance.status === 'pending' && (
          <a href={`/workflows/${instance.id}`} className="btn btn-primary btn-sm">
            Start
          </a>
        )}
        {instance.status === 'completed' && (
          <a href={`/workflows/${instance.id}`} className="btn btn-secondary btn-sm">
            View
          </a>
        )}
        <button className="btn btn-danger btn-sm" onClick={onDelete}>
          Delete
        </button>
      </div>
    </div>
  );
}

export default WorkflowsPage;
