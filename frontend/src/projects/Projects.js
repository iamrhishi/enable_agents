import React, { useState, useEffect } from 'react';
import { useNavigate } from 'react-router-dom';
import './Projects.css';
import { showToast } from '../core/toast';
import Header from '../core/Header';
import { BackButton, showConfirm } from '../components';
import { AGENTS } from '../config/agentsConfig';
import { initializeDemoProjects } from '../hooks/useProjectData';
import { useMode } from '../contexts';

const API_URL = process.env.REACT_APP_API_URL || 'http://localhost:5000';
const PROJECTS_STORAGE_KEY = 'enableAgentsProjects';

// Helper to get/save projects from localStorage
const getStoredProjects = () => {
  try {
    const data = localStorage.getItem(PROJECTS_STORAGE_KEY);
    return data ? JSON.parse(data) : [];
  } catch {
    return [];
  }
};

const saveStoredProjects = (projects) => {
  try {
    localStorage.setItem(PROJECTS_STORAGE_KEY, JSON.stringify(projects));
  } catch (e) {
    console.warn('Failed to save projects:', e);
  }
};

// Initialize demo projects on first load
initializeDemoProjects();

const Icons = {
  ArrowLeft: () => (
    <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
      <path d="M19 12H5M12 19l-7-7 7-7"/>
    </svg>
  ),
  Plus: () => (
    <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
      <line x1="12" y1="5" x2="12" y2="19"/><line x1="5" y1="12" x2="19" y2="12"/>
    </svg>
  ),
  Folder: () => (
    <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
      <path d="M22 19a2 2 0 0 1-2 2H4a2 2 0 0 1-2-2V5a2 2 0 0 1 2-2h5l2 3h9a2 2 0 0 1 2 2z"/>
    </svg>
  ),
  Trash: () => (
    <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
      <polyline points="3 6 5 6 21 6"/><path d="M19 6v14a2 2 0 0 1-2 2H7a2 2 0 0 1-2-2V6m3 0V4a2 2 0 0 1 2-2h4a2 2 0 0 1 2 2v2"/>
    </svg>
  ),
  X: () => (
    <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
      <line x1="18" y1="6" x2="6" y2="18"/><line x1="6" y1="6" x2="18" y2="18"/>
    </svg>
  ),
  Users: () => (
    <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
      <path d="M17 21v-2a4 4 0 0 0-4-4H5a4 4 0 0 0-4 4v2"/><circle cx="9" cy="7" r="4"/><path d="M23 21v-2a4 4 0 0 0-3-3.87"/><path d="M16 3.13a4 4 0 0 1 0 7.75"/>
    </svg>
  ),
};

function Projects() {
  const navigate = useNavigate();
  const userEmail = localStorage.getItem('userEmail') || '';
  const { isDemoMode } = useMode();

  const [projects, setProjects] = useState([]);
  const [loading, setLoading] = useState(true);
  const [showCreateModal, setShowCreateModal] = useState(false);
  const [newProject, setNewProject] = useState({
    name: '',
    description: '',
  });
  const [creating, setCreating] = useState(false);

  useEffect(() => {
    fetchProjects();
  }, [isDemoMode]);

  const fetchProjects = async () => {
    if (isDemoMode) {
      // Use localStorage for demo mode
      const storedProjects = getStoredProjects();
      setProjects(storedProjects);
      setLoading(false);
      return;
    }

    try {
      const res = await fetch(`${API_URL}/api/projects`, {
        headers: { 'X-User-Id': userEmail },
      });
      if (res.ok) {
        const data = await res.json();
        setProjects(data.projects || []);
      }
    } catch (err) {
      console.error('Failed to fetch projects:', err);
    } finally {
      setLoading(false);
    }
  };

  const handleCreate = async () => {
    if (!newProject.name.trim()) {
      showToast('Please enter a project name', 'warning');
      return;
    }

    // All agents are enabled by default
    const allAgents = ['marketResearch', 'salesHelper', 'contentMarketing',
      'communityNetwork', 'eventNetworking', 'executiveAssistant', 'dataInsights'];

    if (isDemoMode) {
      const project = {
        id: `proj-${Date.now()}`,
        name: newProject.name,
        description: newProject.description,
        agents: allAgents,
        owner: userEmail || 'demo@example.com',
        status: 'active',
        createdAt: new Date().toISOString().split('T')[0],
        updatedAt: new Date().toISOString(),
        data: {},
      };
      const updatedProjects = [...projects, project];
      setProjects(updatedProjects);
      saveStoredProjects(updatedProjects);
      setNewProject({ name: '', description: '' });
      setShowCreateModal(false);
      showToast('Project created', 'success');
      return;
    }

    setCreating(true);
    try {
      const res = await fetch(`${API_URL}/api/projects`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
          'X-User-Id': userEmail,
        },
        body: JSON.stringify(newProject),
      });

      if (res.ok) {
        showToast('Project created', 'success');
        setNewProject({ name: '', description: '' });
        setShowCreateModal(false);
        fetchProjects();
      } else {
        const data = await res.json();
        showToast(data.error || 'Failed to create', 'error');
      }
    } catch {
      showToast('Failed to create project', 'error');
    } finally {
      setCreating(false);
    }
  };

  const handleDelete = async (projectId) => {
    const project = projects.find(p => p.id === projectId);
    const confirmed = await showConfirm({
      title: 'Delete project?',
      message: `This will permanently remove "${project?.name || 'this project'}" and its agent data. This cannot be undone.`,
      confirmLabel: 'Delete',
      cancelLabel: 'Cancel',
      variant: 'danger',
    });
    if (!confirmed) return;

    if (isDemoMode) {
      const updatedProjects = projects.filter(p => p.id !== projectId);
      setProjects(updatedProjects);
      saveStoredProjects(updatedProjects);
      showToast('Project deleted', 'success');
      return;
    }

    try {
      const res = await fetch(`${API_URL}/api/projects/${projectId}`, {
        method: 'DELETE',
        headers: { 'X-User-Id': userEmail },
      });

      if (res.ok) {
        showToast('Project deleted', 'success');
        fetchProjects();
      }
    } catch {
      showToast('Failed to delete project', 'error');
    }
  };

  const getAgentName = (agentId) => {
    return AGENTS[agentId]?.name || agentId;
  };

  const Icon = ({ name }) => {
    const I = Icons[name];
    return I ? <I /> : null;
  };

  return (
    <div className="projects-page">
      <Header />
      <div className="projects-container">
        <header className="projects-header">
          <BackButton />
          <div className="header-content">
            <h1>Projects</h1>
            <p className="text-muted">
              Shared workspaces across agents and team members
            </p>
          </div>
          <button className="btn-create" onClick={() => setShowCreateModal(true)}>
            <Icon name="Plus" />
            New Project
          </button>
        </header>

        {loading ? (
          <div className="loading">Loading...</div>
        ) : projects.length === 0 ? (
          <div className="empty-state-card">
            <Icon name="Folder" />
            <h3>No projects yet</h3>
            <p>Create a project to share work across agents and team members.</p>
            <button className="btn-primary" onClick={() => setShowCreateModal(true)}>
              Create First Project
            </button>
          </div>
        ) : (
          <div className="projects-grid">
            {projects.map(project => (
              <div key={project.id} className="project-card">
                <div className="project-card-header">
                  <div className="project-icon">
                    <Icon name="Folder" />
                  </div>
                  <div className="project-title">
                    <h3>{project.name}</h3>
                    <span className={`status-badge ${project.status}`}>
                      {project.status}
                    </span>
                  </div>
                  <button
                    className="btn-icon danger"
                    onClick={() => handleDelete(project.id)}
                    title="Delete project"
                  >
                    <Icon name="Trash" />
                  </button>
                </div>

                {project.description && (
                  <p className="project-description">{project.description}</p>
                )}

                <div className="project-agents">
                  <span className="label">Agents:</span>
                  <div className="agent-tags">
                    {project.agents.map(agentId => (
                      <span key={agentId} className="agent-tag">
                        {getAgentName(agentId)}
                      </span>
                    ))}
                  </div>
                </div>

                <div className="project-meta">
                  <span className="meta-item">
                    Created {project.createdAt}
                  </span>
                  {project.data && Object.keys(project.data).length > 0 && (
                    <div className="project-stats">
                      {project.data.tasks && <span>{project.data.tasks} tasks</span>}
                      {project.data.leads && <span>{project.data.leads} leads</span>}
                      {project.data.content && <span>{project.data.content} content</span>}
                      {project.data.attendees && <span>{project.data.attendees} attendees</span>}
                    </div>
                  )}
                </div>

                <div className="project-actions">
                  <button
                    className="btn-open"
                    onClick={() => {
                      // Navigate to first enabled agent with this project
                      const firstAgent = project.agents[0];
                      const route = AGENTS[firstAgent]?.route;
                      if (route) {
                        navigate(`${route}?project=${project.id}`);
                      }
                    }}
                  >
                    Open Project
                  </button>
                </div>
              </div>
            ))}
          </div>
        )}
      </div>

      {/* Create Project Modal */}
      {showCreateModal && (
        <div className="modal-overlay" onClick={() => setShowCreateModal(false)}>
          <div className="modal modal-lg" onClick={e => e.stopPropagation()}>
            <div className="modal-header">
              <h3>Create New Project</h3>
              <button className="btn-icon" onClick={() => setShowCreateModal(false)}>
                <Icon name="X" />
              </button>
            </div>
            <div className="modal-body">
              <div className="field">
                <label>Project Name *</label>
                <input
                  type="text"
                  value={newProject.name}
                  onChange={(e) => setNewProject({ ...newProject, name: e.target.value })}
                  placeholder="e.g., Q3 Marketing Campaign"
                  autoFocus
                />
              </div>
              <div className="field">
                <label>Description</label>
                <textarea
                  value={newProject.description}
                  onChange={(e) => setNewProject({ ...newProject, description: e.target.value })}
                  placeholder="Brief description of the project"
                  rows={2}
                />
              </div>
              <p className="field-hint" style={{ marginTop: '8px', color: 'var(--color-text-muted)' }}>
                All agents will have access to this project.
              </p>
            </div>
            <div className="modal-footer">
              <button className="btn-secondary" onClick={() => setShowCreateModal(false)}>
                Cancel
              </button>
              <button
                className="btn-primary"
                onClick={handleCreate}
                disabled={!newProject.name.trim() || creating}
              >
                {creating ? 'Creating...' : 'Create Project'}
              </button>
            </div>
          </div>
        </div>
      )}
    </div>
  );
}

export default Projects;
