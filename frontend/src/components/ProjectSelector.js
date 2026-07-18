/**
 * ProjectSelector Component
 *
 * Dropdown to select/switch projects within an agent.
 * Shows only projects that have this agent enabled.
 */

import React, { useState, useEffect, useRef } from 'react';
import { useNavigate, useSearchParams } from 'react-router-dom';
import './ProjectSelector.css';
import { initializeDemoProjects } from '../hooks/useProjectData';
import { useMode } from '../contexts';

const API_URL = process.env.REACT_APP_API_URL || 'http://localhost:5000';
const PROJECTS_STORAGE_KEY = 'enableAgentsProjects';

// Helper to get projects from localStorage
const getStoredProjects = () => {
  try {
    const data = localStorage.getItem(PROJECTS_STORAGE_KEY);
    return data ? JSON.parse(data) : [];
  } catch {
    return [];
  }
};

// Initialize demo projects if needed
initializeDemoProjects();

function ProjectSelector({ agentKey, onProjectChange }) {
  const navigate = useNavigate();
  const [searchParams] = useSearchParams();
  const userEmail = localStorage.getItem('userEmail') || '';
  const { isDemoMode } = useMode();

  const [projects, setProjects] = useState([]);
  const [selectedProject, setSelectedProject] = useState(null);
  const [isOpen, setIsOpen] = useState(false);
  const [loading, setLoading] = useState(true);
  const dropdownRef = useRef(null);

  // Get project ID from URL
  const projectIdFromUrl = searchParams.get('project');

  // Fetch projects for this agent
  useEffect(() => {
    fetchProjects();
  }, [isDemoMode, agentKey]);

  // Set selected project from URL; auto-select first project in demo mode
  useEffect(() => {
    if (projectIdFromUrl && projects.length > 0) {
      const project = projects.find(p => p.id === projectIdFromUrl);
      if (project) {
        setSelectedProject(project);
        onProjectChange?.(project);
      }
      return;
    }

    if (isDemoMode && !projectIdFromUrl && projects.length > 0 && !loading) {
      const first = projects[0];
      const currentPath = window.location.pathname;
      navigate(`${currentPath}?project=${first.id}`, { replace: true });
    }
  }, [projectIdFromUrl, projects, isDemoMode, loading, navigate]);

  // Close dropdown on outside click
  useEffect(() => {
    const handleClickOutside = (e) => {
      if (dropdownRef.current && !dropdownRef.current.contains(e.target)) {
        setIsOpen(false);
      }
    };
    document.addEventListener('mousedown', handleClickOutside);
    return () => document.removeEventListener('mousedown', handleClickOutside);
  }, []);

  const fetchProjects = async () => {
    if (isDemoMode) {
      // Show ALL projects (not filtered by agent)
      const allProjects = getStoredProjects();
      setProjects(allProjects);
      setLoading(false);
      return;
    }

    try {
      // Fetch all user projects (not filtered by agent)
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

  const handleSelect = (project) => {
    setSelectedProject(project);
    setIsOpen(false);

    // Update URL with project param
    const currentPath = window.location.pathname;
    navigate(`${currentPath}?project=${project.id}`, { replace: true });

    onProjectChange?.(project);
  };

  const handleClearProject = () => {
    setSelectedProject(null);
    setIsOpen(false);

    // Remove project param from URL
    const currentPath = window.location.pathname;
    navigate(currentPath, { replace: true });

    onProjectChange?.(null);
  };

  if (loading) {
    return null;
  }

  // Don't show if no projects available
  if (projects.length === 0) {
    return (
      <div className="project-selector project-selector--empty">
        <button
          className="project-selector-btn project-selector-btn--create"
          onClick={() => navigate('/projects')}
        >
          + New Project
        </button>
      </div>
    );
  }

  return (
    <div className="project-selector" ref={dropdownRef}>
      <button
        className={`project-selector-btn ${selectedProject ? 'has-project' : ''}`}
        onClick={() => setIsOpen(!isOpen)}
        aria-expanded={isOpen}
        aria-haspopup="listbox"
      >
        <span className="project-name">
          {selectedProject ? selectedProject.name : 'Select project...'}
        </span>
        <span className="dropdown-arrow">
          <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
            <polyline points="6 9 12 15 18 9"/>
          </svg>
        </span>
      </button>

      {isOpen && (
        <div className="project-dropdown" role="listbox">
          {selectedProject && (
            <button
              className="project-option project-option--clear"
              onClick={handleClearProject}
            >
              <span className="clear-icon">×</span>
              Clear selection
            </button>
          )}

          {projects.map(project => (
            <button
              key={project.id}
              className={`project-option ${selectedProject?.id === project.id ? 'selected' : ''}`}
              onClick={() => handleSelect(project)}
              role="option"
              aria-selected={selectedProject?.id === project.id}
            >
              <span className="project-option-name">{project.name}</span>
              {selectedProject?.id === project.id && (
                <span className="check-icon">✓</span>
              )}
            </button>
          ))}

          <div className="project-dropdown-divider" />

          <button
            className="project-option project-option--create"
            onClick={() => navigate('/projects')}
          >
            <span className="plus-icon">+</span>
            Manage Projects
          </button>
        </div>
      )}
    </div>
  );
}

export default ProjectSelector;
