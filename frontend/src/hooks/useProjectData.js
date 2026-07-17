/**
 * useProjectData Hook
 *
 * Manages project-scoped data persistence for agents.
 * - Loads data when project is selected
 * - Saves data to project context (localStorage in demo, API in live)
 * - Provides shared data from other agents in the same project
 */

import { useState, useEffect, useCallback, useRef } from 'react';
import { useSearchParams } from 'react-router-dom';

const API_URL = process.env.REACT_APP_API_URL || 'http://localhost:5000';
const PROJECTS_STORAGE_KEY = 'enableAgentsProjects';

/**
 * Get projects from localStorage (demo mode)
 */
const getStoredProjects = () => {
  try {
    const data = localStorage.getItem(PROJECTS_STORAGE_KEY);
    return data ? JSON.parse(data) : [];
  } catch {
    return [];
  }
};

/**
 * Save projects to localStorage (demo mode)
 */
const saveStoredProjects = (projects) => {
  try {
    localStorage.setItem(PROJECTS_STORAGE_KEY, JSON.stringify(projects));
  } catch (e) {
    console.warn('Failed to save projects:', e);
  }
};

/**
 * Hook for managing project-scoped data
 *
 * @param {string} agentKey - The agent identifier (e.g., 'executiveAssistant')
 * @param {object} options - Configuration options
 * @param {object} options.defaultData - Default data structure for this agent
 * @param {function} options.onProjectLoad - Callback when project data is loaded
 */
export function useProjectData(agentKey, options = {}) {
  const { defaultData = {}, onProjectLoad } = options;

  // Use refs for callback and defaultData to avoid re-creating loadProject on every render
  const onProjectLoadRef = useRef(onProjectLoad);
  onProjectLoadRef.current = onProjectLoad;

  const defaultDataRef = useRef(defaultData);

  const [searchParams] = useSearchParams();
  const projectId = searchParams.get('project');

  const [isDemoMode] = useState(() => {
    return localStorage.getItem('enableAgentsMode') !== 'live';
  });

  const [project, setProject] = useState(null);
  const [agentData, setAgentData] = useState(() => defaultDataRef.current);
  const [sharedData, setSharedData] = useState({});
  const [loading, setLoading] = useState(false);
  const [lastSaved, setLastSaved] = useState(null);

  const userEmail = localStorage.getItem('userEmail') || '';

  /**
   * Load project and its data
   */
  const loadProject = useCallback(async (id) => {
    if (!id) {
      setProject(null);
      setAgentData(defaultDataRef.current);
      setSharedData({});
      return;
    }

    setLoading(true);

    if (isDemoMode) {
      // Load from localStorage
      const projects = getStoredProjects();
      const proj = projects.find(p => p.id === id);

      if (proj) {
        setProject(proj);
        // Load agent-specific data
        const myData = proj.data?.[agentKey] || defaultDataRef.current;
        setAgentData(myData);

        // Load shared data from other agents
        const shared = {};
        Object.entries(proj.data || {}).forEach(([key, value]) => {
          if (key !== agentKey) {
            shared[key] = value;
          }
        });
        setSharedData(shared);

        onProjectLoadRef.current?.(proj, myData, shared);
      }

      setLoading(false);
      return;
    }

    // Live mode - fetch from API
    try {
      const res = await fetch(`${API_URL}/api/projects/${id}`, {
        headers: { 'X-User-Id': userEmail },
      });

      if (res.ok) {
        const data = await res.json();
        const proj = data.project;

        setProject(proj);
        const myData = proj.data?.[agentKey] || defaultDataRef.current;
        setAgentData(myData);

        // Shared data
        const shared = {};
        Object.entries(proj.data || {}).forEach(([key, value]) => {
          if (key !== agentKey) {
            shared[key] = value;
          }
        });
        setSharedData(shared);

        onProjectLoadRef.current?.(proj, myData, shared);
      }
    } catch (err) {
      console.error('Failed to load project:', err);
    } finally {
      setLoading(false);
    }
  }, [agentKey, isDemoMode, userEmail]);

  /**
   * Save agent data to project
   */
  const saveData = useCallback(async (newData) => {
    if (!project) {
      // No project selected - just update local state
      setAgentData(newData);
      return;
    }

    setAgentData(newData);

    if (isDemoMode) {
      // Save to localStorage
      const projects = getStoredProjects();
      const idx = projects.findIndex(p => p.id === project.id);

      if (idx >= 0) {
        if (!projects[idx].data) {
          projects[idx].data = {};
        }
        projects[idx].data[agentKey] = newData;
        projects[idx].updatedAt = new Date().toISOString();
        saveStoredProjects(projects);
        setLastSaved(new Date());
      }
      return;
    }

    // Live mode - save to API
    try {
      await fetch(`${API_URL}/api/projects/${project.id}/data`, {
        method: 'PUT',
        headers: {
          'Content-Type': 'application/json',
          'X-User-Id': userEmail,
        },
        body: JSON.stringify({
          agent: agentKey,
          data: newData,
        }),
      });
      setLastSaved(new Date());
    } catch (err) {
      console.error('Failed to save project data:', err);
    }
  }, [project, agentKey, isDemoMode, userEmail]);

  /**
   * Update specific fields in agent data
   */
  const updateData = useCallback((updates) => {
    const newData = { ...agentData, ...updates };
    saveData(newData);
  }, [agentData, saveData]);

  /**
   * Get data from another agent in the same project
   */
  const getSharedAgentData = useCallback((otherAgentKey) => {
    return sharedData[otherAgentKey] || null;
  }, [sharedData]);

  /**
   * Check if we have access to another agent's data
   */
  const hasSharedData = useCallback((otherAgentKey) => {
    return !!sharedData[otherAgentKey];
  }, [sharedData]);

  // Load project on mount or when projectId changes
  useEffect(() => {
    loadProject(projectId);
  }, [projectId, loadProject]);

  return {
    // Project info
    project,
    projectId,
    hasProject: !!project,

    // Agent's own data
    data: agentData,
    saveData,
    updateData,

    // Shared data from other agents
    sharedData,
    getSharedAgentData,
    hasSharedData,

    // Loading state
    loading,
    lastSaved,

    // Refresh
    refresh: () => loadProject(projectId),
  };
}

/**
 * Initialize demo projects in localStorage if not present
 */
export function initializeDemoProjects() {
  const existing = getStoredProjects();
  if (existing.length > 0) return;

  const demoProjects = [
    {
      id: 'proj-1',
      name: 'Q3 Product Launch',
      description: 'New product launch campaign across all channels',
      owner: 'demo@example.com',
      agents: ['executiveAssistant', 'contentMarketing', 'salesHelper', 'dataInsights'],
      status: 'active',
      createdAt: '2026-06-15',
      updatedAt: new Date().toISOString(),
      data: {
        executiveAssistant: {
          tasks: [
            { id: 't1', title: 'Finalize launch timeline', status: 'in-progress', priority: 'high' },
            { id: 't2', title: 'Review marketing materials', status: 'pending', priority: 'medium' },
          ],
          notes: 'Q3 launch on track. Marketing team aligned.',
        },
        salesHelper: {
          leads: [
            { id: 'l1', name: 'Acme Corp', status: 'qualified', value: 50000 },
            { id: 'l2', name: 'TechStart', status: 'contacted', value: 25000 },
          ],
          totalPipeline: 75000,
        },
        contentMarketing: {
          drafts: 3,
          published: 1,
          topics: ['Product Features', 'Customer Success', 'Industry Trends'],
        },
      },
    },
    {
      id: 'proj-2',
      name: 'Enterprise Outreach',
      description: 'B2B sales campaign targeting enterprise customers',
      owner: 'demo@example.com',
      agents: ['marketResearch', 'salesHelper', 'contentMarketing', 'communityNetwork'],
      status: 'active',
      createdAt: '2026-06-20',
      updatedAt: new Date().toISOString(),
      data: {
        salesHelper: {
          leads: [
            { id: 'l3', name: 'Enterprise Corp', status: 'qualified', value: 150000 },
            { id: 'l4', name: 'BigCo Inc', status: 'negotiation', value: 200000 },
          ],
          totalPipeline: 350000,
        },
        communityNetwork: {
          connections: 45,
          warmIntros: 8,
          recentActivity: 'Connected with VP at Enterprise Corp',
        },
      },
    },
    {
      id: 'proj-3',
      name: 'Tech Summit 2026',
      description: 'Event networking and follow-up management',
      owner: 'demo@example.com',
      agents: ['eventNetworking', 'communityNetwork', 'executiveAssistant'],
      status: 'active',
      createdAt: '2026-07-01',
      updatedAt: new Date().toISOString(),
      data: {
        eventNetworking: {
          attendees: 85,
          followUps: 23,
          meetings: 12,
        },
        executiveAssistant: {
          tasks: [
            { id: 't3', title: 'Send thank you emails', status: 'pending', priority: 'high' },
            { id: 't4', title: 'Schedule follow-up calls', status: 'in-progress', priority: 'high' },
          ],
        },
      },
    },
  ];

  saveStoredProjects(demoProjects);
}

export default useProjectData;
