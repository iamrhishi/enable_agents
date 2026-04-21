/**
 * Executive Assistant API Service
 * Handles all API calls for Projects, Tasks, and Team Members
 */

import { API_CONFIG } from '../config/apiConfig';

const API_BASE_URL = `${API_CONFIG.API_URL}/api`;

// Default fetch options
const defaultOptions = {
  headers: {
    'Content-Type': 'application/json',
    'Accept': 'application/json'
  }
};

// ========== PROJECTS ==========

export const projectAPI = {
  // Get all projects
  getAll: async () => {
    try {
      const response = await fetch(`${API_BASE_URL}/projects`, {
        method: 'GET',
        ...defaultOptions
      });
      if (!response.ok) throw new Error(`Failed to fetch projects: ${response.status}`);
      const data = await response.json();
      return data || [];
    } catch (error) {
      console.error('Error fetching projects:', error);
      throw error;
    }
  },

  // Get a specific project
  getById: async (projectId) => {
    try {
      const response = await fetch(`${API_BASE_URL}/projects/${projectId}`, {
        method: 'GET',
        ...defaultOptions
      });
      if (!response.ok) throw new Error(`Failed to fetch project: ${response.status}`);
      return await response.json();
    } catch (error) {
      console.error('Error fetching project:', error);
      throw error;
    }
  },

  // Create a new project
  create: async (projectData) => {
    try {
      const response = await fetch(`${API_BASE_URL}/projects`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify(projectData),
      });
      if (!response.ok) throw new Error('Failed to create project');
      return await response.json();
    } catch (error) {
      console.error('Error creating project:', error);
      throw error;
    }
  },

  // Update a project
  update: async (projectId, projectData) => {
    try {
      const response = await fetch(`${API_BASE_URL}/projects/${projectId}`, {
        method: 'PUT',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify(projectData),
      });
      if (!response.ok) throw new Error('Failed to update project');
      return await response.json();
    } catch (error) {
      console.error('Error updating project:', error);
      throw error;
    }
  },

  // Delete a project
  delete: async (projectId) => {
    try {
      const response = await fetch(`${API_BASE_URL}/projects/${projectId}`, {
        method: 'DELETE',
        ...defaultOptions
      });
      if (!response.ok) throw new Error(`Failed to delete project: ${response.status}`);
      return await response.json();
    } catch (error) {
      console.error('Error deleting project:', error);
      throw error;
    }
  },
};

// ========== TEAM MEMBERS (PEOPLE) ==========

export const peopleAPI = {
  // Get all team members
  getAll: async () => {
    try {
      const response = await fetch(`${API_BASE_URL}/people`, {
        method: 'GET',
        ...defaultOptions
      });
      if (!response.ok) throw new Error(`Failed to fetch team members: ${response.status}`);
      const data = await response.json();
      return data || [];
    } catch (error) {
      console.error('Error fetching team members:', error);
      throw error;
    }
  },

  // Get a specific team member
  getById: async (personId) => {
    try {
      const response = await fetch(`${API_BASE_URL}/people/${personId}`, {
        method: 'GET',
        ...defaultOptions
      });
      if (!response.ok) throw new Error(`Failed to fetch team member: ${response.status}`);
      return await response.json();
    } catch (error) {
      console.error('Error fetching team member:', error);
      throw error;
    }
  },

  // Create a new team member
  create: async (personData) => {
    try {
      const response = await fetch(`${API_BASE_URL}/people`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify(personData),
      });
      if (!response.ok) throw new Error('Failed to create team member');
      return await response.json();
    } catch (error) {
      console.error('Error creating team member:', error);
      throw error;
    }
  },

  // Update a team member
  update: async (personId, personData) => {
    try {
      const response = await fetch(`${API_BASE_URL}/people/${personId}`, {
        method: 'PUT',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify(personData),
      });
      if (!response.ok) throw new Error('Failed to update team member');
      return await response.json();
    } catch (error) {
      console.error('Error updating team member:', error);
      throw error;
    }
  },

  // Delete a team member
  delete: async (personId) => {
    try {
      const response = await fetch(`${API_BASE_URL}/people/${personId}`, {
        method: 'DELETE',
        ...defaultOptions
      });
      if (!response.ok) throw new Error(`Failed to delete team member: ${response.status}`);
      return await response.json();
    } catch (error) {
      console.error('Error deleting team member:', error);
      throw error;
    }
  },
};

// ========== TASKS ==========

export const tasksAPI = {
  // Get all tasks with optional filters
  getAll: async (filters = {}) => {
    try {
      const params = new URLSearchParams();
      if (filters.status) params.append('status', filters.status);
      if (filters.priority) params.append('priority', filters.priority);
      if (filters.project_id) params.append('project_id', filters.project_id);
      if (filters.assigned_to) params.append('assigned_to', filters.assigned_to);

      const url = `${API_BASE_URL}/tasks${params.toString() ? '?' + params.toString() : ''}`;
      const response = await fetch(url, {
        method: 'GET',
        ...defaultOptions
      });
      if (!response.ok) throw new Error(`Failed to fetch tasks: ${response.status}`);
      const data = await response.json();
      return data || [];
    } catch (error) {
      console.error('Error fetching tasks:', error);
      throw error;
    }
  },

  // Get tasks by project
  getByProject: async (projectId) => {
    return tasksAPI.getAll({ project_id: projectId });
  },

  // Get tasks by status
  getByStatus: async (status) => {
    return tasksAPI.getAll({ status });
  },

  // Get tasks by priority
  getByPriority: async (priority) => {
    return tasksAPI.getAll({ priority });
  },

  // Get tasks assigned to a person
  getByAssignee: async (personId) => {
    return tasksAPI.getAll({ assigned_to: personId });
  },

  // Get a specific task
  getById: async (taskId) => {
    try {
      const response = await fetch(`${API_BASE_URL}/tasks/${taskId}`, {
        method: 'GET',
        ...defaultOptions
      });
      if (!response.ok) throw new Error(`Failed to fetch task: ${response.status}`);
      return await response.json();
    } catch (error) {
      console.error('Error fetching task:', error);
      throw error;
    }
  },

  // Create a new task
  create: async (taskData) => {
    try {
      const response = await fetch(`${API_BASE_URL}/tasks`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify(taskData),
      });
      if (!response.ok) throw new Error('Failed to create task');
      return await response.json();
    } catch (error) {
      console.error('Error creating task:', error);
      throw error;
    }
  },

  // Update a task
  update: async (taskId, taskData) => {
    try {
      const response = await fetch(`${API_BASE_URL}/tasks/${taskId}`, {
        method: 'PUT',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify(taskData),
      });
      if (!response.ok) throw new Error('Failed to update task');
      return await response.json();
    } catch (error) {
      console.error('Error updating task:', error);
      throw error;
    }
  },

  // Delete a task
  delete: async (taskId) => {
    try {
      const response = await fetch(`${API_BASE_URL}/tasks/${taskId}`, {
        method: 'DELETE',
      });
      if (!response.ok) throw new Error('Failed to delete task');
      return await response.json();
    } catch (error) {
      console.error('Error deleting task:', error);
      throw error;
    }
  },
};

// ========== WHATSAPP MESSAGES ==========

export const messagesAPI = {
  // Send a WhatsApp message
  send: async (messageData) => {
    try {
      const response = await fetch(`${API_BASE_URL}/send-whatsapp`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify(messageData),
      });
      if (!response.ok) {
        const errorData = await response.json();
        throw new Error(errorData.error || 'Failed to send message');
      }
      return await response.json();
    } catch (error) {
      console.error('Error sending WhatsApp message:', error);
      throw error;
    }
  },

  // Get message history (with optional filters)
  getHistory: async (filters = {}) => {
    try {
      const params = new URLSearchParams();
      if (filters.recipient_id) params.append('recipient_id', filters.recipient_id);
      if (filters.task_id) params.append('task_id', filters.task_id);
      if (filters.project_id) params.append('project_id', filters.project_id);
      if (filters.status) params.append('status', filters.status);
      if (filters.limit) params.append('limit', filters.limit);
      if (filters.offset) params.append('offset', filters.offset);

      const url = params.toString() ? `${API_BASE_URL}/messages?${params}` : `${API_BASE_URL}/messages`;
      const response = await fetch(url, {
        method: 'GET',
        ...defaultOptions
      });
      if (!response.ok) throw new Error(`Failed to fetch message history: ${response.status}`);
      return await response.json();
    } catch (error) {
      console.error('Error fetching message history:', error);
      throw error;
    }
  },

  // Get history for a specific recipient
  getRecipientHistory: async (recipientId, limit = 20) => {
    try {
      const params = new URLSearchParams();
      params.append('limit', limit);
      const response = await fetch(`${API_BASE_URL}/messages-by-recipient/${recipientId}?${params}`, {
        method: 'GET',
        ...defaultOptions
      });
      if (!response.ok) throw new Error(`Failed to fetch recipient history: ${response.status}`);
      return await response.json();
    } catch (error) {
      console.error('Error fetching recipient message history:', error);
      throw error;
    }
  },

  // Get a specific message status
  getStatus: async (messageId) => {
    try {
      const response = await fetch(`${API_BASE_URL}/messages/${messageId}`, {
        method: 'GET',
        ...defaultOptions
      });
      if (!response.ok) throw new Error(`Failed to fetch message status: ${response.status}`);
      return await response.json();
    } catch (error) {
      console.error('Error fetching message status:', error);
      throw error;
    }
  },

  // Resend a failed message
  resend: async (messageId) => {
    try {
      const response = await fetch(`${API_BASE_URL}/messages/${messageId}/resend`, {
        method: 'POST',
        ...defaultOptions
      });
      if (!response.ok) {
        const errorData = await response.json();
        throw new Error(errorData.error || 'Failed to resend message');
      }
      return await response.json();
    } catch (error) {
      console.error('Error resending message:', error);
      throw error;
    }
  },
};

export default {
  projects: projectAPI,
  people: peopleAPI,
  tasks: tasksAPI,
  messages: messagesAPI,
};
