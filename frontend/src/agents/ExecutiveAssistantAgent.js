import React, { useState, useEffect, useCallback } from 'react';
import { useSearchParams } from 'react-router-dom';
import Header from '../core/Header';
import { BackButton, ProjectSelector, LiveModeHint, AgentOutcomesStrip, ProjectGate, EmptyState, Input, Textarea, Select } from '../components';
import '../styles/ExecutiveAssistantPage.css';
import { showToast } from '../core/toast';
import { useSelectedProjectId } from '../hooks/useSelectedProjectId';

// Storage key for state persistence
const STATE_KEY = 'executiveAssistantState';

// Demo data
const DEMO_TEAM = [
  { id: 'tm-1', name: 'Sarah Chen', initials: 'SC', email: 'sarah@company.com', role: 'Engineering Lead', color: 'ember' },
  { id: 'tm-2', name: 'Michael Roberts', initials: 'MR', email: 'michael@company.com', role: 'Product Manager', color: 'ink' },
  { id: 'tm-3', name: 'Emily Watson', initials: 'EW', email: 'emily@company.com', role: 'Designer', color: 'sage' },
  { id: 'tm-4', name: 'David Kim', initials: 'DK', email: 'david@company.com', role: 'Developer', color: 'slate' },
];

const DEMO_PROJECTS = [
  { id: 'proj-1', name: 'Q3 Product Launch', description: 'Launch new features for enterprise customers', status: 'active', dueDate: '2026-09-30' },
  { id: 'proj-2', name: 'Infrastructure Upgrade', description: 'Migrate to new cloud infrastructure', status: 'active', dueDate: '2026-08-15' },
  { id: 'proj-3', name: 'Customer Portal v2', description: 'Redesign customer-facing portal', status: 'on hold', dueDate: '2026-10-31' },
];

const DEMO_TASKS = [
  { id: 'task-1', projectId: 'proj-1', title: 'Finalize feature specs', description: 'Complete PRD for new features', assigneeId: 'tm-2', priority: 'high', status: 'in-progress', dueDate: '2026-07-20' },
  { id: 'task-2', projectId: 'proj-1', title: 'Design review meeting', description: 'Review mockups with stakeholders', assigneeId: 'tm-3', priority: 'medium', status: 'pending', dueDate: '2026-07-22' },
  { id: 'task-3', projectId: 'proj-1', title: 'Backend API development', description: 'Build REST endpoints', assigneeId: 'tm-4', priority: 'high', status: 'in-progress', dueDate: '2026-07-25' },
  { id: 'task-4', projectId: 'proj-2', title: 'Database migration plan', description: 'Plan data migration strategy', assigneeId: 'tm-1', priority: 'high', status: 'completed', dueDate: '2026-07-18' },
  { id: 'task-5', projectId: 'proj-2', title: 'Set up staging environment', description: 'Configure new cloud resources', assigneeId: 'tm-4', priority: 'medium', status: 'pending', dueDate: '2026-07-28' },
  { id: 'task-6', projectId: 'proj-3', title: 'User research interviews', description: 'Interview 10 customers', assigneeId: 'tm-3', priority: 'low', status: 'pending', dueDate: '2026-08-05' },
];

function ExecutiveAssistantAgent() {
  const selectedProjectId = useSelectedProjectId();
  const [searchParams, setSearchParams] = useSearchParams();

  // Load persisted state
  const loadPersistedState = useCallback(() => {
    try {
      const saved = sessionStorage.getItem(STATE_KEY);
      return saved ? JSON.parse(saved) : {};
    } catch {
      return {};
    }
  }, []);

  // Demo mode detection
  const [isDemoMode, setIsDemoMode] = useState(() => {
    return localStorage.getItem('enableAgentsMode') !== 'live';
  });

  // State
  const [activeTab, setActiveTab] = useState(() => {
    return searchParams.get('tab') || loadPersistedState().activeTab || 'tasks';
  });
  const [team, setTeam] = useState([]);
  const [projects, setProjects] = useState([]);
  const [tasks, setTasks] = useState([]);
  const [isLoading, setIsLoading] = useState(false);

  // Task management
  const [showAddTask, setShowAddTask] = useState(false);
  const [newTask, setNewTask] = useState({ title: '', description: '', assigneeId: '', priority: 'medium', dueDate: '' });
  const [selectedProjectFilter, setSelectedProjectFilter] = useState('all');

  // Reminder modal
  const [showReminderModal, setShowReminderModal] = useState(false);
  const [reminderTarget, setReminderTarget] = useState(null);
  const [reminderMessage, setReminderMessage] = useState('');

  // Expanded project (for project view)
  const [expandedProject, setExpandedProject] = useState(null);

  // Persist state
  useEffect(() => {
    const state = { activeTab, selectedProjectFilter };
    sessionStorage.setItem(STATE_KEY, JSON.stringify(state));

    const params = new URLSearchParams(searchParams);
    if (activeTab !== 'tasks') {
      params.set('tab', activeTab);
    } else {
      params.delete('tab');
    }
    setSearchParams(params, { replace: true });
  }, [activeTab, selectedProjectFilter, searchParams, setSearchParams]);

  // Listen for mode changes
  useEffect(() => {
    const handleModeChange = () => {
      const newMode = localStorage.getItem('enableAgentsMode') !== 'live';
      if (newMode !== isDemoMode) {
        setIsDemoMode(newMode);
        setTasks([]);
        setProjects([]);
        setTeam([]);
      }
    };
    window.addEventListener('storage', handleModeChange);
    return () => window.removeEventListener('storage', handleModeChange);
  }, [isDemoMode]);

  // Load data when project selected
  useEffect(() => {
    if (!selectedProjectId) {
      setTasks([]);
      setProjects([]);
      setTeam([]);
      return;
    }

    if (isDemoMode) {
      setTeam(DEMO_TEAM);
      setProjects(DEMO_PROJECTS);
      setTasks(DEMO_TASKS);
    } else {
      // Live mode - would fetch from API
      setTeam([]);
      setProjects([]);
      setTasks([]);
    }
  }, [isDemoMode, selectedProjectId]);

  // Helpers
  const getTeamMember = (id) => team.find(m => m.id === id);

  const getTasksByStatus = (status) => {
    let filtered = tasks;
    if (selectedProjectFilter !== 'all') {
      filtered = filtered.filter(t => t.projectId === selectedProjectFilter);
    }
    return filtered.filter(t => t.status === status);
  };

  const getTasksForProject = (projectId) => tasks.filter(t => t.projectId === projectId);

  const getProjectStats = (projectId) => {
    const projectTasks = getTasksForProject(projectId);
    return {
      total: projectTasks.length,
      completed: projectTasks.filter(t => t.status === 'completed').length,
      pending: projectTasks.filter(t => t.status === 'pending').length,
      inProgress: projectTasks.filter(t => t.status === 'in-progress').length,
    };
  };

  // Handlers
  const handleAddTask = () => {
    if (!newTask.title.trim()) return;

    const task = {
      id: `task-${Date.now()}`,
      projectId: selectedProjectFilter !== 'all' ? selectedProjectFilter : projects[0]?.id,
      ...newTask,
      status: 'pending',
    };

    setTasks(prev => [...prev, task]);
    setNewTask({ title: '', description: '', assigneeId: '', priority: 'medium', dueDate: '' });
    setShowAddTask(false);
    showToast('Task created', 'success');
  };

  const handleDragStart = (e, taskId) => {
    e.dataTransfer.setData('taskId', taskId);
    e.target.classList.add('dragging');
  };

  const handleDragEnd = (e) => {
    e.target.classList.remove('dragging');
  };

  const handleDragOver = (e) => {
    e.preventDefault();
    e.currentTarget.classList.add('drag-over');
  };

  const handleDragLeave = (e) => {
    e.currentTarget.classList.remove('drag-over');
  };

  const handleDrop = (e, newStatus) => {
    e.preventDefault();
    e.currentTarget.classList.remove('drag-over');
    const taskId = e.dataTransfer.getData('taskId');

    setTasks(prev => prev.map(t =>
      t.id === taskId ? { ...t, status: newStatus } : t
    ));
    showToast(`Task moved to ${newStatus.replace('-', ' ')}`, 'info');
  };

  const handleDeleteTask = (taskId) => {
    setTasks(prev => prev.filter(t => t.id !== taskId));
    showToast('Task deleted', 'info');
  };

  const handleOpenReminder = (task) => {
    const assignee = getTeamMember(task.assigneeId);
    setReminderTarget({ task, assignee });
    setReminderMessage(`Hi ${assignee?.name || 'Team'},\n\nThis is a reminder about the task: "${task.title}"\n\nDue: ${task.dueDate}\n\nPlease provide an update on the progress.\n\nThanks!`);
    setShowReminderModal(true);
  };

  const handleSendReminder = () => {
    setIsLoading(true);
    setTimeout(() => {
      setIsLoading(false);
      setShowReminderModal(false);
      showToast(`Reminder sent to ${reminderTarget?.assignee?.name || 'team'}`, 'success');
    }, 1000);
  };

  // Render task card
  const renderTaskCard = (task) => {
    const assignee = getTeamMember(task.assigneeId);

    return (
      <div
        key={task.id}
        className={`kanban-card ${task.priority}`}
        draggable
        onDragStart={(e) => handleDragStart(e, task.id)}
        onDragEnd={handleDragEnd}
      >
        <div className="kanban-card-content">
          <div className="kanban-card-title">{task.title}</div>
          <div className="kanban-card-meta">
            {assignee && (
              <span className={`assignee-dot ${assignee.color}`}>{assignee.initials}</span>
            )}
            {task.dueDate && <span className="kanban-due">{task.dueDate}</span>}
          </div>
        </div>
        <div className="kanban-card-actions">
          <button className="card-action-btn remind" onClick={() => handleOpenReminder(task)} title="Send reminder">
            <img src="/assets/icons/mail.png" alt="" width={12} height={12} />
          </button>
          <button className="card-action-btn delete" onClick={() => handleDeleteTask(task.id)} title="Delete">×</button>
        </div>
      </div>
    );
  };

  return (
    <div className="executive-assistant-page">
      <Header />

      <div className="agent-page-header">
        <div className="agent-header-left">
          <BackButton />
          <div className="agent-header-content">
            <div className="agent-title-row">
              <h1>Executive Assistant</h1>
            </div>
            <p className="text-muted">
              Manage tasks, coordinate team, and send automated reminders
            </p>
          </div>
        </div>
        <div className="agent-header-right">
          <ProjectSelector agentKey="executiveAssistant" />
        </div>
      </div>

      <AgentOutcomesStrip
        items={[
          { iconSrc: '/assets/icons/checklist.png', title: 'Task Management', description: 'Kanban board for project tasks' },
          { iconSrc: '/assets/icons/users.png', title: 'Team Coordination', description: 'Assign and track team work' },
          { iconSrc: '/assets/icons/mail.png', title: 'Smart Reminders', description: 'Automated email notifications' },
        ]}
      />

      <LiveModeHint
        requireProject
        message="Select a project to manage tasks and team coordination."
      />

      <ProjectGate agentLabel="Executive Assistant workspace">
        <div className="ea-container">
          {/* Team Strip */}
          {team.length > 0 && (
            <div className="team-strip">
              <span className="team-strip-label">Team</span>
              <div className="team-strip-members">
                {team.map(member => {
                  const memberTasks = tasks.filter(t => t.assigneeId === member.id && t.status !== 'completed');
                  return (
                    <div key={member.id} className="team-strip-member">
                      <div className={`team-strip-avatar ${member.color}`}>{member.initials}</div>
                      <span className="team-strip-name">{member.name.split(' ')[0]}</span>
                      {memberTasks.length > 0 && (
                        <span className="team-strip-badge">{memberTasks.length}</span>
                      )}
                    </div>
                  );
                })}
              </div>
            </div>
          )}

          {/* Tabs */}
          <div className="ea-tabs">
            <button
              className={`di-main-tab ${activeTab === 'tasks' ? 'di-main-tab--active' : ''}`}
              onClick={() => setActiveTab('tasks')}
            >
              <img src="/assets/icons/checklist.png" alt="" className="di-tab-icon" />
              Tasks Board
            </button>
            <button
              className={`di-main-tab ${activeTab === 'projects' ? 'di-main-tab--active' : ''}`}
              onClick={() => setActiveTab('projects')}
            >
              <img src="/assets/icons/dashboards.png" alt="" className="di-tab-icon" />
              Projects
            </button>
            <button
              className={`di-main-tab ${activeTab === 'team' ? 'di-main-tab--active' : ''}`}
              onClick={() => setActiveTab('team')}
            >
              <img src="/assets/icons/users.png" alt="" className="di-tab-icon" />
              Team
            </button>
          </div>

          {/* Tasks Board Tab */}
          {activeTab === 'tasks' && (
            <div className="ea-content">
              <div className="section-header">
                <h2>Task Board</h2>
                <div className="header-actions">
                  <Select
                    value={selectedProjectFilter}
                    onChange={(e) => setSelectedProjectFilter(e.target.value)}
                    style={{ width: '200px' }}
                  >
                    <option value="all">All Projects</option>
                    {projects.map(p => (
                      <option key={p.id} value={p.id}>{p.name}</option>
                    ))}
                  </Select>
                  <button className="btn-compact" onClick={() => setShowAddTask(true)}>+ Add Task</button>
                </div>
              </div>

              {/* Quick Add Panel */}
              {showAddTask && (
                <div className="inline-panel" style={{ marginBottom: 'var(--space-4)' }}>
                  <div className="inline-panel-body">
                    <div className="task-form-grid">
                      <div className="field field-full">
                        <label>Task Title</label>
                        <Input
                          placeholder="What needs to be done?"
                          value={newTask.title}
                          onChange={(e) => setNewTask(prev => ({ ...prev, title: e.target.value }))}
                        />
                      </div>
                      <div className="field">
                        <label>Assignee</label>
                        <Select
                          value={newTask.assigneeId}
                          onChange={(e) => setNewTask(prev => ({ ...prev, assigneeId: e.target.value }))}
                        >
                          <option value="">Unassigned</option>
                          {team.map(m => (
                            <option key={m.id} value={m.id}>{m.name}</option>
                          ))}
                        </Select>
                      </div>
                      <div className="field">
                        <label>Priority</label>
                        <Select
                          value={newTask.priority}
                          onChange={(e) => setNewTask(prev => ({ ...prev, priority: e.target.value }))}
                        >
                          <option value="low">Low</option>
                          <option value="medium">Medium</option>
                          <option value="high">High</option>
                        </Select>
                      </div>
                      <div className="field">
                        <label>Due Date</label>
                        <Input
                          type="date"
                          value={newTask.dueDate}
                          onChange={(e) => setNewTask(prev => ({ ...prev, dueDate: e.target.value }))}
                        />
                      </div>
                    </div>
                  </div>
                  <div className="inline-panel-footer">
                    <button className="btn-secondary" onClick={() => setShowAddTask(false)}>Cancel</button>
                    <button className="btn-primary" onClick={handleAddTask} disabled={!newTask.title.trim()}>Create Task</button>
                  </div>
                </div>
              )}

              {/* Kanban Board */}
              {tasks.length === 0 ? (
                <EmptyState
                  iconType="document"
                  title="No tasks yet"
                  description="Create your first task to get started with task management."
                />
              ) : (
                <div className="kanban-board">
                  <div
                    className="kanban-column pending"
                    onDragOver={handleDragOver}
                    onDragLeave={handleDragLeave}
                    onDrop={(e) => handleDrop(e, 'pending')}
                  >
                    <div className="kanban-column-header">
                      <span className="column-title">To Do</span>
                      <span className="column-count">{getTasksByStatus('pending').length}</span>
                    </div>
                    <div className="kanban-column-body">
                      {getTasksByStatus('pending').map(renderTaskCard)}
                      {getTasksByStatus('pending').length === 0 && (
                        <p className="kanban-empty-hint">Drop tasks here</p>
                      )}
                    </div>
                  </div>

                  <div
                    className="kanban-column in-progress"
                    onDragOver={handleDragOver}
                    onDragLeave={handleDragLeave}
                    onDrop={(e) => handleDrop(e, 'in-progress')}
                  >
                    <div className="kanban-column-header">
                      <span className="column-title">In Progress</span>
                      <span className="column-count">{getTasksByStatus('in-progress').length}</span>
                    </div>
                    <div className="kanban-column-body">
                      {getTasksByStatus('in-progress').map(renderTaskCard)}
                      {getTasksByStatus('in-progress').length === 0 && (
                        <p className="kanban-empty-hint">Drop tasks here</p>
                      )}
                    </div>
                  </div>

                  <div
                    className="kanban-column completed"
                    onDragOver={handleDragOver}
                    onDragLeave={handleDragLeave}
                    onDrop={(e) => handleDrop(e, 'completed')}
                  >
                    <div className="kanban-column-header">
                      <span className="column-title">Done</span>
                      <span className="column-count">{getTasksByStatus('completed').length}</span>
                    </div>
                    <div className="kanban-column-body">
                      {getTasksByStatus('completed').map(renderTaskCard)}
                      {getTasksByStatus('completed').length === 0 && (
                        <p className="kanban-empty-hint">Drop tasks here</p>
                      )}
                    </div>
                  </div>
                </div>
              )}
            </div>
          )}

          {/* Projects Tab */}
          {activeTab === 'projects' && (
            <div className="ea-content">
              <div className="section-header">
                <h2>Projects</h2>
              </div>

              {projects.length === 0 ? (
                <EmptyState
                  iconType="document"
                  title="No projects"
                  description="Projects will appear here when configured."
                />
              ) : (
                <div className="projects-list">
                  {projects.map(project => {
                    const stats = getProjectStats(project.id);
                    const isExpanded = expandedProject === project.id;

                    return (
                      <div key={project.id} className={`project-row ${isExpanded ? 'expanded' : ''}`}>
                        <div
                          className="project-header-row"
                          onClick={() => setExpandedProject(isExpanded ? null : project.id)}
                        >
                          <span className="expand-icon">{isExpanded ? '▼' : '▶'}</span>
                          <div className="project-info">
                            <h3>{project.name}</h3>
                            <span className="project-meta">
                              {stats.total} tasks · {stats.completed} done · Due {project.dueDate}
                            </span>
                          </div>
                          <span className={`status-badge ${project.status}`}>{project.status}</span>
                        </div>

                        {isExpanded && (
                          <div className="project-tasks-inline">
                            <p className="project-description">{project.description}</p>
                            {getTasksForProject(project.id).length === 0 ? (
                              <p className="no-tasks-hint">No tasks for this project</p>
                            ) : (
                              <div className="inline-tasks-list">
                                {getTasksForProject(project.id).map(task => {
                                  const assignee = getTeamMember(task.assigneeId);
                                  return (
                                    <div key={task.id} className="inline-task-row">
                                      <span className={`priority-dot ${task.priority}`} />
                                      <span className="task-title">{task.title}</span>
                                      <span className="task-assignee">{assignee?.name || 'Unassigned'}</span>
                                      <span className={`status-pill ${task.status}`}>{task.status}</span>
                                      <span className="task-due">{task.dueDate}</span>
                                    </div>
                                  );
                                })}
                              </div>
                            )}
                          </div>
                        )}
                      </div>
                    );
                  })}
                </div>
              )}
            </div>
          )}

          {/* Team Tab */}
          {activeTab === 'team' && (
            <div className="ea-content">
              <div className="section-header">
                <h2>Team Members</h2>
              </div>

              {team.length === 0 ? (
                <EmptyState
                  iconType="document"
                  title="No team members"
                  description="Team members will appear here."
                />
              ) : (
                <div className="visiting-cards-grid">
                  {team.map(member => {
                    const memberTasks = tasks.filter(t => t.assigneeId === member.id);
                    const activeTasks = memberTasks.filter(t => t.status !== 'completed');

                    return (
                      <div key={member.id} className="visiting-card">
                        <div className="visiting-card-accent" />
                        <div className="visiting-card-top">
                          <div className={`visiting-card-monogram visiting-card-monogram--${member.color}`}>
                            {member.initials}
                          </div>
                          <div className="visiting-card-identity">
                            <h4 className="visiting-card-name">{member.name}</h4>
                            <p className="visiting-card-role">{member.role}</p>
                          </div>
                        </div>
                        <div className="visiting-card-divider" />
                        <div className="visiting-card-contacts">
                          <a href={`mailto:${member.email}`} className="visiting-card-contact">
                            <img src="/assets/icons/mail.png" alt="" className="visiting-card-contact-icon" />
                            <span>{member.email}</span>
                          </a>
                        </div>
                        <div style={{ fontSize: 'var(--text-small)', color: 'var(--color-text-muted)', marginBottom: 'var(--space-3)' }}>
                          {activeTasks.length} active tasks · {memberTasks.filter(t => t.status === 'completed').length} completed
                        </div>
                        <button
                          className="visiting-card-action"
                          onClick={() => {
                            setReminderTarget({ assignee: member, task: null });
                            setReminderMessage(`Hi ${member.name},\n\nThis is a friendly reminder about your pending tasks.\n\nPlease provide an update at your earliest convenience.\n\nThanks!`);
                            setShowReminderModal(true);
                          }}
                        >
                          Send Reminder
                        </button>
                      </div>
                    );
                  })}
                </div>
              )}
            </div>
          )}
        </div>
      </ProjectGate>

      {/* Reminder Modal */}
      {showReminderModal && (
        <div className="modal-overlay" onClick={() => setShowReminderModal(false)}>
          <div className="modal" onClick={(e) => e.stopPropagation()}>
            <div className="modal-header">
              <h3>Send Reminder</h3>
              <button className="btn-close" onClick={() => setShowReminderModal(false)}>×</button>
            </div>
            <div className="modal-body">
              <div className="field">
                <label>To</label>
                <Input
                  value={reminderTarget?.assignee?.email || ''}
                  readOnly
                />
              </div>
              <div className="field">
                <label>Subject</label>
                <Input
                  value={reminderTarget?.task ? `Reminder: ${reminderTarget.task.title}` : 'Task Update Reminder'}
                  readOnly
                />
              </div>
              <div className="field">
                <label>Message</label>
                <Textarea
                  rows={6}
                  value={reminderMessage}
                  onChange={(e) => setReminderMessage(e.target.value)}
                />
              </div>
            </div>
            <div className="modal-footer">
              <button className="btn-secondary" onClick={() => setShowReminderModal(false)}>Cancel</button>
              <button className="btn-primary" onClick={handleSendReminder} disabled={isLoading}>
                {isLoading ? 'Sending...' : 'Send Email'}
              </button>
            </div>
          </div>
        </div>
      )}
    </div>
  );
}

export default ExecutiveAssistantAgent;
