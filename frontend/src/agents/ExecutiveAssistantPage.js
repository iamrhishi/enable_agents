import { API_CONFIG } from '../config/apiConfig';
import React, { useState, useEffect } from 'react';
import Header from '../core/Header';
import '../styles/ExecutiveAssistantPage.css';
import { showToast } from '../core/toast';
import { Input, Textarea, Select } from '../components';
import { getAgentData, setAgentData, AGENT_KEYS } from '../utils';
import { formatDate } from '../utils/dateFormat';

// Demo data for Executive Assistant
const DEMO_PROJECTS = [
  { id: 'demo-p1', name: 'Q3 Product Launch', description: 'Launch new product features', status: 'Active', dueDate: '2026-09-15' },
  { id: 'demo-p2', name: 'Marketing Campaign', description: 'Summer marketing initiative', status: 'Active', dueDate: '2026-08-01' },
  { id: 'demo-p3', name: 'Team Expansion', description: 'Hire 5 new engineers', status: 'Active', dueDate: '2026-07-30' },
];

const DEMO_TASKS = [
  { id: 'demo-t1', projectId: 'demo-p1', title: 'Finalize feature specs', description: 'Complete technical specifications', assignedTo: 'John Smith', dueDate: '2026-07-15', priority: 'High', status: 'In Progress' },
  { id: 'demo-t2', projectId: 'demo-p1', title: 'Design review', description: 'Review UI/UX designs', assignedTo: 'Sarah Johnson', dueDate: '2026-07-20', priority: 'Medium', status: 'Pending' },
  { id: 'demo-t3', projectId: 'demo-p2', title: 'Create ad copy', description: 'Write marketing copy for ads', assignedTo: 'Mike Wilson', dueDate: '2026-07-10', priority: 'High', status: 'Completed' },
];

const DEMO_PEOPLE = [
  { id: 'demo-pe1', name: 'John Smith', email: 'john@company.com', phone: '+1 555-0101', whatsappNumber: '+1 555-0101', role: 'Product Manager', projects: ['demo-p1'] },
  { id: 'demo-pe2', name: 'Sarah Johnson', email: 'sarah@company.com', phone: '+1 555-0102', whatsappNumber: '+1 555-0102', role: 'Designer', projects: ['demo-p1', 'demo-p2'] },
  { id: 'demo-pe3', name: 'Mike Wilson', email: 'mike@company.com', phone: '+1 555-0103', whatsappNumber: '+1 555-0103', role: 'Marketing Lead', projects: ['demo-p2'] },
];

function ExecutiveAssistantPage() {
  // Demo mode detection with change listener
  const [isDemoMode, setIsDemoMode] = useState(() => {
    return localStorage.getItem('enableAgentsMode') !== 'live';
  });
  const [activeTab, setActiveTab] = useState('projects-tasks');
  const [projects, setProjects] = useState([]);
  const [tasks, setTasks] = useState([]);
  const [people, setPeople] = useState([]);
  const [showProjectForm, setShowProjectForm] = useState(false);
  const [showTaskForm, setShowTaskForm] = useState(false);
  const [showPersonForm, setShowPersonForm] = useState(false);
  const [showWhatsAppReminder, setShowWhatsAppReminder] = useState(false);
  const [selectedProject, setSelectedProject] = useState(null);
  const [selectedTask, setSelectedTask] = useState(null);
  const [selectedPerson, setSelectedPerson] = useState(null);
  const [expandedProjects, setExpandedProjects] = useState({});
  const [inlineReminderPerson, setInlineReminderPerson] = useState(null);

  // Toggle project expansion to show inline tasks
  const toggleProjectExpand = (projectId) => {
    setExpandedProjects(prev => ({
      ...prev,
      [projectId]: !prev[projectId]
    }));
  };

  // Form states
  const [newProject, setNewProject] = useState({
    id: '',
    name: '',
    description: '',
    status: 'Active',
    dueDate: ''
  });

  const [newTask, setNewTask] = useState({
    id: '',
    projectId: '',
    title: '',
    description: '',
    assignedTo: '',
    dueDate: '',
    priority: 'Medium',
    status: 'Pending'
  });

  const [newPerson, setNewPerson] = useState({
    id: '',
    name: '',
    email: '',
    phone: '',
    whatsappNumber: '',
    role: '',
    projects: []
  });

  const [reminderMessage, setReminderMessage] = useState('');
  const [reminderDetails, setReminderDetails] = useState({
    person: '',
    task: '',
    project: '',
    message: '',
    sendTime: 'now'
  });

  // Listen for mode changes
  useEffect(() => {
    const handleModeChange = () => {
      const newMode = localStorage.getItem('enableAgentsMode') !== 'live';
      setIsDemoMode(newMode);
    };
    window.addEventListener('storage', handleModeChange);
    const interval = setInterval(handleModeChange, 1000);
    return () => {
      window.removeEventListener('storage', handleModeChange);
      clearInterval(interval);
    };
  }, []);

  // Load data when mode changes - using centralized storage
  useEffect(() => {
    const savedData = getAgentData(AGENT_KEYS.EXECUTIVE_ASSISTANT, isDemoMode);

    if (isDemoMode && (!savedData || !savedData.projects?.length)) {
      // Demo mode with no saved data - load demo defaults
      setProjects(DEMO_PROJECTS);
      setTasks(DEMO_TASKS);
      setPeople(DEMO_PEOPLE);
    } else if (savedData) {
      setProjects(savedData.projects || []);
      setTasks(savedData.tasks || []);
      setPeople(savedData.people || []);
    } else {
      // Live mode with no data - clear everything
      setProjects([]);
      setTasks([]);
      setPeople([]);
    }
  }, [isDemoMode]);

  // Save to centralized storage whenever data changes
  useEffect(() => {
    if (projects.length > 0 || tasks.length > 0 || people.length > 0) {
      setAgentData(AGENT_KEYS.EXECUTIVE_ASSISTANT, { projects, tasks, people }, isDemoMode);
    }
  }, [projects, tasks, people, isDemoMode]);

  useEffect(() => {
    localStorage.setItem('ea_tasks', JSON.stringify(tasks));
  }, [tasks]);

  useEffect(() => {
    localStorage.setItem('ea_people', JSON.stringify(people));
  }, [people]);

  // Add Project
  const handleAddProject = () => {
    if (newProject.name.trim()) {
      const project = {
        ...newProject,
        id: Date.now().toString()
      };
      setProjects([...projects, project]);
      setNewProject({ id: '', name: '', description: '', status: 'Active', dueDate: '' });
      setShowProjectForm(false);
    }
  };

  // Delete Project
  const handleDeleteProject = (id) => {
    setProjects(projects.filter(p => p.id !== id));
    setTasks(tasks.filter(t => t.projectId !== id));
  };

  // Add Task
  const handleAddTask = () => {
    if (newTask.title.trim() && newTask.projectId) {
      const task = {
        ...newTask,
        id: Date.now().toString()
      };
      setTasks([...tasks, task]);
      setNewTask({
        id: '',
        projectId: '',
        title: '',
        description: '',
        assignedTo: '',
        dueDate: '',
        priority: 'Medium',
        status: 'Pending'
      });
      setShowTaskForm(false);
    }
  };

  // Delete Task
  const handleDeleteTask = (id) => {
    setTasks(tasks.filter(t => t.id !== id));
  };

  // Add Person
  const handleAddPerson = () => {
    if (newPerson.name.trim() && newPerson.whatsappNumber.trim()) {
      const person = {
        ...newPerson,
        id: Date.now().toString()
      };
      setPeople([...people, person]);
      setNewPerson({
        id: '',
        name: '',
        email: '',
        phone: '',
        whatsappNumber: '',
        role: '',
        projects: []
      });
      setShowPersonForm(false);
    }
  };

  // Delete Person
  const handleDeletePerson = (id) => {
    setPeople(people.filter(p => p.id !== id));
  };

  // Send WhatsApp Reminder
  const handleSendReminder = async () => {
    if (!reminderDetails.person || !reminderDetails.message.trim()) {
      showToast('Please select a person and enter a message', 'warning');
      return;
    }

    const person = people.find(p => p.id === reminderDetails.person);
    const task = tasks.find(t => t.id === reminderDetails.task);
    const project = projects.find(p => p.id === reminderDetails.project);

    let fullMessage = reminderDetails.message;
    if (task) {
      fullMessage = `Task Reminder: ${task.title}\n${reminderDetails.message}`;
    }
    if (project) {
      fullMessage += `\nProject: ${project.name}`;
    }

    try {
      // WhatsApp integration via Twilio or WhatsApp Business API
      const response = await fetch(`${API_CONFIG.API_URL}/send-whatsapp-reminder`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          whatsappNumber: person.whatsappNumber,
          message: fullMessage,
          personName: person.name,
          sendTime: reminderDetails.sendTime,
          taskId: task?.id,
          projectId: project?.id
        })
      });

      if (response.ok) {
        showToast(`Reminder sent to ${person.name} via WhatsApp!`, 'info');
        setReminderDetails({ person: '', task: '', project: '', message: '', sendTime: 'now' });
        setShowWhatsAppReminder(false);
      } else {
        showToast('Failed to send reminder. Please try again.', 'error');
      }
    } catch (error) {
      console.error('Error sending reminder:', error);
      showToast('Error sending reminder. Make sure WhatsApp is configured.', 'error');
    }
  };

  // Get tasks for selected project
  const getProjectTasks = (projectId) => {
    return tasks.filter(t => t.projectId === projectId);
  };

  // Get people for project
  const getProjectPeople = (projectId) => {
    return people.filter(p => p.projects.includes(projectId));
  };

  return (
    <div className="executive-assistant-page">
      <Header />

      <div className="ea-container">
        {/* Tab Navigation - Consolidated from 4 to 2 tabs */}
        <div className="module-tabs">
          <button
            className={`module-tab ${activeTab === 'projects-tasks' ? 'module-tab--active' : ''}`}
            onClick={() => setActiveTab('projects-tasks')}
          >
            Projects & Tasks ({projects.length}/{tasks.length})
          </button>
          <button
            className={`module-tab ${activeTab === 'team' ? 'module-tab--active' : ''}`}
            onClick={() => setActiveTab('team')}
          >
            Team ({people.length})
          </button>
        </div>

        {/* Projects & Tasks Tab - Consolidated view with inline tasks */}
        {activeTab === 'projects-tasks' && (
          <div className="ea-content projects-tasks-section">
            <div className="section-header">
              <h2>Projects & Tasks</h2>
              <div className="header-actions">
                <button className="add-button" onClick={() => setShowProjectForm(!showProjectForm)}>
                  + Project
                </button>
                <button className="add-button add-button-secondary" onClick={() => setShowTaskForm(!showTaskForm)}>
                  + Task
                </button>
              </div>
            </div>

            {/* Inline Project Form */}
            {showProjectForm && (
              <div className="inline-form-card">
                <h3>New Project</h3>
                <div className="inline-form-row">
                  <Input
                    placeholder="Project Name"
                    value={newProject.name}
                    onChange={(e) => setNewProject({ ...newProject, name: e.target.value })}
                  />
                  <Input
                    type="date"
                    value={newProject.dueDate}
                    onChange={(e) => setNewProject({ ...newProject, dueDate: e.target.value })}
                  />
                  <Select
                    value={newProject.status}
                    onChange={(e) => setNewProject({ ...newProject, status: e.target.value })}
                    variant="outlined"
                  >
                    <option value="Active">Active</option>
                    <option value="On Hold">On Hold</option>
                    <option value="Completed">Completed</option>
                  </Select>
                </div>
                <Textarea
                  placeholder="Project Description"
                  value={newProject.description}
                  onChange={(e) => setNewProject({ ...newProject, description: e.target.value })}
                />
                <div className="form-buttons">
                  <button className="btn btn-primary" onClick={handleAddProject}>Save</button>
                  <button className="btn btn-secondary" onClick={() => setShowProjectForm(false)}>Cancel</button>
                </div>
              </div>
            )}

            {/* Inline Task Form */}
            {showTaskForm && (
              <div className="inline-form-card">
                <h3>New Task</h3>
                <div className="inline-form-row">
                  <Select
                    value={newTask.projectId}
                    onChange={(e) => setNewTask({ ...newTask, projectId: e.target.value })}
                    variant="outlined"
                  >
                    <option value="">Select Project</option>
                    {projects.map((p) => (
                      <option key={p.id} value={p.id}>{p.name}</option>
                    ))}
                  </Select>
                  <Input
                    placeholder="Task Title"
                    value={newTask.title}
                    onChange={(e) => setNewTask({ ...newTask, title: e.target.value })}
                  />
                  <Select
                    value={newTask.priority}
                    onChange={(e) => setNewTask({ ...newTask, priority: e.target.value })}
                    variant="outlined"
                  >
                    <option value="Low">Low</option>
                    <option value="Medium">Medium</option>
                    <option value="High">High</option>
                  </Select>
                </div>
                <div className="inline-form-row">
                  <Select
                    value={newTask.assignedTo}
                    onChange={(e) => setNewTask({ ...newTask, assignedTo: e.target.value })}
                    variant="outlined"
                  >
                    <option value="">Assign to</option>
                    {people.map((p) => (
                      <option key={p.id} value={p.id}>{p.name}</option>
                    ))}
                  </Select>
                  <Input
                    type="date"
                    value={newTask.dueDate}
                    onChange={(e) => setNewTask({ ...newTask, dueDate: e.target.value })}
                  />
                  <Select
                    value={newTask.status}
                    onChange={(e) => setNewTask({ ...newTask, status: e.target.value })}
                    variant="outlined"
                  >
                    <option value="Pending">Pending</option>
                    <option value="In Progress">In Progress</option>
                    <option value="Completed">Completed</option>
                  </Select>
                </div>
                <Textarea
                  placeholder="Task Description"
                  value={newTask.description}
                  onChange={(e) => setNewTask({ ...newTask, description: e.target.value })}
                />
                <div className="form-buttons">
                  <button className="btn btn-primary" onClick={handleAddTask}>Save</button>
                  <button className="btn btn-secondary" onClick={() => setShowTaskForm(false)}>Cancel</button>
                </div>
              </div>
            )}

            {/* Projects List with Expandable Tasks */}
            <div className="projects-list">
              {projects.length === 0 ? (
                <p className="empty-state">No projects yet. Create one to get started!</p>
              ) : (
                projects.map((project) => {
                  const projectTasks = getProjectTasks(project.id);
                  const isExpanded = expandedProjects[project.id];
                  return (
                    <div key={project.id} className={`project-row ${isExpanded ? 'expanded' : ''}`}>
                      <div className="project-header-row" onClick={() => toggleProjectExpand(project.id)}>
                        <span className="expand-icon">{isExpanded ? '▼' : '▶'}</span>
                        <div className="project-info">
                          <h3>{project.name}</h3>
                          <span className="project-meta">
                            {projectTasks.length} task{projectTasks.length !== 1 ? 's' : ''}
                            {project.dueDate && ` · Due ${formatDate(project.dueDate)}`}
                          </span>
                        </div>
                        <span className={`status-badge ${project.status.toLowerCase().replace(' ', '-')}`}>
                          {project.status}
                        </span>
                        <button
                          className="btn-delete-small"
                          onClick={(e) => { e.stopPropagation(); handleDeleteProject(project.id); }}
                        >
                          ×
                        </button>
                      </div>

                      {isExpanded && (
                        <div className="project-tasks-inline">
                          {project.description && (
                            <p className="project-description">{project.description}</p>
                          )}
                          {projectTasks.length === 0 ? (
                            <p className="no-tasks-hint">No tasks yet</p>
                          ) : (
                            <div className="inline-tasks-list">
                              {projectTasks.map((task) => {
                                const assignedPerson = people.find(p => p.id === task.assignedTo);
                                return (
                                  <div key={task.id} className="inline-task-row">
                                    <span className={`priority-dot ${task.priority.toLowerCase()}`}></span>
                                    <span className="task-title">{task.title}</span>
                                    {assignedPerson && <span className="task-assignee">{assignedPerson.name}</span>}
                                    {task.dueDate && <span className="task-due">{formatDate(task.dueDate)}</span>}
                                    <span className={`status-pill ${task.status.toLowerCase().replace(' ', '-')}`}>
                                      {task.status}
                                    </span>
                                    <button
                                      className="btn-delete-small"
                                      onClick={() => handleDeleteTask(task.id)}
                                    >
                                      ×
                                    </button>
                                  </div>
                                );
                              })}
                            </div>
                          )}
                        </div>
                      )}
                    </div>
                  );
                })
              )}
            </div>
          </div>
        )}

        {/* Team Tab - Consolidated People + WhatsApp Reminders */}
        {activeTab === 'team' && (
          <div className="ea-content team-section">
            <div className="section-header">
              <h2>Team Members</h2>
              <button className="add-button" onClick={() => setShowPersonForm(!showPersonForm)}>
                + Add Member
              </button>
            </div>

            {/* Inline Add Person Form */}
            {showPersonForm && (
              <div className="inline-form-card">
                <h3>Add Team Member</h3>
                <div className="inline-form-row">
                  <Input
                    placeholder="Full Name"
                    value={newPerson.name}
                    onChange={(e) => setNewPerson({ ...newPerson, name: e.target.value })}
                  />
                  <Input
                    placeholder="Role/Position"
                    value={newPerson.role}
                    onChange={(e) => setNewPerson({ ...newPerson, role: e.target.value })}
                  />
                </div>
                <div className="inline-form-row">
                  <Input
                    type="email"
                    placeholder="Email"
                    value={newPerson.email}
                    onChange={(e) => setNewPerson({ ...newPerson, email: e.target.value })}
                  />
                  <Input
                    type="tel"
                    placeholder="Phone"
                    value={newPerson.phone}
                    onChange={(e) => setNewPerson({ ...newPerson, phone: e.target.value })}
                  />
                  <Input
                    type="tel"
                    placeholder="WhatsApp (+1234567890)"
                    value={newPerson.whatsappNumber}
                    onChange={(e) => setNewPerson({ ...newPerson, whatsappNumber: e.target.value })}
                  />
                </div>
                <div className="form-buttons">
                  <button className="btn btn-primary" onClick={handleAddPerson}>Save</button>
                  <button className="btn btn-secondary" onClick={() => setShowPersonForm(false)}>Cancel</button>
                </div>
              </div>
            )}

            {/* Team Members Table with Inline Reminder */}
            <div className="team-table-wrapper">
              {people.length === 0 ? (
                <p className="empty-state">No team members yet. Add one to get started!</p>
              ) : (
                <table className="team-table">
                  <thead>
                    <tr>
                      <th>Name</th>
                      <th>Role</th>
                      <th>Contact</th>
                      <th>WhatsApp</th>
                      <th>Actions</th>
                    </tr>
                  </thead>
                  <tbody>
                    {people.map((person) => (
                      <React.Fragment key={person.id}>
                        <tr className={inlineReminderPerson === person.id ? 'row-active' : ''}>
                          <td className="name-cell">{person.name}</td>
                          <td>{person.role || '-'}</td>
                          <td>
                            {person.email && <span className="contact-line">{person.email}</span>}
                            {person.phone && <span className="contact-line">{person.phone}</span>}
                          </td>
                          <td>{person.whatsappNumber || '-'}</td>
                          <td className="actions-cell">
                            <button
                              className={`btn-remind-inline ${inlineReminderPerson === person.id ? 'active' : ''}`}
                              onClick={() => {
                                if (inlineReminderPerson === person.id) {
                                  setInlineReminderPerson(null);
                                } else {
                                  setInlineReminderPerson(person.id);
                                  setReminderDetails({ ...reminderDetails, person: person.id });
                                }
                              }}
                            >
                              {inlineReminderPerson === person.id ? 'Close' : 'Message'}
                            </button>
                            <button
                              className="btn-delete-small"
                              onClick={() => handleDeletePerson(person.id)}
                            >
                              ×
                            </button>
                          </td>
                        </tr>
                        {/* Inline Reminder Form Row */}
                        {inlineReminderPerson === person.id && (
                          <tr className="inline-reminder-row">
                            <td colSpan="5">
                              <div className="inline-reminder-form">
                                <div className="reminder-form-row">
                                  <Select
                                    value={reminderDetails.task}
                                    onChange={(e) => setReminderDetails({ ...reminderDetails, task: e.target.value })}
                                    variant="outlined"
                                  >
                                    <option value="">Task (optional)</option>
                                    {tasks.map((t) => (
                                      <option key={t.id} value={t.id}>{t.title}</option>
                                    ))}
                                  </Select>
                                  <Select
                                    value={reminderDetails.project}
                                    onChange={(e) => setReminderDetails({ ...reminderDetails, project: e.target.value })}
                                    variant="outlined"
                                  >
                                    <option value="">Project (optional)</option>
                                    {projects.map((p) => (
                                      <option key={p.id} value={p.id}>{p.name}</option>
                                    ))}
                                  </Select>
                                  <Select
                                    value={reminderDetails.sendTime}
                                    onChange={(e) => setReminderDetails({ ...reminderDetails, sendTime: e.target.value })}
                                    variant="outlined"
                                  >
                                    <option value="now">Send Now</option>
                                    <option value="tomorrow">Tomorrow</option>
                                    <option value="next-day">Next Day</option>
                                  </Select>
                                </div>
                                <div className="reminder-form-row">
                                  <Textarea
                                    placeholder="Your message..."
                                    value={reminderDetails.message}
                                    onChange={(e) => setReminderDetails({ ...reminderDetails, message: e.target.value })}
                                    rows={2}
                                  />
                                  <button className="btn btn-primary btn-send" onClick={handleSendReminder}>
                                    Send WhatsApp
                                  </button>
                                </div>
                              </div>
                            </td>
                          </tr>
                        )}
                      </React.Fragment>
                    ))}
                  </tbody>
                </table>
              )}
            </div>
          </div>
        )}
      </div>
    </div>
  );
}

export default ExecutiveAssistantPage;
