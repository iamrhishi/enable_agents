import { API_CONFIG } from '../config/apiConfig';
import React, { useState, useEffect } from 'react';
import Header from '../core/Header';
import '../styles/ExecutiveAssistantPage.css';
import { showToast } from '../core/toast';
import { Input, Textarea, Select } from '../components';

function ExecutiveAssistantPage() {
  const [activeTab, setActiveTab] = useState('projects');
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

  // Load data from localStorage on mount
  useEffect(() => {
    const savedProjects = JSON.parse(localStorage.getItem('ea_projects') || '[]');
    const savedTasks = JSON.parse(localStorage.getItem('ea_tasks') || '[]');
    const savedPeople = JSON.parse(localStorage.getItem('ea_people') || '[]');
    
    setProjects(savedProjects);
    setTasks(savedTasks);
    setPeople(savedPeople);
  }, []);

  // Save to localStorage whenever data changes
  useEffect(() => {
    localStorage.setItem('ea_projects', JSON.stringify(projects));
  }, [projects]);

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
        {/* Tab Navigation */}
        <div className="module-tabs">
          <button
            className={`module-tab ${activeTab === 'projects' ? 'module-tab--active' : ''}`}
            onClick={() => setActiveTab('projects')}
          >
            Projects ({projects.length})
          </button>
          <button
            className={`module-tab ${activeTab === 'tasks' ? 'module-tab--active' : ''}`}
            onClick={() => setActiveTab('tasks')}
          >
            Tasks ({tasks.length})
          </button>
          <button
            className={`module-tab ${activeTab === 'people' ? 'module-tab--active' : ''}`}
            onClick={() => setActiveTab('people')}
          >
            People ({people.length})
          </button>
          <button
            className={`module-tab ${activeTab === 'reminders' ? 'module-tab--active' : ''}`}
            onClick={() => setActiveTab('reminders')}
          >
            WhatsApp Reminders
          </button>
        </div>

        {/* Projects Tab */}
        {activeTab === 'projects' && (
          <div className="ea-content projects-section">
            <div className="section-header">
              <h2>Your Projects</h2>
              <button className="add-button" onClick={() => setShowProjectForm(!showProjectForm)}>
                + Add Project
              </button>
            </div>

            {showProjectForm && (
              <div className="form-card">
                <h3>Create New Project</h3>
                <Input
                  placeholder="Project Name"
                  value={newProject.name}
                  onChange={(e) => setNewProject({ ...newProject, name: e.target.value })}
                />
                <Textarea
                  placeholder="Project Description"
                  value={newProject.description}
                  onChange={(e) => setNewProject({ ...newProject, description: e.target.value })}
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
                <div className="form-buttons">
                  <button className="btn btn-primary" onClick={handleAddProject}>Save Project</button>
                  <button className="btn btn-secondary" onClick={() => setShowProjectForm(false)}>Cancel</button>
                </div>
              </div>
            )}

            <div className="projects-grid">
              {projects.length === 0 ? (
                <p className="empty-state">No projects yet. Create one to get started!</p>
              ) : (
                projects.map((project) => (
                  <div key={project.id} className="project-card">
                    <div className="card-header">
                      <h3>{project.name}</h3>
                      <span className={`status-badge ${project.status.toLowerCase()}`}>
                        {project.status}
                      </span>
                    </div>
                    <p className="description">{project.description}</p>
                    {project.dueDate && (
                      <p className="due-date">📅 Due: {new Date(project.dueDate).toLocaleDateString()}</p>
                    )}
                    <div className="card-stats">
                      <span>Tasks: {getProjectTasks(project.id).length}</span>
                      <span>People: {getProjectPeople(project.id).length}</span>
                    </div>
                    <div className="card-actions">
                      <button
                        className="btn-view"
                        onClick={() => {
                          setSelectedProject(project);
                          setActiveTab('tasks');
                        }}
                      >
                        View Tasks
                      </button>
                      <button
                        className="btn-delete"
                        onClick={() => handleDeleteProject(project.id)}
                      >
                        Delete
                      </button>
                    </div>
                  </div>
                ))
              )}
            </div>
          </div>
        )}

        {/* Tasks Tab */}
        {activeTab === 'tasks' && (
          <div className="ea-content tasks-section">
            <div className="section-header">
              <h2>Tasks {selectedProject && `- ${selectedProject.name}`}</h2>
              <button className="add-button" onClick={() => setShowTaskForm(!showTaskForm)}>
                + Add Task
              </button>
            </div>

            {showTaskForm && (
              <div className="form-card">
                <h3>Create New Task</h3>
                <Select
                  value={newTask.projectId}
                  onChange={(e) => setNewTask({ ...newTask, projectId: e.target.value })}
                  variant="outlined"
                >
                  <option value="">Select Project</option>
                  {projects.map((p) => (
                    <option key={p.id} value={p.id}>
                      {p.name}
                    </option>
                  ))}
                </Select>
                <Input
                  placeholder="Task Title"
                  value={newTask.title}
                  onChange={(e) => setNewTask({ ...newTask, title: e.target.value })}
                />
                <Textarea
                  placeholder="Task Description"
                  value={newTask.description}
                  onChange={(e) => setNewTask({ ...newTask, description: e.target.value })}
                />
                <Select
                  value={newTask.assignedTo}
                  onChange={(e) => setNewTask({ ...newTask, assignedTo: e.target.value })}
                  variant="outlined"
                >
                  <option value="">Assign to Person</option>
                  {people.map((p) => (
                    <option key={p.id} value={p.id}>
                      {p.name}
                    </option>
                  ))}
                </Select>
                <Input
                  type="date"
                  value={newTask.dueDate}
                  onChange={(e) => setNewTask({ ...newTask, dueDate: e.target.value })}
                />
                <Select
                  value={newTask.priority}
                  onChange={(e) => setNewTask({ ...newTask, priority: e.target.value })}
                  variant="outlined"
                >
                  <option value="Low">Low Priority</option>
                  <option value="Medium">Medium Priority</option>
                  <option value="High">High Priority</option>
                </Select>
                <Select
                  value={newTask.status}
                  onChange={(e) => setNewTask({ ...newTask, status: e.target.value })}
                  variant="outlined"
                >
                  <option value="Pending">Pending</option>
                  <option value="In Progress">In Progress</option>
                  <option value="Completed">Completed</option>
                </Select>
                <div className="form-buttons">
                  <button className="btn btn-primary" onClick={handleAddTask}>Save Task</button>
                  <button className="btn btn-secondary" onClick={() => setShowTaskForm(false)}>Cancel</button>
                </div>
              </div>
            )}

            <div className="tasks-list">
              {tasks.length === 0 ? (
                <p className="empty-state">No tasks yet. Create one to get started!</p>
              ) : (
                tasks
                  .filter((t) => !selectedProject || t.projectId === selectedProject.id)
                  .map((task) => {
                    const assignedPerson = people.find(p => p.id === task.assignedTo);
                    const project = projects.find(p => p.id === task.projectId);
                    return (
                      <div key={task.id} className="task-card">
                        <div className="task-header">
                          <h3>{task.title}</h3>
                          <div className="task-badges">
                            <span className={`priority-badge ${task.priority.toLowerCase()}`}>
                              {task.priority}
                            </span>
                            <span className={`status-badge ${task.status.toLowerCase()}`}>
                              {task.status}
                            </span>
                          </div>
                        </div>
                        <p className="description">{task.description}</p>
                        <div className="task-info">
                          {project && <span>📋 {project.name}</span>}
                          {assignedPerson && <span>👤 {assignedPerson.name}</span>}
                          {task.dueDate && <span>📅 {new Date(task.dueDate).toLocaleDateString()}</span>}
                        </div>
                        <div className="task-actions">
                          <button
                            className="btn-remind"
                            onClick={() => {
                              setSelectedTask(task);
                              setReminderDetails({
                                ...reminderDetails,
                                person: task.assignedTo,
                                task: task.id
                              });
                              setActiveTab('reminders');
                            }}
                          >
                            Send Reminder
                          </button>
                          <button
                            className="btn-delete"
                            onClick={() => handleDeleteTask(task.id)}
                          >
                            Delete
                          </button>
                        </div>
                      </div>
                    );
                  })
              )}
            </div>
          </div>
        )}

        {/* People Tab */}
        {activeTab === 'people' && (
          <div className="ea-content people-section">
            <div className="section-header">
              <h2>Team Members</h2>
              <button className="add-button" onClick={() => setShowPersonForm(!showPersonForm)}>
                + Add Person
              </button>
            </div>

            {showPersonForm && (
              <div className="form-card">
                <h3>Add Team Member</h3>
                <Input
                  placeholder="Full Name"
                  value={newPerson.name}
                  onChange={(e) => setNewPerson({ ...newPerson, name: e.target.value })}
                />
                <Input
                  type="email"
                  placeholder="Email"
                  value={newPerson.email}
                  onChange={(e) => setNewPerson({ ...newPerson, email: e.target.value })}
                />
                <Input
                  type="tel"
                  placeholder="Phone Number"
                  value={newPerson.phone}
                  onChange={(e) => setNewPerson({ ...newPerson, phone: e.target.value })}
                />
                <Input
                  type="tel"
                  placeholder="WhatsApp Number (with country code, e.g., +1234567890)"
                  value={newPerson.whatsappNumber}
                  onChange={(e) => setNewPerson({ ...newPerson, whatsappNumber: e.target.value })}
                />
                <Input
                  placeholder="Role/Position"
                  value={newPerson.role}
                  onChange={(e) => setNewPerson({ ...newPerson, role: e.target.value })}
                />
                <div className="form-buttons">
                  <button className="btn btn-primary" onClick={handleAddPerson}>Save Person</button>
                  <button className="btn btn-secondary" onClick={() => setShowPersonForm(false)}>Cancel</button>
                </div>
              </div>
            )}

            <div className="people-grid">
              {people.length === 0 ? (
                <p className="empty-state">No team members yet. Add one to get started!</p>
              ) : (
                people.map((person) => (
                  <div key={person.id} className="person-card">
                    <div className="person-avatar">👤</div>
                    <h3>{person.name}</h3>
                    {person.role && <p className="role">{person.role}</p>}
                    <div className="contact-info">
                      {person.email && <p>📧 {person.email}</p>}
                      {person.phone && <p>📱 {person.phone}</p>}
                      {person.whatsappNumber && <p>💬 {person.whatsappNumber}</p>}
                    </div>
                    <div className="person-actions">
                      <button
                        className="btn-remind"
                        onClick={() => {
                          setSelectedPerson(person);
                          setReminderDetails({ ...reminderDetails, person: person.id });
                          setActiveTab('reminders');
                        }}
                      >
                        Send WhatsApp
                      </button>
                      <button
                        className="btn-delete"
                        onClick={() => handleDeletePerson(person.id)}
                      >
                        Delete
                      </button>
                    </div>
                  </div>
                ))
              )}
            </div>
          </div>
        )}

        {/* WhatsApp Reminders Tab */}
        {activeTab === 'reminders' && (
          <div className="ea-content reminders-section">
            <div className="section-header">
              <h2>WhatsApp Reminders</h2>
            </div>

            {people.length === 0 ? (
              <div className="empty-state-card">
                <p>No team members added yet. Add team members first to send reminders.</p>
              </div>
            ) : (
              <div className="reminder-card">
                <h3>Send WhatsApp Reminder</h3>

                <div className="reminder-form">
                  <div className="form-group">
                    <label>Select Person:</label>
                    <Select
                      value={reminderDetails.person}
                      onChange={(e) => setReminderDetails({ ...reminderDetails, person: e.target.value })}
                      variant="outlined"
                    >
                      <option value="">Choose a team member</option>
                      {people.map((p) => (
                        <option key={p.id} value={p.id}>
                          {p.name} ({p.whatsappNumber})
                        </option>
                      ))}
                    </Select>
                  </div>

                  <div className="form-group">
                    <label>Related Task (Optional):</label>
                    <Select
                      value={reminderDetails.task}
                      onChange={(e) => setReminderDetails({ ...reminderDetails, task: e.target.value })}
                      variant="outlined"
                    >
                      <option value="">Select a task</option>
                      {tasks.map((t) => (
                        <option key={t.id} value={t.id}>
                          {t.title}
                        </option>
                      ))}
                    </Select>
                  </div>

                  <div className="form-group">
                    <label>Related Project (Optional):</label>
                    <Select
                      value={reminderDetails.project}
                      onChange={(e) => setReminderDetails({ ...reminderDetails, project: e.target.value })}
                      variant="outlined"
                    >
                      <option value="">Select a project</option>
                      {projects.map((p) => (
                        <option key={p.id} value={p.id}>
                          {p.name}
                        </option>
                      ))}
                    </Select>
                  </div>

                  <div className="form-group">
                    <label>Message:</label>
                    <Textarea
                      placeholder="Enter your reminder message"
                      value={reminderDetails.message}
                      onChange={(e) => setReminderDetails({ ...reminderDetails, message: e.target.value })}
                      rows={4}
                    />
                  </div>

                  <div className="form-group">
                    <label>Send When:</label>
                    <Select
                      value={reminderDetails.sendTime}
                      onChange={(e) => setReminderDetails({ ...reminderDetails, sendTime: e.target.value })}
                      variant="outlined"
                    >
                      <option value="now">Send Now</option>
                      <option value="tomorrow">Tomorrow Morning</option>
                      <option value="next-day">Next Day</option>
                      <option value="weekly">Weekly</option>
                    </Select>
                  </div>

                  <button className="btn btn-primary" onClick={handleSendReminder}>
                    Send via WhatsApp
                  </button>
                </div>

                <div className="reminder-preview">
                  <h4>Preview:</h4>
                  <div className="message-preview">
                    {reminderDetails.person && (
                      <p><strong>To:</strong> {people.find(p => p.id === reminderDetails.person)?.name}</p>
                    )}
                    {reminderDetails.message && (
                      <p><strong>Message:</strong> {reminderDetails.message}</p>
                    )}
                    {reminderDetails.task && (
                      <p><strong>Task:</strong> {tasks.find(t => t.id === reminderDetails.task)?.title}</p>
                    )}
                    {reminderDetails.project && (
                      <p><strong>Project:</strong> {projects.find(p => p.id === reminderDetails.project)?.name}</p>
                    )}
                  </div>
                </div>
              </div>
            )}
          </div>
        )}
      </div>
    </div>
  );
}

export default ExecutiveAssistantPage;
