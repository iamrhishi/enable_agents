import React, { useState, useEffect } from 'react';
import Header from '../core/Header';
import { showToast } from '../core/toast';
import { Input, Textarea } from '../components';

// Path to the JSON file for persistent task storage
const TASKS_FILE = '/data/user_data/user_001/executive_tasks.json';

const defaultTask = {
  name: '',
  description: '',
  stakeholders: [], // [{name, role, whatsapp}]
  status: 'pending',
  history: []
};

function ExecutiveAssistantAgent() {
  const [tasks, setTasks] = useState([]);
  const [todayPrompted, setTodayPrompted] = useState(false);
  const [newTask, setNewTask] = useState(defaultTask);
  const [selectedTask, setSelectedTask] = useState(null);
  const [showAddTask, setShowAddTask] = useState(false);
  const [whatsappMessage, setWhatsappMessage] = useState('');
  const [selectedStakeholders, setSelectedStakeholders] = useState([]);
  const [confirmSend, setConfirmSend] = useState(false);
  const [loading, setLoading] = useState(false);

  // Load tasks from JSON file (simulate fetch)
  useEffect(() => {
    fetch(TASKS_FILE)
      .then(res => res.ok ? res.json() : [])
      .then(data => setTasks(Array.isArray(data) ? data : []))
      .catch(() => setTasks([]));
  }, []);

  // Prompt for today's tasks on first load
  useEffect(() => {
    if (!todayPrompted) {
      setTimeout(() => setShowAddTask(true), 500);
      setTodayPrompted(true);
    }
  }, [todayPrompted]);

  // Handle new task input
  const handleTaskInput = (e) => {
    const { name, value } = e.target;
    setNewTask(prev => ({ ...prev, [name]: value }));
  };

  // Add stakeholder to new task
  const addStakeholder = (stakeholder) => {
    setNewTask(prev => ({
      ...prev,
      stakeholders: [...prev.stakeholders, stakeholder]
    }));
  };

  // Save new task
  const saveTask = () => {
    if (!newTask.name || !newTask.description || newTask.stakeholders.length === 0) return;
    const updatedTasks = [...tasks, { ...newTask, status: 'pending', history: [] }];
    setTasks(updatedTasks);
    setShowAddTask(false);
    setNewTask(defaultTask);
    // Simulate saving to backend
    fetch(TASKS_FILE, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(updatedTasks)
    });
  };

  // Select task (old)
  const selectTask = (task) => {
    setSelectedTask(task);
    setShowAddTask(false);
  };

  // Prepare WhatsApp message
  const prepareWhatsapp = (task) => {
    setWhatsappMessage(`Reminder: ${task.name}\n${task.description}`);
    setSelectedStakeholders(task.stakeholders);
    setConfirmSend(true);
  };

  // Send WhatsApp message (placeholder)
  const sendWhatsapp = () => {
    setLoading(true);
    // Simulate WhatsApp API call
    setTimeout(() => {
      setLoading(false);
      setConfirmSend(false);
      showToast('WhatsApp message sent to selected stakeholders!', 'success');
    }, 1200);
  };

  return (
    <div className="executive-assistant-agent">
      <Header />
      <div className="main-container">
        <h2>Executive Assistant Agent</h2>
        <p>I help you keep your tasks on track and remind stakeholders via WhatsApp.</p>
        {showAddTask && (
          <div className="add-task-form">
            <h3>Add a New Task</h3>
            <input name="name" placeholder="Task Name" value={newTask.name} onChange={handleTaskInput} />
            <textarea name="description" placeholder="Task Description" value={newTask.description} onChange={handleTaskInput} />
            <h4>Stakeholders</h4>
            <StakeholderInput onAdd={addStakeholder} />
            <ul>
              {newTask.stakeholders.map((s, i) => (
                <li key={i}>{s.name} ({s.role}) - {s.whatsapp}</li>
              ))}
            </ul>
            <button onClick={saveTask} disabled={!newTask.name || !newTask.description || newTask.stakeholders.length === 0}>Save Task</button>
          </div>
        )}
        {!showAddTask && (
          <>
            <h3>Today's Tasks</h3>
            <ul>
              {tasks.map((task, idx) => (
                <li key={idx}>
                  <b>{task.name}</b> - {task.status}
                  <button onClick={() => selectTask(task)}>View</button>
                  <button onClick={() => prepareWhatsapp(task)}>Remind via WhatsApp</button>
                </li>
              ))}
            </ul>
          </>
        )}
        {selectedTask && (
          <div className="task-details">
            <h4>{selectedTask.name}</h4>
            <p>{selectedTask.description}</p>
            <ul>
              {selectedTask.stakeholders.map((s, i) => (
                <li key={i}>{s.name} ({s.role}) - {s.whatsapp}</li>
              ))}
            </ul>
            <p>Status: {selectedTask.status}</p>
            <button onClick={() => setSelectedTask(null)}>Back</button>
          </div>
        )}
        {confirmSend && (
          <div className="confirm-send">
            <h4>Send WhatsApp Reminder?</h4>
            <p>Message: {whatsappMessage}</p>
            <p>To:</p>
            <ul>
              {selectedStakeholders.map((s, i) => (
                <li key={i}>{s.name} ({s.role}) - {s.whatsapp}</li>
              ))}
            </ul>
            <button onClick={sendWhatsapp} disabled={loading}>{loading ? 'Sending...' : 'Send'}</button>
            <button onClick={() => setConfirmSend(false)}>Cancel</button>
          </div>
        )}
      </div>
    </div>
  );
}

// Stakeholder input subcomponent
function StakeholderInput({ onAdd }) {
  const [name, setName] = useState('');
  const [role, setRole] = useState('');
  const [whatsapp, setWhatsapp] = useState('');
  return (
    <div className="stakeholder-input">
      <input placeholder="Name" value={name} onChange={e => setName(e.target.value)} />
      <input placeholder="Role" value={role} onChange={e => setRole(e.target.value)} />
      <input placeholder="WhatsApp Number" value={whatsapp} onChange={e => setWhatsapp(e.target.value)} />
      <button onClick={() => {
        if (name && role && whatsapp) {
          onAdd({ name, role, whatsapp });
          setName(''); setRole(''); setWhatsapp('');
        }
      }}>Add</button>
    </div>
  );
}

export default ExecutiveAssistantAgent;
