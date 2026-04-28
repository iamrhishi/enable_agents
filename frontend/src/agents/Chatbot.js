import { API_CONFIG } from '../config/apiConfig';
import React, { useState, useRef, useEffect } from 'react';
import Header from '../core/Header';
import '../styles/Chatbot.css';

function Chatbot({ fileName }) {
  const [messages, setMessages] = useState([
    { sender: 'ai', text: 'Hi! Ask me anything about your document.' }
  ]);
  const [input, setInput] = useState('');
  const [loading, setLoading] = useState(false);
  const [chatHistory, setChatHistory] = useState([
    { subject: 'Welcome', summary: 'Introduction to the chatbot.' }
  ]);
  const chatEndRef = useRef(null);

  // Example chatbot properties
  const botProps = {
    description: "This AI chatbot helps you query and analyze your uploaded documents.",
    expertise: "Document Q&A, Data Extraction, Summarization",
    dataSources: "Uploaded PDFs, Internal Knowledge Base",
    responseFormat: "Concise, Contextual, Human-readable"
  };

  useEffect(() => {
    chatEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, [messages]);

  const sendMessage = async () => {
  if (!input.trim()) return;
  const userMessage = { sender: 'user', text: input };
  setMessages(prev => [...prev, userMessage]);
  setInput('');
  setLoading(true);

  try {
    const response = await fetch(`${API_CONFIG.API_URL}/chat_api`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        query: input
      }),
    });
    const data = await response.json();
    setMessages(prev => [
      ...prev,
      { sender: 'ai', text: data.answer || data.error || 'No answer.' }
    ]);
    // Optionally update chat history here if needed
  } catch (error) {
    setMessages(prev => [
      ...prev,
      { sender: 'ai', text: 'Error contacting server.' }
    ]);
  }
  setLoading(false);
};

  const handleKeyDown = (e) => {
    if (e.key === 'Enter' && !loading) sendMessage();
  };

  return (
    <>
      <Header />
      <div className="chatbot-page">
        <div className="chatbot-layout">
          {/* Left: Chatbot Properties */}
          <div className="chatbot-properties">
            <h3>Chatbot Properties</h3>
            <div><strong>Description:</strong> <span>{botProps.description}</span></div>
            <div><strong>Expertise:</strong> <span>{botProps.expertise}</span></div>
            <div><strong>Data Sources:</strong> <span>{botProps.dataSources}</span></div>
            <div><strong>Response Format:</strong> <span>{botProps.responseFormat}</span></div>
          </div>
          {/* Center: Chat Window */}
          <div className="chatbot-card">
            <h2 className="chatbot-title">AI Chatbot</h2>
            <div className="chatbot-window">
              <div className="chatbot-messages">
                {messages.map((msg, idx) => (
                  <div
                    key={idx}
                    className={`chatbot-message ${msg.sender === 'user' ? 'user' : 'ai'}`}
                  >
                    {msg.text}
                  </div>
                ))}
                <div ref={chatEndRef} />
              </div>
              <div className="chatbot-input-row">
                <input
                  type="text"
                  value={input}
                  onChange={e => setInput(e.target.value)}
                  onKeyDown={handleKeyDown}
                  placeholder="Type your question..."
                  disabled={loading}
                />
                <button onClick={sendMessage} disabled={loading || !input.trim()}>
                  {loading ? '...' : 'Send'}
                </button>
              </div>
            </div>
          </div>
          {/* Right: Chat History */}
          <div className="chatbot-history">
            <h3>Chat History</h3>
            <ul>
              {chatHistory.map((item, idx) => (
                <li key={idx}>
                  <strong>{item.subject}</strong>
                  <div className="chatbot-history-summary">{item.summary}</div>
                </li>
              ))}
            </ul>
          </div>
        </div>
      </div>
    </>
  );
}

export default Chatbot;