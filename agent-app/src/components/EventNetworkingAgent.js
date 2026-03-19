import React, { useState } from 'react';
import procurementUsecases from '../procurement.json';
import { FaLinkedin, FaEnvelope, FaWhatsapp, FaGlobe, FaBolt, FaChevronLeft, FaChevronRight, FaShareAlt } from 'react-icons/fa';
import '../styles/EventNetworkingAgent.css';



function getLoggedInUsername() {
  return localStorage.getItem('firstName') || 'Enable User';
}
function getLoggedInLinkedin() {
  return localStorage.getItem('linkedin') || 'https://www.linkedin.com/company/enableai/';
}
function getLoggedInEmail() {
  return localStorage.getItem('email') || 'hello@enableai.com';
}

const defaultIntro = (
  <div className="agent-details">
      <h3>Hey there! It's {getLoggedInUsername() || 'Enable User'}. Great that we are connecting! </h3>
  </div>
);

const topics = [
  'High-RoI AI Use Cases '
  // 'AI Lego Pieces',
  // 'AI-First Business Modules',
  // 'AI Security & Compliance', 
  // 'AI Testing'
];

function EventNetworkingAgent() {
  const [chat, setChat] = useState([
    { sender: 'agent', content: defaultIntro },
    { sender: 'agent', content: (
      <div>
        <div className="agent-details">
          <h3>
           We just want to share our learning and also learn from you!
          </h3>
        </div>
      </div>
    )},
    { sender: 'agent', content: (
      <div className="topic-options">
        {topics.map((topic, idx) => (
          <button
            key={topic}
            className="topic-btn"
            onClick={() => handleTopicSelect(topic)}
          >
            {topic}
          </button>
        ))}
      </div>
    )}
  ]);
  const [userSelected, setUserSelected] = useState(null);
  const [showPopup, setShowPopup] = useState(false);
  const [popupIndex, setPopupIndex] = useState(0);
  const [shareEmail, setShareEmail] = useState('');
  const [shareStatus, setShareStatus] = useState('');

  const handleShare = () => {
    // Simulate sending use case to email
    if (!shareEmail || !shareEmail.includes('@')) {
      setShareStatus('Please enter a valid email address.');
      return;
    }
    setShareStatus('Use case sent to ' + shareEmail + '!');
    setTimeout(() => setShareStatus(''), 3000);
    // Here you would add actual email sending logic
  };


  const handleTopicSelect = (topic) => {
    setUserSelected(topic);
    setShowPopup(true);
    setPopupIndex(0);
    setChat([...chat, { sender: 'user', content: topic }]);
  };

  const handleClosePopup = () => {
    setShowPopup(false);
  };

  const handlePrevCard = () => {
    setPopupIndex((prev) => (prev > 0 ? prev - 1 : procurementUsecases.length - 1));
  };
  const handleNextCard = () => {
    setPopupIndex((prev) => (prev < procurementUsecases.length - 1 ? prev + 1 : 0));
  };

  return (
    <div className="event-networking-agent-chat" style={{ position: 'relative' }}>
      <div className="chat-header enable-header">
        <div className="header-flex">
          <img src={process.env.PUBLIC_URL + '/assets/images/enable_logo.png'} alt="Enable Logo" className="header-logo" />
          <img src={process.env.PUBLIC_URL + '/assets/icons/ai-technology.png'} alt="AI Technology Logo" className="header-ai-logo" />
        </div>
      </div>
      <div className="floating-contact-icons">
        <a href={getLoggedInLinkedin()} target="_blank" rel="noopener noreferrer" title="LinkedIn">
          <FaLinkedin />
        </a>
        <a href={`mailto:${getLoggedInEmail()}`} title="Email">
          <FaEnvelope />
        </a>
        {/* <a href="https://wa.me/919075425207" target="_blank" rel="noopener noreferrer" title="WhatsApp">
          <FaWhatsapp />
        </a> */}
        <a href="https://enableyou.co" target="_blank" rel="noopener noreferrer" title="Website">
          <FaGlobe />
        </a>
      </div>
      <div className="chat-body">
        {chat.map((msg, idx) => (
          msg.sender === 'agent' ? (
            <div key={idx} className="chat-row agent">
              <div className="agent-message-with-avatar">
                <img
                  src={process.env.PUBLIC_URL + '/assets/images/Gemini_Generated_Image_p5j4wqp5j4wqp5j4.png'}
                  alt="Rhishi Agent Avatar"
                  className="agent-avatar"
                />
                <div className="agent-message-content">
                  {msg.content}
                </div>
              </div>
            </div>
          ) : (
            <div key={idx} className="chat-row user">{msg.content}</div>
          )
        ))}
      </div>
      {showPopup && (
        <div className="agent-popup-overlay">
          <div className="agent-popup-card">
            <button className="agent-popup-close" onClick={handleClosePopup} title="Close">&times;</button>
            <div className="agent-popup-content">
              <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', width: '100%', position: 'relative' }}>
                <button className="agent-popup-arrow-btn left" onClick={handlePrevCard} title="Previous" aria-label="Previous"><FaChevronLeft /></button>
                <div className="agent-popup-usecase-card">
                  <div className="usecase-card-content">
                    <h3 className="usecase-title">{procurementUsecases[popupIndex].outcome}</h3>
                    <div className="usecase-row usecase-row-highlight">
                      <span className="usecase-label usecase-label-main">Use Case:</span>
                      <span className="usecase-value usecase-value-main">{procurementUsecases[popupIndex].title}</span>
                    </div>
                    <div className="usecase-row">
                      <span className="usecase-label">Problem:</span>
                      <span className="usecase-value">{procurementUsecases[popupIndex].problem}</span>
                    </div>
                    <div className="usecase-row">
                      <span className="usecase-label">Solution:</span>
                      <span className="usecase-value">{procurementUsecases[popupIndex].solution}</span>
                    </div>
                    <div className="usecase-keywords">
                      {procurementUsecases[popupIndex].persona.map((p, i) => (
                        <span className="usecase-keyword" key={i}>{p}</span>
                      ))}
                      {procurementUsecases[popupIndex].category.map((c, i) => (
                        <span className="usecase-keyword" key={i}>{c}</span>
                      ))}
                    </div>
                    <div className="usecase-statistic">{procurementUsecases[popupIndex].statistic}</div>
                  </div>
                </div>
                <div className="usecase-share-block">
                  <input
                    type="email"
                    className="usecase-share-email"
                    placeholder="Email Use Cases"
                    value={shareEmail}
                    onChange={e => setShareEmail(e.target.value)}
                  />
                  <button className="usecase-share-btn" onClick={handleShare} title="Share use case">
                    <FaEnvelope />
                  </button>
                  
                </div>
                <button className="agent-popup-arrow-btn right" onClick={handleNextCard} title="Next" aria-label="Next"><FaChevronRight /></button>
              </div>
              {/* {shareStatus && (
                    <div className="usecase-share-status">{shareStatus}</div>
              )} */}
            </div>
          </div>
        </div>
      )}
    </div>
  );
}

export default EventNetworkingAgent;
