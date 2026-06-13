import React, { useState, useEffect } from 'react';
import Header from '../core/Header';
import BackButton from '../components/BackButton';
import '../styles/RequirementsGathering.css';
import { API_CONFIG } from '../config/apiConfig';

function CampaignDashboard() {
  const userId = localStorage.getItem("firstName") || "";
  const [campaigns, setCampaigns] = useState([]);
  const [selectedCampaign, setSelectedCampaign] = useState(null);
  const [recipients, setRecipients] = useState([]);
  const [isLoading, setIsLoading] = useState(true);

  useEffect(() => {
    fetchCampaigns();
    const intervalId = setInterval(fetchCampaigns, 30000);
    return () => clearInterval(intervalId);
  }, []);

  useEffect(() => {
    if (!selectedCampaign) return;
    viewCampaign(selectedCampaign);
    const intervalId = setInterval(() => viewCampaign(selectedCampaign), 30000);
    return () => clearInterval(intervalId);
  }, [selectedCampaign]);

  const fetchCampaigns = async () => {
    setIsLoading(true);
    try {
      const res = await fetch(`${API_CONFIG.GET_CAMPAIGNS_STATS}?username=${encodeURIComponent(userId)}`);
      const data = await res.json();
      if (data.success) {
        setCampaigns(data.campaigns);
      }
    } catch (e) {
      console.error(e);
    } finally {
      setIsLoading(false);
    }
  };

  const viewCampaign = async (campaignId) => {
    try {
      const res = await fetch(API_CONFIG.GET_CAMPAIGN_RECIPIENTS.replace('{campaignId}', campaignId));
      const data = await res.json();
      if (data.success) {
        setRecipients(data.recipients);
        setSelectedCampaign(campaignId);
      }
    } catch (e) {
      console.error(e);
    }
  };

  return (
    <div className="requirements-page">
      <Header />
      <BackButton />
      <div className="requirements-container">
        
        <div className="main-workspace-area">
          <div className="tabs-container">
            <button className="workspace-tab" onClick={() => window.location.href='/requirements'}>Leads</button>
            <button className="workspace-tab active-tab">Campaign Dashboard</button>
          </div>

          <div className="workspace-content-box">
            <div className="ai-assisted" style={{ background: 'transparent', boxShadow: 'none' }}>
              {!selectedCampaign ? (
                <div style={{ display: 'flex', flexDirection: 'column', height: '100%', flex: 1, minHeight: 0 }}>
                  <h2 style={{ color: '#1E3A5F', borderBottom: '2px solid #F1EAE4', paddingBottom: '5px', marginBottom: '4px', flexShrink: 0 }}>Campaign Performance</h2>
                  <p style={{ margin: '0 0 10px 0', color: '#4b5563', fontSize: '12px' }}>Reply data auto-refreshes every 30 seconds.</p>
                  {isLoading ? <p>Loading...</p> : (
                    <div className="table-wrapper">
                      <table className="research-table" style={{ width: '100%' }}>  
                        <thead>
                          <tr>
                            <th style={{ textAlign: 'left', position: 'sticky', top: 0, background: '#F1EAE4', zIndex: 1 }}>Date</th>
                            <th style={{ textAlign: 'left', position: 'sticky', top: 0, background: '#F1EAE4', zIndex: 1 }}>Campaign Name</th>    
                            <th style={{ textAlign: 'left', position: 'sticky', top: 0, background: '#F1EAE4', zIndex: 1 }}>Subject Line</th>     
                            <th style={{ position: 'sticky', top: 0, background: '#F1EAE4', zIndex: 1 }}>Sent</th>
                            <th style={{ position: 'sticky', top: 0, background: '#F1EAE4', zIndex: 1 }}>Replies</th>
                            <th style={{ position: 'sticky', top: 0, background: '#F1EAE4', zIndex: 1 }}>Rate</th>
                            <th style={{ textAlign: 'center', position: 'sticky', top: 0, background: '#F1EAE4', zIndex: 1 }}>Action</th>
                          </tr>
                        </thead>
                        <tbody>
                          {campaigns.length === 0 ? (
                            <tr><td colSpan="7" style={{ textAlign: 'center', padding: '20px' }}>No campaigns sent yet.</td></tr>
                          ) : campaigns.map(c => (
                            <tr key={c.id}>
                              <td>{new Date(c.createdAt).toLocaleDateString()}</td> 
                              <td>{c.name}</td>
                              <td>{c.subject}</td>
                              <td style={{ textAlign: 'center' }}>{c.totalSent}</td>
                              <td style={{ textAlign: 'center' }}>{c.totalReplied}</td>
                              <td style={{ textAlign: 'center' }}>{c.replyRate}%</td>
                              <td style={{ textAlign: 'center' }}>
                                <button
                                  className="export-button compact"
                                  onClick={() => viewCampaign(c.id)}
                                  style={{ backgroundColor: '#1E3A5F', color: 'white', border: 'none', padding: '4px 10px', borderRadius: '4px', cursor: 'pointer', fontWeight: 'bold' }}
                                >
                                  View
                                </button>
                              </td>
                            </tr>
                          ))}
                        </tbody>
                      </table>
                    </div>
                  )}
                </div>
              ) : (
                <div style={{ display: 'flex', flexDirection: 'column', height: '100%', flex: 1, minHeight: 0 }}>
                  <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', borderBottom: '2px solid #F1EAE4', paddingBottom: '5px', marginBottom: '10px', flexShrink: 0 }}>
                    <h2 style={{ color: '#1E3A5F', margin: 0 }}>Recipient Details</h2>
                    <button
                      onClick={() => setSelectedCampaign(null)}
                      style={{ backgroundColor: '#D6C7B8', color: '#1E3A5F', border: 'none', padding: '6px 12px', borderRadius: '4px', cursor: 'pointer', fontWeight: 'bold', fontSize: '12px' }}>
                      ← Back to Campaigns
                    </button>
                  </div>
                  <div className="table-wrapper">
                    <table className="research-table" style={{ width: '100%' }}>    
                      <thead>
                        <tr>
                          <th style={{ textAlign: 'left', position: 'sticky', top: 0, background: '#F1EAE4', zIndex: 1 }}>Business Name</th>      
                          <th style={{ textAlign: 'left', position: 'sticky', top: 0, background: '#F1EAE4', zIndex: 1 }}>Email Address</th>      
                          <th style={{ textAlign: 'left', position: 'sticky', top: 0, background: '#F1EAE4', zIndex: 1 }}>Sent At</th>
                          <th style={{ textAlign: 'center', position: 'sticky', top: 0, background: '#F1EAE4', zIndex: 1 }}>Reply Status</th>     
                          <th style={{ textAlign: 'left', position: 'sticky', top: 0, background: '#F1EAE4', zIndex: 1 }}>Replied At</th>
                        </tr>
                      </thead>
                      <tbody>
                        {recipients.map((r, i) => (
                          <tr key={i}>
                            <td>{r.name || 'N/A'}</td>
                            <td>{r.email}</td>
                            <td>{new Date(r.sentAt).toLocaleString()}</td>
                            <td style={{ textAlign: 'center' }}>
                              <span style={{
                                padding: '4px 8px',
                                borderRadius: '12px',
                                fontSize: '0.85em',
                                fontWeight: 'bold',
                                backgroundColor: r.replyStatus === 'Replied' ? '#D1FAE5' : '#F1EAE4',
                                color: r.replyStatus === 'Replied' ? '#065F46' : '#1E3A5F'
                              }}>
                                {r.replyStatus}
                              </span>
                            </td>
                            <td>{r.repliedAt ? new Date(r.repliedAt).toLocaleString() : '-'}</td>
                          </tr>
                        ))}
                      </tbody>
                    </table>
                  </div>
                </div>
              )}
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}

export default CampaignDashboard;
