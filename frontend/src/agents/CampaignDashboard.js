import React, { useState, useEffect } from 'react';
import Header from '../core/Header';
import BackButton from '../components/BackButton';
import '../styles/RequirementsGathering.css';
import { API_CONFIG } from '../config/apiConfig';

// Demo mock campaigns
const DEMO_CAMPAIGNS = [
  { id: 'demo-camp-1', name: 'Tech Startup Outreach', subject: 'Partnership Opportunity with Enable Agents', totalSent: 24, totalReplied: 8, replyRate: 33, createdAt: '2026-06-20T10:30:00Z' },
  { id: 'demo-camp-2', name: 'HR Solutions Follow-up', subject: 'How Enable Agents Can Automate Your HR', totalSent: 18, totalReplied: 5, replyRate: 28, createdAt: '2026-06-22T14:15:00Z' },
  { id: 'demo-camp-3', name: 'Enterprise Leads Q2', subject: 'Introducing AI-Powered Business Automation', totalSent: 32, totalReplied: 12, replyRate: 38, createdAt: '2026-06-25T09:00:00Z' },
];

const DEMO_RECIPIENTS = [
  { name: 'TechFlow Solutions', email: 'contact@techflow.io', sentAt: '2026-06-25T09:05:00Z', replyStatus: 'Replied', repliedAt: '2026-06-25T14:30:00Z' },
  { name: 'CloudHR Systems', email: 'info@cloudhr.com', sentAt: '2026-06-25T09:06:00Z', replyStatus: 'Replied', repliedAt: '2026-06-26T10:15:00Z' },
  { name: 'PeopleFirst Inc', email: 'sales@peoplefirst.io', sentAt: '2026-06-25T09:07:00Z', replyStatus: 'No Reply', repliedAt: null },
  { name: 'WorkStream AI', email: 'hello@workstream.ai', sentAt: '2026-06-25T09:08:00Z', replyStatus: 'Replied', repliedAt: '2026-06-25T16:45:00Z' },
  { name: 'HRNova Solutions', email: 'contact@hrnova.com', sentAt: '2026-06-25T09:09:00Z', replyStatus: 'No Reply', repliedAt: null },
];

function CampaignDashboard() {
  const userId = localStorage.getItem("firstName") || "";
  const [campaigns, setCampaigns] = useState([]);
  const [selectedCampaign, setSelectedCampaign] = useState(null);
  const [recipients, setRecipients] = useState([]);
  const [isLoading, setIsLoading] = useState(true);
  const [loadError, setLoadError] = useState('');
  const [isRefreshingReplies, setIsRefreshingReplies] = useState(false);

  // Demo mode detection
  const [isDemoMode, setIsDemoMode] = useState(() => {
    const stored = localStorage.getItem('enableAgentsMode');
    return stored !== 'live';
  });

  // Listen for mode changes
  useEffect(() => {
    const handleModeChange = () => {
      const stored = localStorage.getItem('enableAgentsMode');
      setIsDemoMode(stored !== 'live');
    };
    window.addEventListener('storage', handleModeChange);
    const interval = setInterval(handleModeChange, 1000);
    return () => {
      window.removeEventListener('storage', handleModeChange);
      clearInterval(interval);
    };
  }, []);

  useEffect(() => {
    fetchCampaigns();
    if (!isDemoMode) {
      const intervalId = setInterval(fetchCampaigns, 30000);
      return () => clearInterval(intervalId);
    }
  }, [isDemoMode]);

  useEffect(() => {
    if (!selectedCampaign) return;
    viewCampaign(selectedCampaign);
    const intervalId = setInterval(() => viewCampaign(selectedCampaign), 30000);
    return () => clearInterval(intervalId);
  }, [selectedCampaign]);

  const fetchCampaigns = async () => {
    setIsLoading(true);
    setLoadError('');

    // In demo mode, use mock data
    if (isDemoMode) {
      setCampaigns(DEMO_CAMPAIGNS);
      setIsLoading(false);
      return;
    }

    try {
      const res = await fetch(`${API_CONFIG.GET_CAMPAIGNS_STATS}?username=${encodeURIComponent(userId)}`);
      const data = await res.json();
      if (data.success) {
        setCampaigns(data.campaigns);
        if (data.campaigns && data.campaigns.length > 0) {
          refreshRepliesInBackground(data.campaigns);
        }
      } else {
        setLoadError(data.error || 'Failed to load campaign stats.');
      }
    } catch (e) {
      console.error(e);
      setLoadError('Unable to load campaign dashboard right now.');
    } finally {
      setIsLoading(false);
    }
  };

  const refreshRepliesInBackground = async (campaignList) => {
    setIsRefreshingReplies(true);
    try {
      await Promise.all(
        campaignList.map((campaign) =>
          fetch(API_CONFIG.GET_CAMPAIGN_RECIPIENTS.replace('{campaignId}', campaign.id), {
            method: 'GET'
          }).catch((error) => {
            console.error('Reply refresh failed for campaign', campaign.id, error);
          })
        )
      );
      const res = await fetch(`${API_CONFIG.GET_CAMPAIGNS_STATS}?username=${encodeURIComponent(userId)}`);
      const data = await res.json();
      if (data.success) {
        setCampaigns(data.campaigns);
      }
    } catch (error) {
      console.error('Background reply refresh failed:', error);
    } finally {
      setIsRefreshingReplies(false);
    }
  };

  const viewCampaign = async (campaignId) => {
    // In demo mode, use mock recipients
    if (isDemoMode) {
      setRecipients(DEMO_RECIPIENTS);
      setSelectedCampaign(campaignId);
      return;
    }

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
            <button className="workspace-tab" onClick={() => window.location.href='/market-research'}>Leads</button>
            <button className="workspace-tab" onClick={() => window.location.href='/market-research?tab=saved'}>Saved Leads</button>
            <button className="workspace-tab active-tab">Campaign Dashboard</button>
          </div>

          <div className="workspace-content-box">
            <div className="ai-assisted" style={{ background: 'transparent', boxShadow: 'none' }}>
              {!selectedCampaign ? (
                <div style={{ display: 'flex', flexDirection: 'column', height: '100%', flex: 1, minHeight: 0 }}>
                  <h2 style={{ color: '#1E3A5F', borderBottom: '2px solid #F1EAE4', paddingBottom: '5px', marginBottom: '4px', flexShrink: 0 }}>Campaign Performance</h2>
                  <p style={{ margin: '0 0 10px 0', color: '#4b5563', fontSize: '12px' }}>Reply data auto-refreshes every 30 seconds.</p>
                  {isLoading ? <p>Loading...</p> : loadError ? (
                    <p style={{ color: '#b42318' }}>{loadError}</p>
                  ) : (
                    <div className="table-wrapper">
                      {isRefreshingReplies && <p style={{ margin: '0 0 8px 0', color: '#6b7280', fontSize: '12px' }}>Refreshing reply counts...</p>}
                      <table className="research-table" style={{ width: '100%' }}>  
                        <thead>
                          <tr>
                            <th style={{ textAlign: 'left', position: 'sticky', top: 0, background: '#F1EAE4', zIndex: 1, color: '#1E3A5F', fontWeight: 700 }}>Date</th>
                            <th style={{ textAlign: 'left', position: 'sticky', top: 0, background: '#F1EAE4', zIndex: 1, color: '#1E3A5F', fontWeight: 700 }}>Campaign Name</th>
                            <th style={{ textAlign: 'left', position: 'sticky', top: 0, background: '#F1EAE4', zIndex: 1, color: '#1E3A5F', fontWeight: 700 }}>Subject Line</th>
                            <th style={{ position: 'sticky', top: 0, background: '#F1EAE4', zIndex: 1, color: '#1E3A5F', fontWeight: 700, textAlign: 'center' }}>Sent</th>
                            <th style={{ position: 'sticky', top: 0, background: '#F1EAE4', zIndex: 1, color: '#1E3A5F', fontWeight: 700, textAlign: 'center' }}>Replies</th>
                            <th style={{ position: 'sticky', top: 0, background: '#F1EAE4', zIndex: 1, color: '#1E3A5F', fontWeight: 700, textAlign: 'center' }}>Rate</th>
                            <th style={{ textAlign: 'center', position: 'sticky', top: 0, background: '#F1EAE4', zIndex: 1, color: '#1E3A5F', fontWeight: 700 }}>Action</th>
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
                          <th style={{ textAlign: 'left', position: 'sticky', top: 0, background: '#F1EAE4', zIndex: 1, color: '#1E3A5F', fontWeight: 700 }}>Business Name</th>
                          <th style={{ textAlign: 'left', position: 'sticky', top: 0, background: '#F1EAE4', zIndex: 1, color: '#1E3A5F', fontWeight: 700 }}>Email Address</th>
                          <th style={{ textAlign: 'left', position: 'sticky', top: 0, background: '#F1EAE4', zIndex: 1, color: '#1E3A5F', fontWeight: 700 }}>Sent At</th>
                          <th style={{ textAlign: 'center', position: 'sticky', top: 0, background: '#F1EAE4', zIndex: 1, color: '#1E3A5F', fontWeight: 700 }}>Reply Status</th>
                          <th style={{ textAlign: 'left', position: 'sticky', top: 0, background: '#F1EAE4', zIndex: 1, color: '#1E3A5F', fontWeight: 700 }}>Replied At</th>
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
