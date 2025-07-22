import React, { useState, useRef, useEffect } from 'react';
import Header from './Header';
import '../styles/CommunityNetworkAgent.css';

function CommunityNetworkAgent() {
  const [messages, setMessages] = useState([
    {
      id: 1,
      text: "Welcome to the Community Network Agent! I can help you:\n\n📊 Analyze uploaded alumni profiles\n📄 Extract skills from your CV using AI\n🎯 Find matching profiles based on your skills and goals\n\nUpload your files on the left to get started!",
      sender: 'agent',
      timestamp: new Date().toLocaleTimeString()
    }
  ]);
  const [inputMessage, setInputMessage] = useState('');
  const [isLoading, setIsLoading] = useState(false);
  const [csvData, setCsvData] = useState(null);
  const [userProfile, setUserProfile] = useState(null);
  const [isAnalyzing, setIsAnalyzing] = useState(false);
  const messagesEndRef = useRef(null);
  const csvFileRef = useRef(null);
  const cvFileRef = useRef(null);
  const [searchResults, setSearchResults] = useState([]);

  const scrollToBottom = () => {
    messagesEndRef.current?.scrollIntoView({ behavior: "smooth" });
  };

  useEffect(() => {
    scrollToBottom();
  }, [messages]);

  const addMessage = (text, sender, data = null) => {
    const newMessage = {
      id: Date.now(),
      text,
      sender,
      timestamp: new Date().toLocaleTimeString(),
      data
    };
    setMessages(prev => [...prev, newMessage]);
  };

  const enrichDataWithOpenAI = async (data) => {
  try {
    addMessage("🤖 Sending data to OpenAI to add required skills for each company...", 'agent');
    
    const response = await fetch('http://localhost:5000/enrich_with_openai', {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify({ data: data })
    });

    const result = await response.json();
    return result;
  } catch (error) {
    console.error('OpenAI enrichment error:', error);
    return {
      success: false,
      error: 'Failed to enrich data with OpenAI',
      data: data
    };
  }
};

  const handleCSVUpload = async (event) => {
  const file = event.target.files[0];
  if (!file) return;

  const allowedTypes = ['.csv', '.xlsx', '.xls'];
  const fileExtension = file.name.toLowerCase().substring(file.name.lastIndexOf('.'));
  
  if (!allowedTypes.includes(fileExtension)) {
    addMessage("Please upload a valid CSV or XLSX file.", 'agent');
    return;
  }

  setIsLoading(true);
  addMessage("📊 Uploading and converting file to JSON format...", 'agent');

  try {
    // Step 1: Upload and convert file
    const formData = new FormData();
    formData.append('file', file);
    formData.append('folder_name', 'alumni_data');
    formData.append('multiple_sheets', 'false');

    const response = await fetch('http://localhost:5000/file_to_json_convert', {
      method: 'POST',
      body: formData
    });

    const conversionResult = await response.json();

    if (conversionResult.success) {
      const profiles = conversionResult.data;
      
      addMessage(`✅ File converted successfully! Found ${profiles.length} profiles. Now enhancing with AI...`, 'agent');
      
      // Step 2: Enrich with OpenAI
      const enrichmentResult = await enrichDataWithOpenAI(profiles);
      console.log('Enrichment Result:', enrichmentResult);
      if (enrichmentResult.success) {
        setCsvData(enrichmentResult.data);
        
        addMessage(`🎯 Perfect! AI has enhanced all ${enrichmentResult.data.length} profiles with required skills based on their companies!`, 'agent', {
          type: 'csv_upload_enriched',
          data: { 
            profiles_count: enrichmentResult.data.length,
            columns: [...(conversionResult.columns || []), 'required_skills'],
            file_type: fileExtension.substring(1).toUpperCase(),
            sample_data: enrichmentResult.data.slice(0, 3)
          }
        });
      } else {
        // Fallback to original data if enrichment fails
        setCsvData(profiles);
        addMessage(`⚠️ File converted but AI enrichment failed: ${enrichmentResult.error}. Using original data.`, 'agent');
      }
    } else {
      addMessage(`❌ Error converting file: ${conversionResult.error}`, 'agent');
    }
  } catch (error) {
    console.error('File processing error:', error);
    addMessage("❌ Error processing file. Please try again.", 'agent');
  } finally {
    setIsLoading(false);
    event.target.value = '';
  }
};

  // Helper function for CSV parsing
  const parseCSVLine = (line) => {
    const values = [];
    let current = '';
    let inQuotes = false;
    
    for (let i = 0; i < line.length; i++) {
      const char = line[i];
      
      if (char === '"' && (i === 0 || line[i-1] === ',')) {
        inQuotes = true;
      } else if (char === '"' && inQuotes && (i === line.length - 1 || line[i+1] === ',')) {
        inQuotes = false;
      } else if (char === ',' && !inQuotes) {
        values.push(current);
        current = '';
      } else {
        current += char;
      }
    }
    
    values.push(current);
    return values;
  };

  // CV Upload Handler
  const handleCVUpload = async (event) => {
    const file = event.target.files[0];
    if (!file) return;

    if (!file.name.endsWith('.pdf')) {
      addMessage("Please upload a PDF document.", 'agent');
      return;
    }

    setIsAnalyzing(true);
    addMessage("📄 Uploading and analyzing your CV... This may take a moment.", 'agent');

    try {
      const formData = new FormData();
      formData.append('file', file);
      formData.append('folder_name', 'cv_analysis');

      const uploadResponse = await fetch('http://localhost:5000/upload', {
        method: 'POST',
        body: formData
      });

      const uploadResult = await uploadResponse.json();

      if (uploadResponse.ok) {
        const skillsPrompt = `
          Analyze this CV and extract:
          1. Technical skills (list 8-12 key skills)
          2. Career goals (list 3-5 goals)
          3. Industry focus
          4. Experience level
          
          Format the response as:
          Skills: skill1, skill2, skill3...
          Goals: goal1, goal2, goal3...
          Industry: industry name
          Level: experience level
        `;

        const ragResponse = await fetch('http://localhost:5000/rag_test', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({
            query: skillsPrompt,
            file_name: file.name,
            data_store: 'cv_analysis'
          })
        });

        const ragResult = await ragResponse.json();

        if (ragResponse.ok) {
          const profile = parseSkillsFromResponse(ragResult.answer);
          setUserProfile(profile);

          const skillsText = profile.skills.join(', ');
          const goalsText = profile.career_goals.join(', ');
          
          addMessage(`🎯 CV Analysis Complete!\n\n💼 Skills: ${skillsText}\n\n🚀 Goals: ${goalsText}\n\n📊 Industry: ${profile.industry}`, 'agent', {
            type: 'cv_analysis',
            data: profile
          });
        } else {
          addMessage(`❌ Error analyzing CV: ${ragResult.error || 'Unknown error'}`, 'agent');
        }
      } else {
        addMessage(`❌ Error uploading CV: ${uploadResult.error}`, 'agent');
      }
    } catch (error) {
      console.error('CV Analysis Error:', error);
      addMessage("❌ Error analyzing CV. Please try again.", 'agent');
    } finally {
      setIsAnalyzing(false);
      event.target.value = '';
    }
  };

  const searchProfiles = (query) => {
  if (!query.trim() || !csvData || csvData.length === 0) {
    addMessage("❌ Please provide a search term and ensure data is uploaded first.", 'agent');
    return;
  }

  const searchTerm = query.toLowerCase().trim();
  
  const results = csvData.filter(profile => {
    // Search in company fields
    const company = (
      profile.company || 
      profile.Company || 
      profile.organization || 
      profile.Organization || 
      profile.employer || 
      profile.Employer || 
      ''
    ).toLowerCase();
    
    // Search in position/title fields
    const position = (
      profile.position || 
      profile.Position || 
      profile.title || 
      profile.Title || 
      profile.job_title || 
      profile.Job_Title || 
      profile.role || 
      profile.Role || 
      ''
    ).toLowerCase();
    
    // Search in name fields
    const name = (
      profile.name || 
      profile.Name || 
      profile.full_name || 
      profile.Full_Name || 
      profile.first_name || 
      profile.last_name ||
      ''
    ).toLowerCase();
    
    // Search in location fields
    const location = (
      profile.location || 
      profile.Location || 
      profile.city || 
      profile.City || 
      ''
    ).toLowerCase();
    
    // Search in skills fields
    const skills = (
      profile.skills || 
      profile.Skills || 
      profile.required_skills?.join(' ') || 
      ''
    ).toLowerCase();
    
    // Return true if search term matches any field
    return company.includes(searchTerm) || 
           position.includes(searchTerm) || 
           name.includes(searchTerm) ||
           location.includes(searchTerm) ||
           skills.includes(searchTerm);
  });

  setSearchResults(results);
  
  // Display search results in chat
  if (results.length > 0) {
    let searchMessage = `🔍 **Found ${results.length} profile${results.length === 1 ? '' : 's'} matching "${query}":**\n\n`;
    
    results.slice(0, 10).forEach((profile, index) => {
      const name = profile.name || profile.Name || profile.full_name || `Profile ${index + 1}`;
      const company = profile.company || profile.Company || profile.organization || 'N/A';
      const position = profile.position || profile.Position || profile.title || profile.job_title || 'N/A';
      const location = profile.location || profile.Location || '';
      
      searchMessage += `**${index + 1}. ${name}**\n`;
      searchMessage += `   🏢 Company: ${company}\n`;
      searchMessage += `   💼 Position: ${position}\n`;
      if (location) searchMessage += `   📍 Location: ${location}\n`;
      
      // Show skills if available
      if (profile.required_skills && profile.required_skills.length > 0) {
        const skillsPreview = profile.required_skills.slice(0, 4).join(', ');
        const extraSkills = profile.required_skills.length > 4 ? ` (+${profile.required_skills.length - 4} more)` : '';
        searchMessage += `   🔧 Skills: ${skillsPreview}${extraSkills}\n`;
      }
      searchMessage += '\n';
    });
    
    if (results.length > 10) {
      searchMessage += `\n*Showing first 10 results. ${results.length - 10} more profiles match your search.*`;
    }
    
    addMessage(searchMessage, 'agent', {
      type: 'search_results',
      data: {
        query: query,
        results_count: results.length,
        results: results
      }
    });
  } else {
    addMessage(`❌ No profiles found matching "${query}". Try searching by:\n• Company name (e.g., "Google", "Microsoft")\n• Position (e.g., "Engineer", "Manager")\n• Person's name\n• Location\n• Skills`, 'agent');
  }
};

  // Skills parsing function
  const parseSkillsFromResponse = (response) => {
    const lines = response.split('\n');
    let skills = [];
    let goals = [];
    let industry = '';
    let level = '';

    lines.forEach(line => {
      const lowerLine = line.toLowerCase();
      if (lowerLine.includes('skills:')) {
        const skillsText = line.split(':')[1] || '';
        skills = skillsText.split(',').map(s => s.trim()).filter(s => s);
      } else if (lowerLine.includes('goals:')) {
        const goalsText = line.split(':')[1] || '';
        goals = goalsText.split(',').map(g => g.trim()).filter(g => g);
      } else if (lowerLine.includes('industry:')) {
        industry = line.split(':')[1]?.trim() || '';
      } else if (lowerLine.includes('level:')) {
        level = line.split(':')[1]?.trim() || '';
      }
    });

    if (skills.length === 0) {
      const words = response.toLowerCase();
      const techSkills = ['python', 'javascript', 'react', 'node.js', 'sql', 'aws', 'docker', 'git', 'java', 'html', 'css', 'machine learning', 'data analysis', 'project management', 'leadership'];
      skills = techSkills.filter(skill => words.includes(skill)).map(skill => 
        skill.split(' ').map(word => word.charAt(0).toUpperCase() + word.slice(1)).join(' ')
      );
    }

    if (goals.length === 0) {
      goals = ['Career Growth', 'Leadership Role', 'Technical Excellence'];
    }

    if (!industry) {
      industry = 'Technology';
    }

    if (!level) {
      level = 'Mid-level';
    }

    return {
      skills: skills.slice(0, 12),
      career_goals: goals.slice(0, 5),
      industry,
      experience_level: level
    };
  };

  // Query handlers and findMatchingProfiles function remain the same...
  const findMatchingProfiles = async (query = null) => {
    if (!csvData || !userProfile) {
      addMessage("Please upload both a CSV file and your CV first.", 'agent');
      return;
    }

    setIsLoading(true);
    addMessage("🔍 Finding matching profiles based on your skills and company offerings...", 'agent');

    try {
      const matches = [];
      const userSkills = new Set(userProfile.skills.map(s => s.toLowerCase().trim()));
      
      // Company services mapping and matching logic...
      const companyServicesMap = {
        'google': ['search', 'cloud computing', 'ai', 'machine learning', 'advertising', 'android', 'software'],
        'microsoft': ['cloud computing', 'software', 'ai', 'productivity', 'azure', 'office', 'windows'],
        'amazon': ['e-commerce', 'cloud computing', 'aws', 'logistics', 'ai', 'retail'],
        'default': ['business', 'services', 'technology']
      };

      csvData.forEach(profile => {
        let matchScore = 0;
        const reasons = [];
        
        const name = profile.name || '';
        const company = (profile.company || '').toLowerCase();
        const position = (profile.position || '').toLowerCase();
        
        const companyServices = companyServicesMap[company] || 
                               companyServicesMap[Object.keys(companyServicesMap).find(key => 
                                 company.includes(key.toLowerCase()))] || 
                               companyServicesMap['default'];
        
        // Skills matching
        const skillMatches = Array.from(userSkills).filter(skill => {
          return companyServices.some(service => 
            service.includes(skill) || skill.includes(service)
          );
        });
        
        if (skillMatches.length > 0) {
          matchScore += Math.min((skillMatches.length / userProfile.skills.length) * 50, 50);
          reasons.push(`Skills relevant to ${company}: ${skillMatches.slice(0, 3).join(', ')}`);
        }
        
        if (matchScore > 30) {
          matches.push({
            name: name,
            company: profile.company || '',
            position: profile.position || '',
            match_score: Math.min(Math.round(matchScore), 95),
            match_reason: reasons.slice(0, 2).join('; ') || 'General compatibility'
          });
        }
      });
      
      matches.sort((a, b) => b.match_score - a.match_score);
      
      if (matches.length > 0) {
        let matchText = `🎯 Found ${matches.length} matching profiles:\n\n`;
        matches.slice(0, 5).forEach((match, index) => {
          matchText += `${index + 1}. **${match.name}**\n`;
          matchText += `   🏢 ${match.company} - ${match.position}\n`;
          matchText += `   🎯 Match Score: ${match.match_score}%\n\n`;
        });

        addMessage(matchText, 'agent', {
          type: 'matches',
          data: { matches: matches.slice(0, 10) }
        });
      } else {
        addMessage("No matching profiles found. Try uploading more alumni data.", 'agent');
      }
    } catch (error) {
      addMessage("❌ Error finding matches. Please try again.", 'agent');
    } finally {
      setIsLoading(false);
    }
  };

  const handleQuery = async (query) => {
    const lowerQuery = query.toLowerCase();

    if (lowerQuery.includes('find match') || lowerQuery.includes('matching profiles')) {
      await findMatchingProfiles(query);
    } else if (lowerQuery.includes('show skills') && userProfile) {
      addMessage(`Your extracted skills: ${userProfile.skills.join(', ')}`, 'agent');
    } else if (lowerQuery.includes('show goals') && userProfile) {
      addMessage(`Your career goals: ${userProfile.career_goals.join(', ')}`, 'agent');
    } else if (lowerQuery.includes('help')) {
      addMessage(`I can help you with:
      
🎯 **Find Matches**: Say "find matching profiles"
💼 **Show Skills**: Say "show my skills"
🚀 **Show Goals**: Say "show my goals"

You can also ask specific questions like:
- "Find profiles in tech industry"
- "Show me senior developers"
- "Find people with AI experience"`, 'agent');
    } else {
      if (csvData && userProfile) {
        await findMatchingProfiles(query);
      } else {
        addMessage("Please upload CSV data and your CV using the upload section on the left to get started!", 'agent');
      }
    }
  };


  const detectSearchCommand = (message) => {
  const lowerMessage = message.toLowerCase();
  
  // Search command patterns
  const searchPatterns = [
    { pattern: /^search\s+(.+)$/i, type: 'search' },
    { pattern: /^find\s+(.+)$/i, type: 'find' },
    { pattern: /^look\s+for\s+(.+)$/i, type: 'look_for' },
    { pattern: /^show\s+me\s+(.+)$/i, type: 'show' },
    { pattern: /^who\s+works?\s+at\s+(.+)$/i, type: 'company_search' },
    { pattern: /^list\s+(.+)$/i, type: 'list' }
  ];
  
  for (const { pattern, type } of searchPatterns) {
    const match = message.match(pattern);
    if (match) {
      return {
        isSearch: true,
        type: type,
        query: match[1].trim()
      };
    }
  }
  
  return { isSearch: false };
};


  // Update your handleSendMessage function to handle both form submissions and direct calls
const handleSendMessage = (e = null) => {
  // Only call preventDefault if event exists and has the method
  if (e && typeof e.preventDefault === 'function') {
    e.preventDefault();
  }
  
  if (!inputMessage.trim()) return;

  const userMessage = inputMessage.trim();
  
  // Add user message to chat
  addMessage(userMessage, 'user');
  setInputMessage('');
  setIsLoading(true);

  // Check if the message is a search command
  if (userMessage.toLowerCase().startsWith('search ') || 
      userMessage.toLowerCase().startsWith('find ') ||
      userMessage.toLowerCase().startsWith('look for ')) {
    
    // Extract search query
    let searchQuery = '';
    if (userMessage.toLowerCase().startsWith('search ')) {
      searchQuery = userMessage.substring(7).trim();
    } else if (userMessage.toLowerCase().startsWith('find ')) {
      searchQuery = userMessage.substring(5).trim();
    } else if (userMessage.toLowerCase().startsWith('look for ')) {
      searchQuery = userMessage.substring(9).trim();
    }
    
    if (searchQuery) {
      setTimeout(() => {
        searchProfiles(searchQuery);
        setIsLoading(false);
      }, 500);
      return;
    }
  }

  // Check for other specific commands
  if (userMessage.toLowerCase().includes('help with search') || 
      userMessage.toLowerCase().includes('how to search')) {
    setTimeout(() => {
      addMessage(`🔍 **Search Help:**\n\nYou can search through uploaded profiles using these commands:\n\n• **search [term]** - Search for any term\n• **find [company]** - Find profiles by company\n• **look for [position]** - Find by position\n\n**Examples:**\n• "search Google"\n• "find Software Engineer"\n• "look for Data Scientist"\n• "search John"\n• "find Microsoft Manager"\n\nI'll search through names, companies, positions, locations, and skills!`, 'agent');
      setIsLoading(false);
    }, 500);
    return;
  }

  // Your existing message handling logic here...
  setTimeout(() => {
    if (!csvData || csvData.length === 0) {
      addMessage("Please upload a CSV/XLSX file first, then I can help you search and analyze the data. You can also upload your CV for matching!", 'agent');
    } else {
      addMessage(`I can help you search through the ${csvData.length} profiles you've uploaded! Try commands like:\n• "search Google" - Find Google employees\n• "find Engineer" - Find all engineers\n• "look for Manager" - Find management positions\n\nOr ask me to analyze the data or find matches with your CV!`, 'agent');
    }
    setIsLoading(false);
  }, 1000);
};

// Also update your handleKeyPress function to properly handle the Enter key
const handleKeyPress = (e) => {
  if (e.key === 'Enter' && !e.shiftKey) {
    e.preventDefault(); // Prevent default here since we have the event
    handleSendMessage(); // Call without passing the event
  }
};


  return (
    <div className="community-network-agent">
      <Header />
      
      <div className="main-container">
        {/* Left Section - File Uploads (1/3) */}
        <div className="upload-section">
          <h3>📁 Data Upload</h3>
          
          {/* CSV Upload */}
          <div className="upload-card">
            <div className="upload-header">
              <h4>📊 Alumni CSV Data</h4>
              <span className="status-indicator">
                {csvData ? `✅ ${csvData.length} profiles loaded` : '⏳ No data'}
              </span>
            </div>
            <p>Upload a CSV file containing alumni profiles with columns like Name, Company, Position, etc.</p>
            <button 
              onClick={() => csvFileRef.current?.click()} 
              className="upload-btn csv-btn"
              disabled={isLoading}
            >
              {isLoading ? '📤 Uploading...' : '📊 Upload Alumni CSV'}
            </button>
            <input
              type="file"
              ref={csvFileRef}
              onChange={handleCSVUpload}
              accept=".csv"
              style={{ display: 'none' }}
            />
          </div>

          {/* Data Preview Table */}
            {csvData && csvData.length > 0 && (
            <div className="data-preview">
                <h4>📋 Data Preview</h4>
                
                <div className="table-container">
                <table className="data-table">
                    <thead>
                    <tr>
                        {Object.keys(csvData[0]).map((column, index) => (
                        <th key={index} title={column}>{column}</th>
                        ))}
                    </tr>
                    </thead>
                    <tbody>
                    {csvData.slice(0, 5).map((row, rowIndex) => (
                        <tr key={rowIndex}>
                        {Object.keys(csvData[0]).map((column, colIndex) => (
                            <td key={colIndex} title={row[column]}>
                            {row[column] || '-'}
                            </td>
                        ))}
                        </tr>
                    ))}
                    </tbody>
                </table>
                
                {csvData.length > 5 && (
                    <div className="table-footer">
                    <span>Showing 5 of {csvData.length} records</span>
                    </div>
                )}
                </div>
            </div>
            )}

          {/* CV Upload */}
          <div className="upload-card">
            <div className="upload-header">
              <h4>📄 Your CV Analysis</h4>
              <span className="status-indicator">
                {userProfile ? '✅ CV analyzed' : '⏳ No CV'}
              </span>
            </div>
            <p>Upload your CV to extract skills, goals, and experience level for matching.</p>
            <button 
              onClick={() => cvFileRef.current?.click()} 
              className="upload-btn cv-btn"
              disabled={isAnalyzing}
            >
              {isAnalyzing ? '🔍 Analyzing...' : '📄 Analyze CV'}
            </button>
            <input
              type="file"
              ref={cvFileRef}
              onChange={handleCVUpload}
              accept=".pdf,.doc,.docx"
              style={{ display: 'none' }}
            />
          </div>

          {/* Profile Summary */}
          {userProfile && (
            <div className="profile-summary">
              <h4>👤 Your Profile</h4>
              <div className="profile-item">
                <strong>Industry:</strong> {userProfile.industry}
              </div>
              <div className="profile-item">
                <strong>Experience:</strong> {userProfile.experience_level}
              </div>
              <div className="profile-item">
                <strong>Key Skills:</strong>
                <div className="skills-container">
                  {userProfile.skills.slice(0, 6).map((skill, index) => (
                    <span key={index} className="skill-tag">{skill}</span>
                  ))}
                </div>
              </div>
            </div>
          )}
        </div>

        {/* Right Section - Chat Interface (2/3) */}
        <div className="chat-section">
          <div className="chat-header">
            <h3>💬 Network Assistant</h3>
            <div className="chat-status">
              {csvData && userProfile ? '🟢 Ready for queries' : '🟡 Upload data to start'}
            </div>
          </div>

          <div className="messages-container">
            {messages.map((message) => (
              <div key={message.id} className={`message ${message.sender}`}>
                <div className="message-content">
                  <p className="message-text">{message.text}</p>
                  <span className="message-time">{message.timestamp}</span>
                  
                  {/* Data visualizations */}
                  {message.data && message.data.type === 'matches' && (
                    <div className="matches-viz">
                      <h4>Top Alumni Matches</h4>
                      <div className="matches-grid">
                        {message.data.data.matches.slice(0, 3).map((match, index) => (
                          <div key={index} className="match-card">
                            <h5>{match.name}</h5>
                            <p><strong>{match.company}</strong></p>
                            <p>{match.position}</p>
                            <div className="match-score">{match.match_score}%</div>
                          </div>
                        ))}
                      </div>
                    </div>
                  )}

                  {message.data && message.data.type === 'cv_analysis' && (
                    <div className="profile-viz">
                      <h4>Extracted Skills</h4>
                      <div className="skills-tags">
                        {message.data.data.skills.map((skill, index) => (
                          <span key={index} className="skill-tag">{skill}</span>
                        ))}
                      </div>
                    </div>
                  )}
                </div>
              </div>
            ))}
            
            {(isLoading || isAnalyzing) && (
              <div className="message agent">
                <div className="message-content">
                  <div className="typing-indicator">
                    <span></span>
                    <span></span>
                    <span></span>
                  </div>
                  <p className="loading-text">
                    {isAnalyzing ? 'Analyzing CV...' : 'Processing...'}
                  </p>
                </div>
              </div>
            )}
            <div ref={messagesEndRef} />
          </div>

          <div className="input-container">
            <textarea
              value={inputMessage}
              onChange={(e) => setInputMessage(e.target.value)}
              onKeyPress={handleKeyPress}
              placeholder="Ask me to find matches, show your skills, or analyze the network..."
              className="message-input"
              rows="1"
            />
            <button 
              onClick={handleSendMessage} 
              className="send-button"
              disabled={isLoading || isAnalyzing}
            >
              Send
            </button>
          </div>
        </div>
      </div>
    </div>
  );
}

export default CommunityNetworkAgent;