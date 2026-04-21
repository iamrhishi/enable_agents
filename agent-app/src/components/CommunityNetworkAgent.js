import { API_CONFIG } from '../config/apiConfig';
import React, { useState, useRef, useEffect } from 'react';
import ReactMarkdown from 'react-markdown';
import Header from './Header';
import '../styles/CommunityNetworkAgent.css';

function CommunityNetworkAgent() {
  const [inputMessage, setInputMessage] = useState('');
  const [isLoading, setIsLoading] = useState(false);
  const [csvData, setCsvData] = useState(null);
  const [userProfile, setUserProfile] = useState(null);
  const [isAnalyzing, setIsAnalyzing] = useState(false);
  const messagesEndRef = useRef(null);
  const csvFileRef = useRef(null);
  const cvFileRef = useRef(null);
  const [existingFiles, setExistingFiles] = useState(new Map());
  
  const [messages, setMessages] = useState([
    {
      id: 1,
      text: "Welcome to the Community Network Agent! Now I can help you enhance your network!",
      sender: 'agent',
      timestamp: new Date().toLocaleTimeString(),
      format: 'markdown'
    }
  ]);

  // Add these state variables at the top with other state
  const [currentUserId, setCurrentUserId] = useState('user_001'); // Replace with actual user ID
  const [userFavorites, setUserFavorites] = useState([]);

  const scrollToBottom = () => {
    messagesEndRef.current?.scrollIntoView({ behavior: "smooth" });
  };

  useEffect(() => {
    scrollToBottom();
  }, [messages]);

  // Updated addMessage function with format parameter
  const addMessage = (text, sender, data = null, format = 'markdown') => {
    const newMessage = {
      id: Date.now(),
      text,
      sender,
      timestamp: new Date().toLocaleTimeString(),
      data,
      format // 'html' or 'markdown'
    };
    setMessages(prev => [...prev, newMessage]);
  };

  // Custom message content renderer
  const MessageContent = ({ message }) => {
    if (message.format === 'html') {
      return (
        <div 
          className="message-text"
          dangerouslySetInnerHTML={{ __html: message.text }}
        />
      );
    } else {
      return (
        <div className="message-text">
          <ReactMarkdown>{message.text}</ReactMarkdown>
        </div>
      );
    }
  };

  // Helper function to check existing files and compare sizes
  const checkExistingFile = async (fileName, newFileSize) => {
    try {
      const response = await fetch(`${API_CONFIG.API_URL}/check_existing_file`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          file_name: fileName,
          new_file_size: newFileSize
        })
      });

      const result = await response.json();
      return result;
    } catch (error) {
      console.error('Error checking existing file:', error);
      return { exists: false, should_skip: false };
    }
  };

  // Function to save JSON data to file
  const saveJSONToFile = async (jsonData, originalFileName) => {
    try {
      const jsonFileName = originalFileName.replace(/\.(csv|xlsx|xls)$/i, '.json');
      
      const response = await fetch(`${API_CONFIG.API_URL}/save_json_file`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          data: jsonData,
          file_name: jsonFileName,
          folder_name: 'dataset'
        })
      });

      const result = await response.json();
      return result;
    } catch (error) {
      console.error('Error saving JSON file:', error);
      return { success: false, error: 'Failed to save JSON file' };
    }
  };

  // Enhanced handleSearch function with better user feedback
  const handleSearch = async (query) => {
    try {
      addMessage("🔍 Searching through the data...", 'agent', null, 'markdown');
      
      const response = await fetch(`${API_CONFIG.API_URL}/simple_search`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          query: query,
          data: csvData
        })
      });

      const result = await response.json();

      if (result.success) {
        if (result.total_found > 0) {
          // Show what keywords were extracted
          const keywordsText = Object.entries(result.keywords)
            .filter(([key, values]) => values && values.length > 0)
            .map(([key, values]) => `${key}: ${values.join(', ')}`)
            .join(' | ');
          
          const resultsText = formatResults(result.results);
          
          // Use HTML format for search results with formatted profiles
          addMessage(`🔍 <strong>Search Results</strong> (${result.total_found} found)<br><br><strong>Keywords extracted:</strong> ${keywordsText}<br><br>${resultsText}`, 'agent', {
            type: 'search_results',
            data: {
              query: query,
              results: result.results,
              total_found: result.total_found,
              keywords: result.keywords
            }
          }, 'html');
          
          // Show more results option if there are many
          if (result.total_found > 5) {
            addMessage(`💡 **Found ${result.total_found} total results.** Showing top 5. Try being more specific to narrow down results.`, 'agent', null, 'markdown');
          }
          
          // Suggest related searches if results are limited
          if (result.total_found < 3 && result.total_found > 0) {
            // addMessage("💡 **Want more results?** Try:\n• Using broader terms\n• Searching by different fields\n• Checking spelling of keywords", 'agent', null, 'markdown');
          }
        } else {
          // No results found - provide helpful suggestions
          const keywordsText = Object.entries(result.keywords)
            .filter(([key, values]) => values && values.length > 0)
            .map(([key, values]) => `${key}: ${values.join(', ')}`)
            .join(' | ');
          
          addMessage(`🔍 **No results found** for: "${query}"\n\n**Keywords I looked for:** ${keywordsText}\n\n💡 **Try:**\n• Different spelling or synonyms\n• Broader search terms\n• Different fields (company vs title vs skills)\n\n**Example searches:**\n• "Find engineers" (instead of "senior software engineers")\n• "People at tech companies"\n• "Alumni with programming skills"`, 'agent', null, 'markdown');
        }
      } else {
        addMessage(`**Search needs more specific input:** ${result.error}\n\nPlease try rephrasing your request e.g., mention company name or job title`, 'agent', null, 'markdown');
      }
    } catch (error) {
      console.error('Search error:', error);
      addMessage("❌ **Connection error.** Please check your connection and try again.", 'agent', null, 'markdown');
    }
  };

  // Updated CSV Upload Handler with file checking and JSON saving
  const handleCSVUpload = async (event) => {
    const file = event.target.files[0];
    if (!file) return;

    const allowedTypes = ['.csv', '.xlsx', '.xls'];
    const fileExtension = file.name.toLowerCase().substring(file.name.lastIndexOf('.'));
    
    if (!allowedTypes.includes(fileExtension)) {
      addMessage("Please upload a valid CSV or XLSX file.", 'agent', null, 'markdown');
      return;
    }

    setIsLoading(true);

    try {
      // Step 1: Check if file already exists and compare sizes
      const fileCheckResult = await checkExistingFile(file.name, file.size);
      
      if (fileCheckResult.should_skip) {
        // Load existing JSON data
        const jsonFileName = file.name.replace(/\.(csv|xlsx|xls)$/i, '.json');
        const loadResponse = await fetch(`${API_CONFIG.API_URL}/load_json_file`, {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({
            file_name: jsonFileName,
            folder_name: 'dataset'
          })
        });

        const loadResult = await loadResponse.json();
        
        if (loadResult.success) {
          setCsvData(loadResult.data);

          // Update existing files tracking
          setExistingFiles(prev => new Map(prev.set(file.name, {
            size: file.size,
            processed: true,
            json_file: jsonFileName,
            profiles_count: loadResult.data.length
          })));

          // Show a preview of the data structure
          const sampleProfile = loadResult.data[0];
          if (sampleProfile) {
            const availableFields = Object.keys(sampleProfile);
          }

          setIsLoading(false);
          event.target.value = '';
          return;
        }
      }

      // Step 2: File is new or has more data, proceed with conversion
      if (fileCheckResult.exists && !fileCheckResult.should_skip) {
        addMessage(`File exists but new version has more data (${file.size} vs ${fileCheckResult.existing_size} bytes). Updating...`, 'agent', null, 'markdown');
      } else {
        // addMessage("New file detected. Converting to JSON format...", 'agent', null, 'markdown');
      }

      // Step 3: Upload and convert file to JSON
      const formData = new FormData();
      formData.append('file', file);
      formData.append('folder_name', 'dataset');
      formData.append('multiple_sheets', 'false');

      const response = await fetch(`${API_CONFIG.API_URL}/file_to_json_convert`, {
        method: 'POST',
        body: formData
      });

      const conversionResult = await response.json();

      if (conversionResult.success) {
        const profiles = conversionResult.data;
        setCsvData(profiles);
        
        // Step 4: Save JSON file for future use
        // addMessage("Saving JSON file for faster future access...", 'agent', null, 'markdown');
        const saveResult = await saveJSONToFile(profiles, file.name);
        
        if (saveResult.success) {
          addMessage(`File processed and saved! Found ${profiles.length} profiles.`, 'agent', {
            type: 'csv_upload_success',
            data: { 
              profiles_count: profiles.length,
              columns: Object.keys(profiles[0] || {}),
              file_type: fileExtension.substring(1).toUpperCase(),
              json_saved: true,
              json_file: saveResult.file_path
            }
          }, 'markdown');

          // Update existing files tracking
          setExistingFiles(prev => new Map(prev.set(file.name, {
            size: file.size,
            processed: true,
            json_file: saveResult.file_name,
            profiles_count: profiles.length
          })));

        } else {
          addMessage(`File converted successfully! Found ${profiles.length} profiles. (JSON save failed: ${saveResult.error})`, 'agent', {
            type: 'csv_upload_success',
            data: { 
              profiles_count: profiles.length,
              columns: Object.keys(profiles[0] || {}),
              file_type: fileExtension.substring(1).toUpperCase(),
              json_saved: false
            }
          }, 'markdown');
        }

        // Show a preview of the data structure
        const sampleProfile = profiles[0];
        if (sampleProfile) {
          const availableFields = Object.keys(sampleProfile);
          // addMessage(`**Data Structure Preview:**\nAvailable fields: ${availableFields.join(', ')}\n\nYou can now search through this data!`, 'agent', null, 'markdown');
        }
        
      } else {
        addMessage(`❌ Error converting file: ${conversionResult.error}`, 'agent', null, 'markdown');
      }
    } catch (error) {
      console.error('File processing error:', error);
      addMessage("❌ Error processing file. Please try again.", 'agent', null, 'markdown');
    } finally {
      setIsLoading(false);
      event.target.value = '';
    }
  };

  // Add this simple matchmaking function
  const getCompanyRecommendations = async (skills, industry, csvData) => {
    try {
      // Extract unique companies from the data
      const companies = [...new Set(csvData.map(record => 
        record.company || record.Company || record.organization || record.Organization
      ).filter(Boolean))];

      const prompt = `Based on these skills: ${skills.join(', ')} and industry: ${industry}, suggest 8-10 companies from this list that would be most relevant: ${companies.join(', ')}. 

  Return only company names separated by commas.`;

      const response = await fetch(`${API_CONFIG.API_URL}/search_suggestions`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          prompt: prompt,
          max_tokens: 200,
          temperature: 0.3
        })
      });

      const result = await response.json();
      
      if (result.success) {
        // Parse company names from response
        const recommendedCompanies = result.response
          .split(',')
          .map(company => company.trim())
          .filter(company => company.length > 0);
        
        return recommendedCompanies;
      }
      return [];
    } catch (error) {
      console.error('Matchmaking error:', error);
      return [];
    }
  };

  // Update the handleCVUpload function - add matchmaking after profile creation:
  const handleCVUpload = async (event) => {
    const file = event.target.files[0];
    if (!file) return;

    if (!file.name.endsWith('.pdf')) {
      addMessage("Please upload a PDF document.", 'agent', null, 'markdown');
      return;
    }

    setIsAnalyzing(true);

    try {
      const formData = new FormData();
      formData.append('file', file);
      formData.append('folder_name', 'cv_analysis');

      const uploadResponse = await fetch(`${API_CONFIG.API_URL}/upload`, {
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

        const ragResponse = await fetch(`${API_CONFIG.API_URL}/rag_test`, {
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
          
          addMessage(`🎯 **We analyzed your CV!**\n\n💼 **Skills:** ${skillsText}\n\n🚀 **Goals:** ${goalsText}\n\n📊 **Industry:** ${profile.industry}`, 'agent', {
            type: 'cv_analysis',
            data: profile
          }, 'markdown');

          // ADD MATCHMAKING HERE - only if csvData is available
          if (csvData && csvData.length > 0) {
            const recommendedCompanies = await getCompanyRecommendations(
              profile.skills, 
              profile.industry, 
              csvData
            );

            if (recommendedCompanies.length > 0) {
              addMessage(`🏢 **Recommended Companies:** ${recommendedCompanies.join(', ')}\n\n💡 **Try searching:** "Find people at ${recommendedCompanies[0]}" or "Show engineers at ${recommendedCompanies[1]}"`, 'agent', null, 'markdown');
            }
          }

        } else {
          addMessage(`❌ Error analyzing CV: ${ragResult.error || 'Unknown error'}`, 'agent', null, 'markdown');
        }
      } else {
        addMessage(`❌ Error uploading CV: ${uploadResult.error}`, 'agent', null, 'markdown');
      }
    } catch (error) {
      console.error('CV Analysis Error:', error);
      addMessage("❌ Error analyzing CV. Please try again.", 'agent', null, 'markdown');
    } finally {
      setIsAnalyzing(false);
      event.target.value = '';
    }
  };

  // CV Skills parsing function
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

  // Enhanced message handler focused on data search and retrieval
  const handleSendMessage = async (e = null) => {
    if (e && typeof e.preventDefault === 'function') {
      e.preventDefault();
    }
    
    if (!inputMessage.trim()) return;

    const userMessage = inputMessage.trim();
    const userMessageLower = userMessage.toLowerCase();
    
    // Add user message to chat
    addMessage(userMessage, 'user', null, 'markdown');
    setInputMessage('');
    setIsLoading(true);

    // Add favorites check here
    if (userMessageLower.includes('favorites') || userMessageLower.includes('saved') || userMessageLower.includes('show favorites')) {
      showUserFavorites();
      setIsLoading(false);
      return;
    }

    // Check if data is available
    if (!csvData || csvData.length === 0) {
      setTimeout(() => {
        addMessage("Please upload a CSV/Excel file first so I can help you search through the data!\n\n💡 **Once you upload data, you can search like:**\n• 'Find people at Google'\n• 'Show me software engineers'\n• 'People with Python skills'\n• 'Alumni in San Francisco'", 'agent', null, 'markdown');
        setIsLoading(false);
      }, 500);
      return;
    }

    // Detect search queries
    const searchKeywords = [
      'find', 'search', 'show', 'get', 'who', 'people', 'alumni', 'working', 'at', 'with',
      'developers', 'engineers', 'managers', 'analysts', 'designers', 'company', 'companies',
      'skills', 'skill', 'technology', 'experience', 'location', 'city', 'country', 'in'
    ];
    
    const isSearchQuery = searchKeywords.some(keyword => 
      userMessageLower.includes(keyword)
    );

    // Handle search queries
    if (isSearchQuery) {
      try {
        await handleSearch(userMessage);
      } catch (error) {
        console.error('Search error:', error);
        addMessage("❌ Error performing search. Please try again.", 'agent', null, 'markdown');
      }
      setIsLoading(false);
      return;
    }

    // Handle general data questions and provide guidance
    setTimeout(() => {
      if (userMessageLower.includes('help') || userMessageLower.includes('what can you do')) {
        addMessage(`🔍 **I can help you search through your data!**\n\n**Search Examples:**\n• "Find people at Microsoft"\n• "Show marketing managers"\n• "People with React skills"\n• "Alumni in New York"\n• "Senior engineers at tech companies"\n• "Developers with Python and JavaScript"\n\n**Data Available:** ${csvData.length} profiles\n**Fields:** ${Object.keys(csvData[0] || {}).join(', ')}\n\nJust ask me naturally!`, 'agent', null, 'markdown');
      } else if (userMessageLower.includes('data') || userMessageLower.includes('profiles') || userMessageLower.includes('how many')) {
        // Show data summary
        const sampleProfile = csvData[0];
        const availableFields = Object.keys(sampleProfile);
        addMessage(`📊 **Data Summary:**\n\n**Total Profiles:** ${csvData.length}\n**Available Fields:** ${availableFields.join(', ')}\n\n**Sample Profile:**\n${formatSampleProfile(sampleProfile)}\n\n🔍 **Try searching:** "Find people at [company]" or "Show [job title]"`, 'agent', null, 'markdown');
      } else {
        // Default response - guide user to search
        addMessage(`I have ${csvData.length} profiles loaded and ready to search!\n\n🔍 **Try searching like:**\n• "Find software engineers"\n• "People working at Google"\n• "Show me data scientists"\n• "Alumni with machine learning skills"\n\n**Available fields:** ${Object.keys(csvData[0] || {}).join(', ')}\n\nWhat would you like to find?`, 'agent', null, 'markdown');
      }
      setIsLoading(false);
    }, 1000);
  };

  // Helper function to format sample profile
  const formatSampleProfile = (profile) => {
    const name = profile.name || profile.Name || profile.full_name || 'Unknown';
    const company = profile.company || profile.Company || profile.organization || 'Unknown';
    const title = profile.title || profile.Title || profile.position || 'Unknown';
    
    return `**Name:** ${name}\n**Company:** ${company}\n**Title:** ${title}`;
  };

  // Function to save profile to favorites
  const saveToFavorites = async (profileData) => {
    try {
      const response = await fetch(`${API_CONFIG.API_URL}/save_user_favorite`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          user_id: currentUserId,
          profile_data: profileData
        })
      });

      const result = await response.json();
      
      if (result.success) {
        addMessage(`✅ **Profile saved to favorites!**\n\n**${profileData.full_name}** from **${profileData.company}** added to your favorites.\n\n**Total favorites:** ${result.favorites_count}`, 'agent', null, 'markdown');
        // Refresh favorites list
        loadUserFavorites();
      } else {
        addMessage(`❌ **Error saving to favorites:** ${result.error}`, 'agent', null, 'markdown');
      }
    } catch (error) {
      console.error('Error saving to favorites:', error);
      addMessage("❌ **Error saving to favorites.** Please try again.", 'agent', null, 'markdown');
    }
  };

  // Function to load user favorites
  const loadUserFavorites = async () => {
    try {
      const response = await fetch(`${API_CONFIG.API_URL}/get_user_favorites`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          user_id: currentUserId
        })
      });

      const result = await response.json();
      
      if (result.success) {
        setUserFavorites(result.favorites);
      }
    } catch (error) {
      console.error('Error loading favorites:', error);
    }
  };

  // Updated formatResults function with PNG icons and adjusted card height
  const formatResults = (results) => {
    return results.slice(0, 5).map((result, index) => {
      const name = result.name || result.Name || result.full_name || 'Unknown';
      const lastname = result.Surname || '';
      const company = result.company || result.Company || result.organization || 'Unknown';
      const title = result['Job title'] || result.title || result.Title || result.position || result.role || 'Unknown';
      const location = result.location || result.Location || result.city || result.City || '';
      
      const full_name = `${name} ${lastname}`.trim(); 

      // Create LinkedIn search URL
      const linkedinSearchUrl = `https://www.linkedin.com/search/results/people/?keywords=${encodeURIComponent(full_name)}`;

      // Create unique ID for this result
      const resultId = `result_${Date.now()}_${index}`;

      let profileText = `<div style="margin-bottom: 16px; padding: 16px; border: 1px solid #e2e8f0; border-radius: 8px; position: relative; background: #fafafa;">
    <!-- Header Section with Name and Action Icons -->
    <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 12px;">
      <div style="flex: 1;">
        <strong style="font-size: 16px; color: #1f2937;">${full_name}</strong>
      </div>
      
      <!-- Action Icons in One Line -->
      <div style="display: flex; gap: 8px; align-items: center;">
        <a href="${linkedinSearchUrl}" target="_blank" style="text-decoration: none;" title="Search on LinkedIn">
          <img src="/assets/icons/linkedin.png" alt="LinkedIn" style="width:20px; height:20px; cursor:pointer; opacity:0.7; transition:opacity 0.2s;" onmouseover="this.style.opacity='1'" onmouseout="this.style.opacity='0.7'" />
        </a>
        
        <img src="/assets/icons/save.png" alt="Save" 
            onclick="window.saveProfileToFavorites('${resultId}')" 
            style="width:20px; height:20px; cursor:pointer; opacity:0.7; transition:opacity 0.2s;" 
            onmouseover="this.style.opacity='1'" 
            onmouseout="this.style.opacity='0.7'" 
            title="Save to favorites" />
        
        <img src="/assets/icons/message.png" alt="Message" 
            onclick="window.showMessageOptions('${JSON.stringify({full_name, company, title, location}).replace(/"/g, "&quot;")}')" 
            style="width:20px; height:20px; cursor:pointer; opacity:0.7; transition:opacity 0.2s;" 
            onmouseover="this.style.opacity='1'" 
            onmouseout="this.style.opacity='0.7'" 
            title="Generate outreach message" />
      </div>
    </div>
    
    <!-- Profile Information -->
    <div style="display: grid; grid-template-columns: 1fr; gap: 4px;">
      <div style="display: flex; align-items: center;">
        <span style="font-weight: 500; color: #374151; min-width: 70px; font-size: 14px;">Company:</span>
        <span style="color: #6b7280; font-size: 14px;">${company}</span>
      </div>
      
      <div style="display: flex; align-items: center;">
        <span style="font-weight: 500; color: #374151; min-width: 70px; font-size: 14px;">Title:</span>
        <span style="color: #6b7280; font-size: 14px;">${title}</span>
      </div>`;

      if (location) {
        profileText += `
      <div style="display: flex; align-items: center;">
        <span style="font-weight: 500; color: #374151; min-width: 70px; font-size: 14px;">Location:</span>
        <span style="color: #6b7280; font-size: 14px;">📍 ${location}</span>
      </div>`;
      }
      
      // Add skills if available
      const skills = result.skills || result.Skills || result.required_skills || result.technologies;
      if (skills) {
        const skillsText = typeof skills === 'string' ? skills : skills.join(', ');
        if (skillsText.length < 60) {
          profileText += `
      <div style="display: flex; align-items: flex-start; margin-top: 2px;">
        <span style="font-weight: 500; color: #374151; min-width: 70px; font-size: 14px;">Skills:</span>
        <span style="color: #6b7280; font-size: 13px;">🔧 ${skillsText}</span>
      </div>`;
        }
      }
      
      profileText += `
    </div>
  </div>`;
      
      // Store the result data for the save function
      if (!window.searchResults) window.searchResults = {};
      window.searchResults[resultId] = {
        name: name,
        lastname: lastname,
        company: company,
        title: title,
        location: location,
        skills: skills,
        full_name: full_name,
        // Include all original data
        ...result
      };
      
      return profileText;
    }).join('');
  };

  // Make save function available globally and load favorites on mount
  useEffect(() => {
    // Make save function available globally
    window.saveProfileToFavorites = (resultId) => {
      const profileData = window.searchResults[resultId];
      if (profileData) {
        saveToFavorites(profileData);
      } else {
        console.error('Profile data not found for ID:', resultId);
        addMessage("❌ **Error:** Profile data not found. Please try again.", 'agent', null, 'markdown');
      }
    };

    // Make remove function available globally
    window.removeFromFavorites = removeFromFavorites;
    
    // Load user favorites when component mounts
    loadUserFavorites();
    
    return () => {
      delete window.saveProfileToFavorites;
      delete window.removeFromFavorites;
      delete window.searchResults;
    };
  }, [currentUserId]);

  // Add function to show favorites (optional - for viewing saved favorites)
  const showUserFavorites = () => {
    if (userFavorites.length === 0) {
      addMessage("📭 **No favorites saved yet!**\n\nSave profiles by clicking the ❤️ **Mark as favorite** button in search results.", 'agent', null, 'markdown');
      return;
    }

    const favoritesHtml = userFavorites.map((favorite, index) => {
      const full_name = favorite.full_name || `${favorite.name || ''} ${favorite.lastname || ''}`.trim();
      const company = favorite.company || favorite.Company || 'Unknown';
      const title = favorite.title || favorite['Job title'] || favorite.Title || 'Unknown';
      const savedDate = new Date(favorite.saved_at).toLocaleDateString();
      
      // Create LinkedIn search URL
      const linkedinSearchUrl = `https://www.linkedin.com/search/results/people/?keywords=${encodeURIComponent(full_name)}`;

      return `<div style="margin-bottom: 16px; padding: 12px; border: 1px; border-radius: 8px;">
<strong>Name:</strong> ${full_name} <a href="${linkedinSearchUrl}" target="_blank" style="margin-left: 8px;"><img src="/assets/icons/linkedin.png" alt="LinkedIn" style="width:16px;height:16px;display:inline;vertical-align:middle;" /></a><br>
<strong>Company:</strong> ${company}<br>
<strong>Title:</strong> ${title}<br>
<small style="color: #666;">Saved on: ${savedDate}</small><br>
<button onclick="window.removeFromFavorites(${favorite.favorite_id})" style="background: #dc2626; color: white; border: none; padding: 4px 8px; border-radius: 4px; cursor: pointer; margin-top: 8px; font-size: 12px;">🗑️ Remove</button>
</div>`;
    }).join('');

    addMessage(`<strong>Your Favorites</strong> (${userFavorites.length} saved)<br><br>${favoritesHtml}`, 'agent', {
      type: 'user_favorites',
      data: { favorites: userFavorites, count: userFavorites.length }
    }, 'html');
  };

  // Fix 3: Add the missing handleKeyPress function
  const handleKeyPress = (e) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      handleSendMessage();
    }
  };

  // Fix 4: Add the missing removeFromFavorites function
  const removeFromFavorites = async (favoriteId) => {
    try {
      const response = await fetch(`${API_CONFIG.API_URL}/remove_user_favorite`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          user_id: currentUserId,
          favorite_id: favoriteId
        })
      });

      const result = await response.json();
      
      if (result.success) {
        addMessage(`✅ **Profile removed from favorites!**\n\n**Total favorites:** ${result.favorites_count}`, 'agent', null, 'markdown');
        // Refresh favorites list
        loadUserFavorites();
      } else {
        addMessage(`❌ **Error removing from favorites:** ${result.error}`, 'agent', null, 'markdown');
      }
    } catch (error) {
      console.error('Error removing from favorites:', error);
      addMessage("❌ **Error removing from favorites.** Please try again.", 'agent', null, 'markdown');
    }
  };

  return (
    <div className="community-network-agent">
      <Header />
      
      <div className="main-container">
        {/* Left Section - File Uploads */}
        <div className="upload-section">
          <h3>Connect Network Data</h3>
          
          {/* CSV Upload */}
          <div className="upload-card">
            <div className="upload-header">
              <h4>Network Data</h4>
              <span className="status-indicator">
                {csvData ? `${csvData.length} profiles loaded` : 'No data'}
              </span>
            </div>
            <p>Upload network data or ask the Admin for access. Write at <strong>engineering@enableyou.co</strong> for data access</p>
            <button 
              onClick={() => csvFileRef.current?.click()} 
              className="upload-btn csv-btn"
              disabled={isLoading}
            >
              {isLoading ? 'Processing...' : 'Connect Dataset'}
            </button>
            <input
              type="file"
              ref={csvFileRef}
              onChange={handleCSVUpload}
              accept=".csv,.xlsx,.xls"
              style={{ display: 'none' }}
            />
          </div>

          {/* Data Preview Table */}
          {csvData && csvData.length > 0 && (
            <div className="data-preview">
              <h4>Data Preview</h4>
              
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
              <h4>Your Profile Analysis</h4>
              <span className="status-indicator">
                {userProfile ? 'Profile analyzed' : 'No Profile'}
              </span>
            </div>
            <p>Upload your Profile to help us understand your skills, goals, and experience level.</p>
            <button 
              onClick={() => cvFileRef.current?.click()} 
              className="upload-btn cv-btn"
              disabled={isAnalyzing}
            >
              {isAnalyzing ? 'Analyzing...' : 'Upload Your Profile'}
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

        {/* Right Section - Chat Interface */}
        <div className="chat-section">
          <div className="chat-header">
            <h3> Professional Network Agent</h3>
            <div className="chat-status">
              {csvData && userProfile ? '🟢 Connected to data source' : 
               csvData ? '🟡 Upload CV to continue' : '🔴 Upload data to start'}
            </div>
          </div>

          <div className="messages-container">
            {messages.map((message) => (
              <div key={message.id} className={`message ${message.sender}`}>
                <div className="message-content">
                  <MessageContent message={message} />
                  <span className="message-time">{message.timestamp}</span>

                  {/* CV Analysis visualization */}
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

                  {/* File upload success visualization
                  {message.data && message.data.type === 'csv_upload_success' && (
                    <div className="upload-viz">
                      <h4>📊 File Processing Summary</h4>
                      <div className="summary-stats">
                        <div className="stat-item">
                          <strong>Profiles:</strong> {message.data.data.profiles_count}
                        </div>
                        <div className="stat-item">
                          <strong>Format:</strong> {message.data.data.file_type}
                        </div>
                        <div className="stat-item">
                          <strong>JSON Saved:</strong> {message.data.data.json_saved ? '✅' : '❌'}
                        </div>
                      </div>
                    </div>
                  )} */}
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
              placeholder="You can ask the Network agent to find relevant people in your network, after you connect network data and analyse your CV"
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