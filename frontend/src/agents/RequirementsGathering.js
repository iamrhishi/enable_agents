import React, { useState, useEffect, useCallback } from 'react';
import * as XLSX from 'xlsx';
import { jsPDF } from 'jspdf';
import autoTable from 'jspdf-autotable';
import Header from '../core/Header';
import '../styles/RequirementsGathering.css';
import { API_CONFIG } from '../config/apiConfig';
import { authJsonHeaders, authOptionalHeaders } from '../core/authHeaders';
import { getAgentData, setAgentData, AGENT_KEYS } from '../utils';

const SUPPLIER_TEMPLATE = `Dear [Vendor Name / Sir / Madam],

Greetings from [Your Company Name].

We came across your company profile and would like to explore the possibility of working together for our ongoing and upcoming business requirements.

We are currently looking for reliable vendors/suppliers for the supply of [Product/Service Category]. We request you to share your company profile and the following details for our evaluation process:

* Product / Service Catalog
* Pricing / Quotation
* GST and Registration Details
* Payment Terms
* Delivery Timelines
* Major Clients / References
* Certifications (if applicable)

Company Details:

* Company Name: [Your Company Name]
* Industry: [Industry Type]
* Location: [Company Location]
* Contact Person: [Your Name]
* Contact Number: [Phone Number]
* Email: [Email Address]

Please feel free to contact us for any further clarification. We look forward to a mutually beneficial business relationship.

Best Regards,
[Your Company Name]
[Company Website]`;

const fillTemplate = (template, vars = {}) => {
  let out = template;
  const replacements = {
    '\\[Vendor Name \\/ Sir \\/ Madam\\]': vars.vendorName || 'Sir / Madam',
    '\\[Vendor Name\\]': vars.vendorName || 'Sir / Madam',
    '\\[Your Company Name\\]': vars.companyName || '',
    '\\[Industry Type\\]': vars.industry || '',
    '\\[Company Location\\]': vars.location || '',
    '\\[Your Name\\]': vars.contactName || '',
    '\\[Phone Number\\]': vars.phoneNumber || '',
    '\\[Email Address\\]': vars.emailAddress || '',
    '\\[Product\\/Service Category\\]': vars.productCategory || '',
    '\\[Company Website\\]': vars.companyWebsite || ''
  };
  Object.keys(replacements).forEach((pat) => {
    out = out.replace(new RegExp(pat, 'gi'), replacements[pat]);
  });
  return out;
};

// Demo mode mock data - realistic examples (persisted in sessionStorage)
const DEMO_MOCK_DATA = {
  overview: 'B2B SaaS platform for HR automation',
  industries: 'Technology',
  countries: 'North America',
  responseFormat: 'Customer Research',
  results: {
    query: 'B2B SaaS platform for HR automation',
    location: 'North America',
    industry: 'Technology',
    totalResults: 8,
    researchType: 'Customer Research',
    businesses: [
      { name: 'TechFlow Solutions', address: 'San Francisco, CA', website: 'https://techflow.io', phone: '+1 (415) 555-0123', email: 'contact@techflow.io', linkedin: 'https://linkedin.com/company/techflow', match_score: 92, summary: 'Leading HR automation platform' },
      { name: 'CloudHR Systems', address: 'Austin, TX', website: 'https://cloudhr.com', phone: '+1 (512) 555-0456', email: 'info@cloudhr.com', linkedin: 'https://linkedin.com/company/cloudhr', match_score: 88, summary: 'Cloud-based HR solutions' },
      { name: 'PeopleFirst Inc', address: 'Seattle, WA', website: 'https://peoplefirst.io', phone: '+1 (206) 555-0789', email: 'sales@peoplefirst.io', linkedin: 'https://linkedin.com/company/peoplefirst', match_score: 85, summary: 'Employee experience platform' },
      { name: 'WorkStream AI', address: 'New York, NY', website: 'https://workstream.ai', phone: '+1 (212) 555-0321', email: 'hello@workstream.ai', linkedin: 'https://linkedin.com/company/workstream', match_score: 91, summary: 'AI-powered workforce management' },
      { name: 'HRNova Solutions', address: 'Boston, MA', website: 'https://hrnova.com', phone: '+1 (617) 555-0654', email: 'contact@hrnova.com', linkedin: 'https://linkedin.com/company/hrnova', match_score: 94, summary: 'Enterprise HR transformation' },
      { name: 'Talent Dynamics', address: 'Denver, CO', website: 'https://talentdynamics.co', phone: '+1 (303) 555-0987', email: 'info@talentdynamics.co', linkedin: 'https://linkedin.com/company/talentdynamics', match_score: 79, summary: 'Recruiting and talent acquisition' },
      { name: 'PayrollPro Systems', address: 'Chicago, IL', website: 'https://payrollpro.io', phone: '+1 (312) 555-0147', email: 'sales@payrollpro.io', linkedin: 'https://linkedin.com/company/payrollpro', match_score: 82, summary: 'Payroll and benefits automation' },
      { name: 'BenefitHub Corp', address: 'Atlanta, GA', website: 'https://benefithub.com', phone: '+1 (404) 555-0258', email: 'team@benefithub.com', linkedin: 'https://linkedin.com/company/benefithub', match_score: 77, summary: 'Employee benefits management' },
    ],
    summary: {
      totalLeads: 8,
      topIndustries: ['HR Technology', 'Enterprise Software', 'AI/ML'],
      avgRating: 4.6,
      region: 'North America'
    }
  },
  savedLists: [
    { id: 'demo-1', name: 'Tech Startups Q1', query_used: 'B2B SaaS startups', created_at: '2026-06-15', lead_count: 24, status: 'active' },
    { id: 'demo-2', name: 'Enterprise HR Leads', query_used: 'Enterprise HR software', created_at: '2026-06-20', lead_count: 18, status: 'active' },
    { id: 'demo-3', name: 'West Coast Prospects', query_used: 'Tech companies California', created_at: '2026-06-25', lead_count: 32, status: 'active' },
  ]
};

// Helper to load/save demo state from sessionStorage
const loadDemoState = () => {
  try {
    const saved = sessionStorage.getItem('demoModeData');
    return saved ? JSON.parse(saved) : null;
  } catch { return null; }
};

const saveDemoState = (data) => {
  try {
    sessionStorage.setItem('demoModeData', JSON.stringify(data));
  } catch { /* ignore */ }
};

function RequirementsGathering() {
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

  // Restore demo state on mount if in demo mode (auto-load if no saved state)
  useEffect(() => {
    if (isDemoMode) {
      const savedDemo = loadDemoState();
      if (savedDemo && savedDemo.isDemo) {
        setOverview(savedDemo.overview || '');
        setIndustries(savedDemo.industries || '');
        setCountries(savedDemo.countries || '');
        setResponseFormat(savedDemo.responseFormat || '');
        if (savedDemo.results) {
          setCustomerResearchResults(savedDemo.results);
        }
      } else {
        // Auto-load demo data if no saved state exists
        setOverview(DEMO_MOCK_DATA.overview);
        setIndustries(DEMO_MOCK_DATA.industries);
        setCountries(DEMO_MOCK_DATA.countries);
        setResponseFormat(DEMO_MOCK_DATA.responseFormat);
        setCustomerResearchResults(DEMO_MOCK_DATA.results);
        // Persist for tab switches
        saveDemoState({
          overview: DEMO_MOCK_DATA.overview,
          industries: DEMO_MOCK_DATA.industries,
          countries: DEMO_MOCK_DATA.countries,
          responseFormat: DEMO_MOCK_DATA.responseFormat,
          results: DEMO_MOCK_DATA.results,
          savedLists: DEMO_MOCK_DATA.savedLists,
          isDemo: true
        });
      }
    }
  }, [isDemoMode]);

  const [overview, setOverview] = useState('');
  const [context, setContext] = useState('');
  const [countries, setCountries] = useState('');
  const [industries, setIndustries] = useState('');
  const [businessFunctions, setBusinessFunctions] = useState('');
  const [analysisFrameworks, setAnalysisFrameworks] = useState('');
  const [responseFormat, setResponseFormat] = useState('');
  const [uploadedFile, setUploadedFile] = useState(null);
  const [aiRequirements, setAiRequirements] = useState([]);
  const [previousPrompts, setPreviousPrompts] = useState([]);
  const [showPromptsPopup, setShowPromptsPopup] = useState(false);
  const [showPopup, setShowPopup] = useState(false);
  const [googleBusinessConnected, setGoogleBusinessConnected] = useState(false);
  // Note: These are initialized to null/false and loaded by useEffect based on mode
  const [customerResearchResults, setCustomerResearchResults] = useState(null);
  const [showCustomerResearchTable, setShowCustomerResearchTable] = useState(false);
  const [minimizedCustomerResearch, setMinimizedCustomerResearch] = useState(false);

  // Email Modal State
  const [showEmailModal, setShowEmailModal] = useState(false);
  const [campaignName, setCampaignName] = useState('');
  const [emailSubject, setEmailSubject] = useState('');
  const [emailBody, setEmailBody] = useState('');
  const [isSendingEmails, setIsSendingEmails] = useState(false);
  const [isGeneratingEmail, setIsGeneratingEmail] = useState({});
  const [selectedLead, setSelectedLead] = useState(null);
  const [useAiBulk, setUseAiBulk] = useState(false);
  const [existingCampaigns, setExistingCampaigns] = useState([]);
  const [selectedCampaignId, setSelectedCampaignId] = useState('');
  const [isAddingNewCampaign, setIsAddingNewCampaign] = useState(false);
  const [emailImages, setEmailImages] = useState([]);

  // Save market research data to centralized mode storage
  useEffect(() => {
    if (customerResearchResults !== null || showCustomerResearchTable || minimizedCustomerResearch) {
      setAgentData(AGENT_KEYS.MARKET_RESEARCH, {
        results: customerResearchResults,
        showTable: showCustomerResearchTable,
        minimized: minimizedCustomerResearch,
        overview,
        industries,
        countries,
        responseFormat,
      }, isDemoMode);
    }
  }, [customerResearchResults, showCustomerResearchTable, minimizedCustomerResearch, overview, industries, countries, responseFormat, isDemoMode]);

  // Load data when mode changes
  useEffect(() => {
    const savedData = getAgentData(AGENT_KEYS.MARKET_RESEARCH, isDemoMode);

    if (isDemoMode && !savedData?.results) {
      // Demo mode with no saved data - load demo defaults
      setCustomerResearchResults(DEMO_MOCK_DATA.results);
      setShowCustomerResearchTable(true);
      setOverview(DEMO_MOCK_DATA.overview);
      setIndustries(DEMO_MOCK_DATA.industries);
      setCountries(DEMO_MOCK_DATA.countries);
      setResponseFormat(DEMO_MOCK_DATA.responseFormat);
    } else if (savedData) {
      setCustomerResearchResults(savedData.results || null);
      setShowCustomerResearchTable(savedData.showTable || false);
      setMinimizedCustomerResearch(savedData.minimized || false);
      if (savedData.overview) setOverview(savedData.overview);
      if (savedData.industries) setIndustries(savedData.industries);
      if (savedData.countries) setCountries(savedData.countries);
      if (savedData.responseFormat) setResponseFormat(savedData.responseFormat);
    } else {
      // Live mode with no data - clear everything
      setCustomerResearchResults(null);
      setShowCustomerResearchTable(false);
      setMinimizedCustomerResearch(false);
      setOverview('');
      setIndustries('');
      setCountries('');
      setResponseFormat('');
    }
  }, [isDemoMode]);

  // Fetch existing campaigns when email modal opens
  useEffect(() => {
    if (showEmailModal && !selectedLead) {
      fetchExistingCampaigns();
    }
  }, [showEmailModal]);

  const fetchExistingCampaigns = async () => {
    try {
      const resp = await fetch(`${API_CONFIG.GET_CAMPAIGNS}?username=${encodeURIComponent(getCurrentUsername())}`, {
        method: 'GET',
        headers: { 'Content-Type': 'application/json' }
      });
      if (resp.ok) {
        const data = await resp.json();
        if (data.campaigns) {
          setExistingCampaigns(data.campaigns);
        }
      }
    } catch (error) {
      console.error('Error fetching campaigns:', error);
    }
  };

  const handleCampaignSelect = async (campaignId) => {
    if (campaignId === 'new') {
      setIsAddingNewCampaign(true);
      setSelectedCampaignId('');
      setCampaignName('');
      setEmailSubject('');
    } else {
      setIsAddingNewCampaign(false);
      const campaign = existingCampaigns.find(c => c.id === campaignId);
      if (campaign) {
        setSelectedCampaignId(campaignId);
        setCampaignName(campaign.name);
        setEmailSubject(campaign.subject || '');
      }
    }
  };

  const handleImageUpload = (e) => {
    const files = Array.from(e.target.files || []);
    files.forEach(file => {
      const reader = new FileReader();
      reader.onload = (event) => {
        setEmailImages(prev => [...prev, {
          name: file.name,
          data: event.target?.result
        }]);
      };
      reader.readAsDataURL(file);
    });
  };

  const removeEmailImage = (index) => {
    setEmailImages(prev => prev.filter((_, i) => i !== index));
  };

  const insertImageIntoBody = (index) => {
    if (emailImages[index]) {
      const imageMarkdown = `\n<img src="${emailImages[index].data}" alt="${emailImages[index].name}" style="max-width: 100%; height: auto; border-radius: 4px;" />\n`;
      setEmailBody(prev => prev + imageMarkdown);
    }
  };

  const [isLoadingResearch, setIsLoadingResearch] = useState(false);
  const [isLoadingEmails, setIsLoadingEmails] = useState(false);
  const [extractingEmailRows, setExtractingEmailRows] = useState({})
  const [extractingLinkedInRows, setExtractingLinkedInRows] = useState({});
  const [showIntegrationModal, setShowIntegrationModal] = useState(false);
  const [showExportModal, setShowExportModal] = useState(false);
  const [showSaveListModal, setShowSaveListModal] = useState(false);
  const [saveListName, setSaveListName] = useState('');
  const [saveListMode, setSaveListMode] = useState('create');
  const [selectedAppendProjectId, setSelectedAppendProjectId] = useState('');
  const [isSavingList, setIsSavingList] = useState(false);
  const [showScoreModal, setShowScoreModal] = useState(false);
  const [scoreQueryText, setScoreQueryText] = useState('');
  const [isScoring, setIsScoring] = useState(false);
  const [showSavedListsView, setShowSavedListsView] = useState(false);
  const [savedLists, setSavedLists] = useState([]);
  const [isLoadingSavedLists, setIsLoadingSavedLists] = useState(false);
  const [activeSavedList, setActiveSavedList] = useState(null);
  const [activeSavedListLeads, setActiveSavedListLeads] = useState([]);
  const [deletingListId, setDeletingListId] = useState(null);

  const [extractionUsage, setExtractionUsage] = useState(null);
  const [googleBusinessForm, setGoogleBusinessForm] = useState({
    clientId: '',
    redirectUri: '',
    hasCredentials: false
  });

  const getCurrentUsername = () => {
    return localStorage.getItem('username') || localStorage.getItem('firstName') || 'anonymous';
  };
  const getCurrentUserEmail = () => {
    return localStorage.getItem('userEmail') || '';
  };

  const getResearchEntityMeta = () => {
    const mode = customerResearchResults?.researchType || activeSavedList?.researchType || responseFormat;
    return mode === 'Supplier Research'
      ? { singular: 'vendor', plural: 'vendors', title: 'Vendors' }
      : { singular: 'lead', plural: 'leads', title: 'Leads' };
  };

  // Check URL params for tab and Google OAuth
  useEffect(() => {
    const params = new URLSearchParams(window.location.search);

    // Handle tab param from Campaign Dashboard navigation
    if (params.get('tab') === 'saved') {
      setShowSavedListsView(true);
      fetchSavedLists();
      // Clean up URL
      window.history.replaceState({}, document.title, window.location.pathname);
    }

    if (params.get('google_connected') === 'true') {
      setGoogleBusinessConnected(true);
      alert('Google Business Account connected successfully!');
      // Clean up URL
      window.history.replaceState({}, document.title, window.location.pathname);
    }
    
    // Fetch pre-configured Google credentials from .env
    const fetchCredentials = async () => {
      try {
        const response = await fetch(API_CONFIG.GET_GOOGLE_CREDENTIALS);
        const data = await response.json();
        
        if (data.success && data.credentials) {
          setGoogleBusinessForm({
            clientId: data.credentials.clientId || '',
            redirectUri: data.credentials.redirectUri || '',
            hasCredentials: data.credentials.hasCredentials || false
          });
          
          // If credentials are configured in .env, mark as connected
          if (data.credentials.hasCredentials) {
            setGoogleBusinessConnected(true);
          }
        }
      } catch (error) {
        console.error('Error fetching credentials:', error);
      }
    };
    
    fetchCredentials();

    const fetchEmailUsage = async () => {
      try {
        const username = getCurrentUsername();
        const response = await fetch(`${API_CONFIG.EMAIL_EXTRACTION_USAGE}?username=${encodeURIComponent(username)}`);

        const data = await response.json();
        if (response.ok && data.success && data.usageSummary) {
          setExtractionUsage(data.usageSummary);
        }
      } catch (error) {
        console.error('Error fetching email extraction usage:', error);
      }
    };

    fetchEmailUsage();
  }, []);

  const handleFileUpload = (e) => {
    const file = e.target.files[0];
    setUploadedFile(file);
    console.log('Uploaded file:', file);
  };

  const handleGenerateAnalysisFrameworks = (e) => {
    setAnalysisFrameworks(e.target.value);
  };

  const handleGenerate = async () => {
    try {
      // Check if Customer Research or Supplier Research format is selected
      if (responseFormat === 'Customer Research' || responseFormat === 'Supplier Research') {
        // Allow customer research even when OAuth is not connected.
        // Backend can run this flow via Google Places API key.

        // Validate required inputs for research
        if (!overview || !industries || !countries) {
          alert('Please fill in Overview, Industries, and Region/Countries for research');
          return;
        }

        // In demo mode, use mock data instead of API call
        if (isDemoMode) {
          setCustomerResearchResults({
            query: overview,
            location: countries,
            industry: industries,
            totalResults: DEMO_MOCK_DATA.results.businesses.length,
            researchType: responseFormat,
            businesses: DEMO_MOCK_DATA.results.businesses
          });
          setShowCustomerResearchTable(true);
          // Persist demo state
          saveDemoState({
            overview,
            industries,
            countries,
            responseFormat,
            results: {
              query: overview,
              location: countries,
              industry: industries,
              totalResults: DEMO_MOCK_DATA.results.businesses.length,
              researchType: responseFormat,
              businesses: DEMO_MOCK_DATA.results.businesses
            },
            savedLists: DEMO_MOCK_DATA.savedLists,
            isDemo: true
          });
          return;
        }

        setIsLoadingResearch(true);

        // Call the search-google-businesses API
        const searchResponse = await fetch(API_CONFIG.SEARCH_GOOGLE_BUSINESSES, {
          method: 'POST',
          headers: {
            'Content-Type': 'application/json',
          },
          body: JSON.stringify({
            query: overview, // Use overview as the search query
            location: countries, // Use countries as location
            page_size: 200 // Get 200 matching businesses
          }),
        });

        if (!searchResponse.ok) {
          const errorData = await searchResponse.json();
          const errorMessage = errorData.error || 'Failed to fetch research data';
          console.error('Search API error:', errorData);
          throw new Error(errorMessage);
        }

        const searchData = await searchResponse.json();

        if (!searchData.success) {
          alert(`Error: ${searchData.error}`);
          setIsLoadingResearch(false);
          return;
        }

        // Store results and show table
        setCustomerResearchResults({
          query: overview,
          location: countries,
          industry: industries,
          context: context,
          researchType: responseFormat,
          businesses: searchData.businesses || [],
          totalResults: searchData.totalResults || 0
        });

        setShowCustomerResearchTable(true);
        setIsLoadingResearch(false);
        return;
      }

      // Original behavior for other response formats
      const googleData = await fetchGoogleBusinessData();

      const payload = {
        overview,
        context,
        countries,
        industries,
        businessFunctions,
        analysisFrameworks,
        responseFormat,
        googleBusinessData: googleData,
      };

      const response = await fetch(API_CONFIG.GENERATE_REQUIREMENTS, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify(payload),
      });

      if (!response.ok) {
        throw new Error('Failed to generate requirements');
      }

      const data = await response.json();
      setAiRequirements(data.requirements.split('\n'));
    } catch (error) {
      console.error('Error generating requirements:', error);
      alert('Error: ' + error.message);
      setIsLoadingResearch(false);
    }
  };

  



  const handleFetchPreviousPrompts = async () => {
    try {
      const response = await fetch(API_CONFIG.PREVIOUS_PROMPTS, {
        method: 'GET',
      });

      if (!response.ok) {
        throw new Error('Failed to fetch previous prompts');
      }

      const data = await response.json();
      setPreviousPrompts(data.prompts);
      setShowPromptsPopup(true);
    } catch (error) {
      console.error('Error fetching previous prompts:', error);
    }
  };

  const handleGetEmails = async () => {
    if (!customerResearchResults || !customerResearchResults.businesses || customerResearchResults.businesses.length === 0) {
      const { plural } = getResearchEntityMeta();
      alert(`No ${plural} to enrich with emails`);
      return;
    }

    // In demo mode, data already has emails
    if (isDemoMode) {
      alert('Demo mode: Email data is already populated in the sample data.');
      return;
    }

    setIsLoadingEmails(true);

    try {
      const response = await fetch(API_CONFIG.ENRICH_BUSINESSES_WITH_EMAILS, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          businesses: customerResearchResults.businesses,
          username: getCurrentUsername()
        }),
      });

      if (!response.ok) {
        const errorData = await response.json();
        throw new Error(errorData.error || 'Failed to enrich businesses with emails');
      }

      const enrichedData = await response.json();
      if (enrichedData.usageSummary) {
        setExtractionUsage(enrichedData.usageSummary);
      }

      if (enrichedData.success && enrichedData.businesses) {
        // Update the research results with enriched businesses
        setCustomerResearchResults({
          ...customerResearchResults,
          businesses: enrichedData.businesses
        });
        const usageLine = enrichedData.usageSummary
          ? `\nUsed: ${enrichedData.usageSummary.usedCount}/${enrichedData.usageSummary.totalAllowed} | Remaining: ${enrichedData.usageSummary.remainingCount}`
          : '';
        alert(`Successfully enriched ${enrichedData.enrichedCount} businesses with email data!${usageLine}`);
      } else {
        alert('Failed to enrich businesses with emails');
      }
    } catch (error) {
      console.error('Error getting emails:', error);
      alert(`Error: ${error.message}`);
    } finally {
      setIsLoadingEmails(false);
    }
  };

  const handleExtractLinkedInForBusiness = async (business, index) => {
    if (!business) return;

    // In demo mode, data already has LinkedIn
    if (isDemoMode) {
      alert('Demo mode: LinkedIn data is already populated in the sample data.');
      return;
    }

    setExtractingLinkedInRows((prev) => ({ ...prev, [index]: true }));
    console.log(`[LINKEDIN_EXTRACTION] Starting extraction for ${business.name}`);

    try {
      const controller = new AbortController();
      const timeoutId = setTimeout(() => {
        console.error('[LINKEDIN_EXTRACTION] Request timeout after 30s');
        controller.abort();
      }, 30000); // 30 second timeout

      console.log(`[LINKEDIN_EXTRACTION] Calling ${API_CONFIG.ENRICH_BUSINESSES_WITH_LINKEDIN}`);

      const response = await fetch(API_CONFIG.ENRICH_BUSINESSES_WITH_LINKEDIN, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ businesses: [business], username: getCurrentUsername() }),
        signal: controller.signal
      });

      clearTimeout(timeoutId);
      console.log(`[LINKEDIN_EXTRACTION] Response status: ${response.status}`);

      if (!response.ok) {
        const errorData = await response.json().catch(() => ({}));
        const errorMsg = errorData.error || `HTTP ${response.status}`;
        console.error(`[LINKEDIN_EXTRACTION] API Error: ${errorMsg}`);
        throw new Error(`Failed to extract LinkedIn: ${errorMsg}`);
      }

      const data = await response.json();
      console.log('[LINKEDIN_EXTRACTION] Response received:', data);

      if (data.success && data.data && data.data.businesses && data.data.businesses.length > 0) {
        const enrichedBusiness = data.data.businesses[0];
        console.log(`[LINKEDIN_EXTRACTION] Extracted LinkedIn: ${enrichedBusiness.linkedin}`);
        
        // Update either saved list or customer research results
        if (activeSavedList && activeSavedListLeads.length > 0) {
          setActiveSavedListLeads(prev => {
            const updated = [...prev];
            updated[index] = { ...updated[index], linkedin: enrichedBusiness.linkedin };
            return updated;
          });
          console.log('[LINKEDIN_EXTRACTION] Updated saved list leads');
        } else {
          setCustomerResearchResults(prev => {
            if (!prev) {
              console.warn('[LINKEDIN_EXTRACTION] No customer research results to update');
              return prev;
            }
            const updatedBusinesses = [...prev.businesses];
            updatedBusinesses[index] = { ...updatedBusinesses[index], linkedin: enrichedBusiness.linkedin };
            console.log('[LINKEDIN_EXTRACTION] Updated customer research results');
            return { ...prev, businesses: updatedBusinesses };
          });
        }
      } else {
        const errorMsg = data.error || 'No LinkedIn profile found.';
        console.warn(`[LINKEDIN_EXTRACTION] ${errorMsg}`);
        alert(errorMsg);
      }
    } catch (error) {
      console.error('[LINKEDIN_EXTRACTION] Error extracting LinkedIn:', error);
      console.error('[LINKEDIN_EXTRACTION] Error details:', {
        message: error.message,
        name: error.name,
        stack: error.stack
      });
      
      if (error.name === 'AbortError') {
        alert('LinkedIn extraction timed out. Please try again.');
      } else {
        alert('Error extracting LinkedIn. Check console.');
      }
    } finally {
      setExtractingLinkedInRows((prev) => ({ ...prev, [index]: false }));
      console.log(`[LINKEDIN_EXTRACTION] Finished extraction for ${business.name}`);
    }
  };

  const handleExtractEmailForBusiness = async (business, index) => {
    if (!business || !business.website) {
      alert('Website not available for this business.');
      return;
    }

    // In demo mode, data already has emails
    if (isDemoMode) {
      alert('Demo mode: Email data is already populated in the sample data.');
      return;
    }

    setExtractingEmailRows((prev) => ({ ...prev, [index]: true }));
    console.log(`[EMAIL_EXTRACTION] Starting extraction for ${business.name}`);

    try {
      const controller = new AbortController();
      const timeoutId = setTimeout(() => {
        console.error('[EMAIL_EXTRACTION] Request timeout after 30s');
        controller.abort();
      }, 30000); // 30 second timeout

      console.log(`[EMAIL_EXTRACTION] Calling ${API_CONFIG.ENRICH_BUSINESSES_WITH_EMAILS}`);
      
      const response = await fetch(API_CONFIG.ENRICH_BUSINESSES_WITH_EMAILS, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          businesses: [business],
          username: getCurrentUsername()
        }),
        signal: controller.signal
      });

      clearTimeout(timeoutId);

      console.log(`[EMAIL_EXTRACTION] Response status: ${response.status}`);

      if (!response.ok) {
        const errorData = await response.json().catch(() => ({}));
        const errorMsg = errorData.error || `HTTP ${response.status}`;
        console.error(`[EMAIL_EXTRACTION] API Error: ${errorMsg}`);
        throw new Error(`Failed to extract email: ${errorMsg}`);
      }

      const enrichedData = await response.json();
      console.log('[EMAIL_EXTRACTION] Response received:', enrichedData);
      
      if (enrichedData.usageSummary) {
        setExtractionUsage(enrichedData.usageSummary);
      }
      
      const enrichedBusiness = enrichedData?.businesses?.[0];

      if (!enrichedBusiness) {
        console.error('[EMAIL_EXTRACTION] No enriched business data in response');
        throw new Error('No enriched business data returned');
      }

      console.log(`[EMAIL_EXTRACTION] Extracted email: ${enrichedBusiness.email}`);

      // Update either customer research results or activated list
      if (activeSavedList && activeSavedListLeads.length > 0) {
        setActiveSavedListLeads((prev) => {
          const updated = [...prev];
          updated[index] = {
            ...updated[index],
            email: enrichedBusiness.email || 'N/A'
          };
          return updated;
        });
        console.log('[EMAIL_EXTRACTION] Updated saved list leads');
      } else {
        setCustomerResearchResults((prev) => {
          if (!prev || !prev.businesses) {
            console.warn('[EMAIL_EXTRACTION] No customer research results to update');
            return prev;
          }

          const updatedBusinesses = [...prev.businesses];
          updatedBusinesses[index] = {
            ...updatedBusinesses[index],
            email: enrichedBusiness.email || 'N/A'
          };

          console.log('[EMAIL_EXTRACTION] Updated customer research results');
          return {
            ...prev,
            businesses: updatedBusinesses
          };
        });
      }
    } catch (error) {
      console.error('[EMAIL_EXTRACTION] Error extracting email for business:', error);
      console.error('[EMAIL_EXTRACTION] Error details:', {
        message: error.message,
        name: error.name,
        stack: error.stack
      });
      
      if (error.name === 'AbortError') {
        alert('Email extraction timed out. Please try again.');
      } else {
        alert(`Error: ${error.message}`);
      }
    } finally {
      setExtractingEmailRows((prev) => ({ ...prev, [index]: false }));
      console.log(`[EMAIL_EXTRACTION] Finished extraction for ${business.name}`);
    }
  };

  const handleCopyToClipboard = () => {
    if (!customerResearchResults || !customerResearchResults.businesses || customerResearchResults.businesses.length === 0) {
      alert('No data to copy.');
      return;
    }

    try {
      // Create tab-separated values format for easy pasting into Excel
      const headers = ['Business Name', 'Address', 'Phone', 'Website', 'Email'];
      const rows = customerResearchResults.businesses.map(business => [
        business.name || 'N/A',
        business.address || 'N/A',
        (business.phone || 'N/A').replace(/^\+/, ''),
        business.website || 'N/A',
        business.email || 'N/A'
      ]);

      // Create TSV (tab-separated values) content
      const tsvContent = [
        headers.join('\t'),
        ...rows.map(row => row.join('\t'))
      ].join('\n');

      // Copy to clipboard
      navigator.clipboard.writeText(tsvContent).then(() => {
        alert(`Successfully copied ${customerResearchResults.businesses.length} businesses to clipboard!`);
      }).catch(err => {
        console.error('Failed to copy:', err);
        alert('Failed to copy to clipboard');
      });
    } catch (error) {
      console.error('Error copying to clipboard:', error);
      alert('Failed to copy data to clipboard');
    }
  };

  const getCustomerResearchRows = () => {
    if (!customerResearchResults || !customerResearchResults.businesses || customerResearchResults.businesses.length === 0) {
      return [];
    }

    return customerResearchResults.businesses.map((business) => ({
      businessName: business.name || 'N/A',
      address: business.address || 'N/A',
      phone: business.phone || 'N/A',
      website: business.website || 'N/A',
      email: business.email || 'N/A',
      linkedin: business.linkedin || (business.linkedin_urls && business.linkedin_urls[0]) || 'N/A',
      summary: business.summary || business.description || 'N/A'
    }));
  };

  const downloadTextFile = (filename, content, mimeType) => {
    const blob = new Blob([content], { type: mimeType });
    const url = window.URL.createObjectURL(blob);
    const anchor = document.createElement('a');
    anchor.href = url;
    anchor.download = filename;
    document.body.appendChild(anchor);
    anchor.click();
    document.body.removeChild(anchor);
    window.URL.revokeObjectURL(url);
  };

  const escapeCsvValue = (value) => {
    const safeValue = String(value ?? '');
    if (safeValue.includes('"') || safeValue.includes(',') || safeValue.includes('\n')) {
      return `"${safeValue.replace(/"/g, '""')}"`;
    }
    return safeValue;
  };

  const buildCsvContent = (rows) => {
    const headers = ['Business Name', 'Address', 'Phone', 'Website', 'Email', 'LinkedIn', 'Summary'];
    const csvRows = rows.map((row) => [
      row.businessName,
      row.address,
      row.phone,
      row.website,
      row.email,
      row.linkedin,
      row.summary
    ].map(escapeCsvValue).join(','));

    return [headers.join(','), ...csvRows].join('\n');
  };

  const handleSaveList = async () => {
    const rows = getCustomerResearchRows();
    if (rows.length === 0) {
      const { plural } = getResearchEntityMeta();
      alert(`No ${plural} available to save.`);
      return;
    }

    // In demo mode, simulate saving
    if (isDemoMode) {
      if (!saveListName.trim() && saveListMode !== 'append') {
        alert('Please provide a name for the list.');
        return;
      }
      const newList = {
        id: `demo-${Date.now()}`,
        name: saveListName || 'New Demo List',
        query_used: customerResearchResults?.query || '',
        created_at: new Date().toISOString(),
        lead_count: rows.length,
        status: 'active'
      };
      setSavedLists(prev => [newList, ...prev]);
      setShowSaveListModal(false);
      setSaveListName('');
      alert('Demo mode: List saved to local view.');
      return;
    }

    const payloadLeads = (customerResearchResults?.businesses || []).map((business) => ({
      name: business.name || 'N/A',
      website: business.website || '',
      phone: business.phone || 'N/A',
      address: business.address || 'N/A',
      email: business.email || 'N/A',
      linkedin: business.linkedin || 'N/A',
      linkedin_urls: business.linkedin ? [business.linkedin] : [],
      social_links: business.social_links || {},
      summary: business.summary || business.description || 'N/A',
      raw_data: business
    }));

    setIsSavingList(true);
    try {
      const isAppendMode = saveListMode === 'append';
      if (isAppendMode) {
        if (!selectedAppendProjectId) {
          alert('Please select a list to append to.');
          setIsSavingList(false);
          return;
        }

        const response = await fetch(API_CONFIG.APPEND_PROJECT, {
          method: 'POST',
          headers: authJsonHeaders(),
          body: JSON.stringify({
            username: getCurrentUsername(),
            projectId: selectedAppendProjectId,
            businesses: payloadLeads
          })
        });

        const data = await response.json();
        if (data.success) {
          alert(data.message || 'Leads appended successfully!');
          setShowSaveListModal(false);
          setSaveListName('');
          setSaveListMode('create');
          setSelectedAppendProjectId('');
          await fetchSavedLists();
          if (activeSavedList && Number(activeSavedList.id) === Number(selectedAppendProjectId)) {
            await loadSavedListDetails(selectedAppendProjectId);
          }
        } else {
          alert('Error appending list: ' + (data.error || 'Unknown error'));
        }
      } else {
        if (!saveListName.trim()) {
          alert("Please provide a name for the list.");
          return;
        }

        const response = await fetch(API_CONFIG.SAVE_PROJECT, {
          method: 'POST',
          headers: authJsonHeaders(),
          body: JSON.stringify({
            username: getCurrentUsername(),
            name: saveListName,
            query: customerResearchResults?.query || '',
            query_used: customerResearchResults?.query || '',
            businesses: payloadLeads
          })
        });
        const data = await response.json();
        console.log('Save project response:', data);
        if (data.success) {
          const { title } = getResearchEntityMeta();
          alert(`${title} list saved successfully!`);
          console.log('Refreshing saved lists...');
          await fetchSavedLists();
          setShowSaveListModal(false);
          setSaveListName('');
          setSaveListMode('create');
          setSelectedAppendProjectId('');
        } else {
          console.error('Error saving list:', data.error);
          alert('Error saving list: ' + data.error);
        }
      }
    } catch (e) {
      console.error('Exception while saving:', e);
      alert('An error occurred while saving the list: ' + e.message);
    }
    setIsSavingList(false);
  };

  const handleScoreLeads = async () => {
    const text = (scoreQueryText || '').trim();
    if (!text) {
      alert('Please provide a short description of what you want to match for.');
      return;
    }

    const isSavedListContext = activeSavedList && activeSavedListLeads && activeSavedListLeads.length > 0;

    // Gather businesses from current view (saved list or live research)
    const sourceBusinesses = (activeSavedList && activeSavedListLeads && activeSavedListLeads.length) ? activeSavedListLeads : (customerResearchResults?.businesses || []);
    if (!sourceBusinesses || sourceBusinesses.length === 0) {
      const { plural } = getResearchEntityMeta();
      alert(`No ${plural} available to score.`);
      return;
    }

    // In demo mode, scores are already populated
    if (isDemoMode) {
      alert('Demo mode: Match scores are already visible in the demo data. In live mode, AI would re-score based on your criteria.');
      setShowScoreModal(false);
      return;
    }

    setIsScoring(true);
    try {
      const payload = sourceBusinesses.map(b => ({
        name: b.name || '',
        website: b.website || '',
        phone: b.phone || '',
        address: b.address || '',
        summary: b.summary || b.description || '',
        raw_data: b.raw_data || b
      }));

      const response = await fetch(API_CONFIG.SCORE_LEADS, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ username: getCurrentUsername(), requirement: text, businesses: payload })
      });
      let data;
      try {
        data = await response.json();
      } catch (parseErr) {
        console.error('[ScoreLeads] failed to parse JSON response', parseErr);
        alert('Scoring failed: invalid JSON response from server');
        setIsScoring(false);
        return;
      }
      
      if (data.success && Array.isArray(data.results)) {
        // Map results back into UI, attach scores/summaries, then sort descending by match_score
        const results = data.results;
        const resultsByIndex = new Map(results.map((item) => [Number(item.index), item]));

        // Pair results with the current sourceBusinesses using the stable backend index
        const paired = sourceBusinesses.map((b, i) => {
          const scored = resultsByIndex.get(i) || {};
          return {
            ...(b || {}),
            match_score: scored.match_score ?? null,
            short_summary: scored.short_summary ?? b.short_summary
          };
        });

        // Sort descending: highest match_score first (null/undefined treated as -1)
        const sorted = paired.slice().sort((a, b) => ( (b.match_score != null ? b.match_score : -1) - (a.match_score != null ? a.match_score : -1) ));

        // Apply to saved-list context or research view accordingly
        if (isSavedListContext) {
          setActiveSavedListLeads(sorted);
        }

        if (customerResearchResults) {
          const newCR = { ...customerResearchResults, businesses: sorted };
          setCustomerResearchResults(newCR);
          try { sessionStorage.setItem('customerResearchResults', JSON.stringify(newCR)); } catch (e) { /* ignore */ }
        }

        if (!isSavedListContext) {
          setShowCustomerResearchTable(true);
          setMinimizedCustomerResearch(false);
        }

        const { plural } = getResearchEntityMeta();
        alert(`Scoring complete — updated ${results.length} ${plural} (sorted by score)`);
        setShowScoreModal(false);
        setScoreQueryText('');
      } else {
        alert('Scoring failed: ' + (data.error || 'Unknown error'));
      }
    } catch (e) {
      console.error(e);
      alert('An error occurred while scoring leads: ' + e.message);
    }
    setIsScoring(false);
  };

  const handleDeleteSavedList = async (projectId) => {
    const confirmed = window.confirm('Delete this saved list permanently?');
    if (!confirmed) return;

    // In demo mode, just remove from local state
    if (isDemoMode) {
      setSavedLists(prev => prev.filter(l => l.id !== projectId));
      if (activeSavedList && activeSavedList.id === projectId) {
        setActiveSavedList(null);
        setActiveSavedListLeads([]);
      }
      alert('Demo mode: List removed from view.');
      return;
    }

    setDeletingListId(projectId);
    try {
      const response = await fetch(`${API_CONFIG.DELETE_SAVED_PROJECT}/${projectId}?username=${encodeURIComponent(getCurrentUsername())}`, {
        method: 'DELETE',
        headers: authOptionalHeaders(),
      });

      const data = await response.json();
      if (data.success) {
        alert(data.message || 'Saved list deleted successfully.');
        if (activeSavedList && Number(activeSavedList.id) === Number(projectId)) {
          setActiveSavedList(null);
          setActiveSavedListLeads([]);
        }
        await fetchSavedLists();
      } else {
        alert('Error deleting list: ' + (data.error || 'Unknown error'));
      }
    } catch (error) {
      console.error('Error deleting saved list:', error);
      alert('An error occurred while deleting the list: ' + error.message);
    } finally {
      setDeletingListId(null);
    }
  };

  const fetchSavedLists = async () => {
     setIsLoadingSavedLists(true);

     // In demo mode, use mock data
     if (isDemoMode) {
       const savedDemo = loadDemoState();
       if (savedDemo && savedDemo.savedLists) {
         setSavedLists(savedDemo.savedLists);
       } else {
         setSavedLists(DEMO_MOCK_DATA.savedLists);
       }
       setIsLoadingSavedLists(false);
       return;
     }

     try {
       const userIdentifier = getCurrentUsername();
       const res = await fetch(`${API_CONFIG.GET_SAVED_PROJECTS}?username=${encodeURIComponent(userIdentifier)}`, {
        headers: authOptionalHeaders(),
       });
       const data = await res.json();
       console.log('Fetched saved lists response:', data);
       if (data.success) {
          setSavedLists(data.projects || []);
       } else {
          console.error("Error fetching saved lists:", data.error);
          setSavedLists([]);
       }
     } catch (e) {
       console.error("Error fetching saved lists", e);
       setSavedLists([]);
     }
     setIsLoadingSavedLists(false);
  };

  useEffect(() => {
    if (showSaveListModal && savedLists.length === 0) {
      fetchSavedLists();
    }
  }, [showSaveListModal]);

  const loadSavedListDetails = async (projectId) => {
     // In demo mode, use mock leads
     if (isDemoMode) {
       const demoList = DEMO_MOCK_DATA.savedLists.find(l => l.id === projectId);
       if (demoList) {
         // Use the main demo businesses as the leads for any demo list
         const leads = DEMO_MOCK_DATA.results.businesses;

         setCustomerResearchResults({
           query: demoList.query_used || demoList.name,
           location: 'North America',
           industry: 'Technology',
           totalResults: leads.length,
           researchType: 'Customer Research',
           businesses: leads
         });

         setActiveSavedList(demoList);
         setActiveSavedListLeads(leads);
         setShowSavedListsView(false);
         setShowCustomerResearchTable(true);
         setMinimizedCustomerResearch(false);
       }
       return;
     }

     try {
        const userIdentifier = getCurrentUsername();
        const res = await fetch(`${API_CONFIG.GET_SAVED_PROJECT_LEADS}/${projectId}/leads?username=${encodeURIComponent(userIdentifier)}`, {
          headers: authOptionalHeaders(),
        });
        const data = await res.json();
        if (data.success) {
           const leads = data.leads.map(l => ({
            name: l.name,
            website: l.website,
            phone: l.phone,
            address: l.address,
            email: Array.isArray(l.emails) ? (l.emails[0] || 'N/A') : (l.emails || 'N/A'),
            emails: l.emails,
            linkedin: Array.isArray(l.linkedin_urls) ? (l.linkedin_urls[0] || '') : (l.linkedin_urls || ''),
            linkedin_urls: l.linkedin_urls,
            social_links: l.social_links,
            summary: l.summary || l.description || 'N/A',
            has_extracted: l.has_extracted
           }));

           // Set customer research results to display in regular leads view
           setCustomerResearchResults({
             query: data.project.query_used || data.project.name,
             researchType: data.project.researchType || 'Customer Research',
             businesses: leads
           });
           try {
             sessionStorage.setItem('customerResearchResults', JSON.stringify({
               query: data.project.query_used || data.project.name,
               researchType: data.project.researchType || 'Customer Research',
               businesses: leads
             }));
           } catch (e) { /* ignore */ }

           // Store saved list info for context
           setActiveSavedList(data.project);
           setActiveSavedListLeads(leads);

           // Switch to leads view (not saved lists view)
           setShowSavedListsView(false);
           setShowCustomerResearchTable(true);
           setMinimizedCustomerResearch(false);
        }
     } catch(e) {
        console.error(e);
     }
  };

  const handleExport = async (format) => {
    const rows = getCustomerResearchRows();
    if (rows.length === 0) {
      alert('No data available to export.');
      return;
    }

    const fileBaseName = `market_research_${new Date().toISOString().slice(0, 10)}`;
    const csvContent = buildCsvContent(rows);

    try {
      if (format === 'excel') {
        const worksheet = XLSX.utils.json_to_sheet(rows);
        const workbook = XLSX.utils.book_new();
        XLSX.utils.book_append_sheet(workbook, worksheet, 'Market Research');
        XLSX.writeFile(workbook, `${fileBaseName}.xlsx`);
      }

      if (format === 'csv') {
        downloadTextFile(`${fileBaseName}.csv`, csvContent, 'text/csv;charset=utf-8;');
      }

      if (format === 'json') {
        downloadTextFile(`${fileBaseName}.json`, JSON.stringify(rows, null, 2), 'application/json;charset=utf-8;');
      }

      if (format === 'pdf') {
        const doc = new jsPDF({ orientation: 'landscape', unit: 'pt', format: 'a4' });
        doc.setFontSize(12);
        doc.text('Market Research Export', 40, 36);

        autoTable(doc, {
          startY: 50,
          head: [['Business Name', 'Address', 'Phone', 'Website', 'Email', 'LinkedIn', 'Summary']],
          body: rows.map((row) => [
            row.businessName,
            row.address,
            row.phone,
            row.website,
            row.email,
            row.linkedin,
            row.summary
          ]),
          styles: { fontSize: 8, cellPadding: 4 },
          headStyles: { fillColor: [30, 58, 95] }
        });

        doc.save(`${fileBaseName}.pdf`);
      }

      if (format === 'sheets') {
        const headers = ['Business Name', 'Address', 'Phone', 'Website', 'Email'];
        const matrixRows = rows.map((row) => [
          row.businessName,
          row.address,
          row.phone,
          row.website,
          row.email
        ]);

        const toSheetsSafeValue = (value) => {
          const strValue = String(value ?? '');
          return /^[=+\-@]/.test(strValue) ? `'${strValue}` : strValue;
        };

        const tsvContent = [
          headers.join('\t'),
          ...matrixRows.map((row) => row.map(toSheetsSafeValue).join('\t'))
        ].join('\n');

        await navigator.clipboard.writeText(tsvContent);
        window.open('https://docs.google.com/spreadsheets/create', '_blank', 'noopener,noreferrer');
        alert('Google Sheets opened. Data is copied to clipboard, paste with Ctrl+V.');
      }
    } catch (error) {
      console.error('Export failed:', error);
      alert('Export failed. Please try again.');
    } finally {
      setShowExportModal(false);
    }
  };

  const closePopup = () => {
    setShowPopup(false);
  };

  const handleGoogleBusinessInputChange = (e) => {
    const { name, value } = e.target;
    setGoogleBusinessForm(prev => ({
      ...prev,
      [name]: value
    }));
  };

  const handleGoogleBusinessConnect = async () => {
    // Check if server has credentials configured
    if (!googleBusinessForm.hasCredentials && !googleBusinessForm.clientId) {
      alert('Google credentials not configured. Contact administrator.');
      return;
    }

    try {
      // Server uses its own credentials from .env - don't send secrets from frontend
      const response = await fetch(API_CONFIG.CONNECT_GOOGLE_BUSINESS, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          clientId: googleBusinessForm.clientId,
          redirectUri: googleBusinessForm.redirectUri
        }),
      });

      const data = await response.json();
      
      if (response.ok && data.authUrl) {
        // Open Google authorization URL in new window
        setShowIntegrationModal(false);
        window.open(data.authUrl, 'google_auth', 'width=500,height=600');
        
        // After user authorizes, the app will redirect to localhost:3000?google_connected=true
        // We'll handle that with a URL parameter check in useEffect
      } else {
        alert(data.error || 'Failed to generate authorization URL');
      }
    } catch (error) {
      console.error('Error connecting Google Business:', error);
      alert('Error connecting to Google Business');
    }
  };

  const fetchGoogleBusinessData = async () => {
    try {
      const response = await fetch(API_CONFIG.GET_GOOGLE_BUSINESS_DATA, {
        method: 'GET',
        headers: {
          'Content-Type': 'application/json',
        },
      });

      if (response.ok) {
        const data = await response.json();
        return data;
      } else {
        console.log('Google Business not connected or data unavailable');
        return null;
      }
    } catch (error) {
      console.error('Error fetching Google Business data:', error);
        return null;
      }
    };

    const handleGeneratePersonalizedEmail = async (business, index) => {
    setIsGeneratingEmail(prev => ({ ...prev, [index]: true }));
    try {
      if (responseFormat === 'Supplier Research') {
        prepareSupplierEmailDraft(business);
        setShowEmailModal(true);
        return;
      }

      // In demo mode, use sample email content
      if (isDemoMode) {
        setEmailSubject(`Partnership Opportunity - ${business.name || 'Your Company'}`);
        setEmailBody(`Hi ${business.name ? business.name.split(' ')[0] : 'there'},

I came across ${business.name || 'your company'} and was impressed by your work in ${business.summary || 'your industry'}.

We at Enable Agents specialize in AI-powered business automation, and I believe there's potential for a valuable partnership.

Would you be open to a brief call this week to explore how we might work together?

Best regards,
${getCurrentUsername() || 'Your Name'}`);
        setCampaignName('Personalized: ' + (business.name || 'Company'));
        setSelectedLead(business);
        setShowEmailModal(true);
        setIsGeneratingEmail(prev => ({ ...prev, [index]: false }));
        return;
      }

      const response = await fetch(API_CONFIG.GENERATE_EMAIL, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          business: business,
          sender_name: getCurrentUsername()
        })
      });
      const data = await response.json();
      if (response.ok && data.subject && data.body) {
        setEmailSubject(data.subject);
        setEmailBody(data.body);
        setCampaignName('Personalized: ' + (business.name || 'Company'));
        setSelectedLead(business);
        setShowEmailModal(true);
      } else {
        alert('Failed to generate personalized email.');
      }
    } catch (err) {
      console.error(err);
      alert('Error generating email.');
    } finally {
      setIsGeneratingEmail(prev => ({ ...prev, [index]: false }));
    }
  };

  const getSenderCompanyData = useCallback(() => {
    return {
      companyName: localStorage.getItem('companyName') || localStorage.getItem('company') || '',
      companyWebsite: localStorage.getItem('companyWebsite') || '',
      industry: localStorage.getItem('companyIndustry') || industries || '',
      location: localStorage.getItem('companyLocation') || countries || '',
      contactName: getCurrentUsername() || localStorage.getItem('firstName') || '',
      phoneNumber: localStorage.getItem('phoneNumber') || '',
      emailAddress: getCurrentUserEmail() || localStorage.getItem('userEmail') || ''
    };
  }, [countries, industries]);

  const prepareSupplierEmailDraft = (business = null) => {
    const vars = getSenderCompanyData();
    vars.productCategory = industries || overview || '';
    vars.vendorName = business?.name || business?.businessName || 'Sir / Madam';

    setSelectedLead(business);
    setEmailSubject(`Supplier Inquiry${vars.productCategory ? ' - ' + vars.productCategory : ''}`);
    setEmailBody(fillTemplate(SUPPLIER_TEMPLATE, vars));
  };

  const handleSendEmails = async () => {
    if (!useAiBulk && (!emailSubject || !emailBody)) {
      alert("Subject and Body required unless using AI Personalization");
      return;
    }

    // In demo mode, simulate sending
    if (isDemoMode) {
      alert('Demo mode: Email sending simulated. In live mode, emails would be sent via your connected account.');
      setShowEmailModal(false);
      setEmailSubject('');
      setEmailBody('');
      setSelectedLead(null);
      return;
    }

    const registeredEmail = getCurrentUserEmail();
    if (!registeredEmail) {
      alert('Registered email not found. Please log in again so the campaign can be sent from your account.');
      return;
    }

    let validEmails = [];
    if (selectedLead) {
      validEmails = selectedLead.email && selectedLead.email !== 'N/A' && selectedLead.email.includes('@')
        ? [{ ...selectedLead }]
        : [];
    } else {
      validEmails = (customerResearchResults?.businesses || [])
        .filter(b => b)
        .map(b => ({
          ...b,
        }));
    }
    
    if (validEmails.length === 0) {
      alert("No valid emails found to send to");
      return;
    }

    setIsSendingEmails(true);
    try {
      const response = await fetch(API_CONFIG.SEND_BULK_EMAILS, {  
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          userId: localStorage.getItem("firstName") || "",
          userEmail: registeredEmail,
          campaignName: campaignName || (selectedLead ? 'Test Outreach - 1-on-1' : 'Test Outreach - Bulk'),
          subject: emailSubject,
          body: emailBody,
          businesses: validEmails,
          use_ai_personalization: useAiBulk
        })
      });
      const data = await response.json();
      if (response.ok && data.success) {
        alert('Successfully sent ' + validEmails.length + ' emails!');
        setShowEmailModal(false);
        setEmailSubject('');
        setEmailBody('');
        setSelectedLead(null);
      } else {
        alert('Error : ' + data.error);
      }
    } catch (e) {
      alert('Error sending emails: ' + e.message);
    } finally {
      setIsSendingEmails(false);
    }
  };

    return (
    <div className="requirements-page">
      <Header />
      <div className="requirements-container">
        
        <div className="requirements-header-bar">
          <div className="header-bar-top">
            <div className="overview-title">
              <span>Market Research Agent</span>
            </div>
            <div className="integration-badge">
              <span className="dot"></span> 3RD PARTY INTEGRATION READY
            </div>
          </div>
          <div className="header-bar-inputs">
            <div className="input-block flex-grow">
              <label>PROJECT CONTEXT & DESCRIPTION</label>
              <input
                type="text"
                placeholder="What is the product or service you need research on?"
                value={overview}
                onChange={(e) => setOverview(e.target.value)}
              />
            </div>
            <div className="input-block">
              <label>INDUSTRY</label>
              <input
                type="text"
                placeholder="e.g. Fintech"
                value={industries}
                onChange={(e) => setIndustries(e.target.value)}
              />
            </div>
            <div className="input-block">
              <label>REGION</label>
              <input
                type="text"
                placeholder="e.g. North Ame"
                value={countries}
                onChange={(e) => setCountries(e.target.value)}
              />
            </div>
            <div className="input-block">
              <label>FORMAT</label>
              <select
                value={responseFormat}
                onChange={(e) => setResponseFormat(e.target.value)}
              >
                 <option value="">Select format...</option>
                 <option value="Detailed PRD">Detailed PRD</option>
                 <option value="Supplier Research">Supplier Research</option>
                 <option value="Customer Research">Customer Research</option>
                 <option value="Industry Use Cases">Industry Use Cases</option>
                 <option value="Product Requirements">Product Requirements</option>
                 <option value="Competitive Research">Competitive Research</option>
              </select>
            </div>
            <div className="input-block button-block">
              <input type="file" id="file-input" onChange={handleFileUpload} style={{ display: 'none' }} />
              <button className="upload-btn" onClick={() => document.getElementById('file-input').click()}>
                <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" style={{marginRight: '6px'}}>
                  <path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4"></path>
                  <polyline points="17 8 12 3 7 8"></polyline>
                  <line x1="12" y1="3" x2="12" y2="15"></line>
                </svg>
                {uploadedFile ? uploadedFile.name : 'Upload Ref'}
              </button>
            </div>
            <div className="input-block button-block">
              <button className="generate-req-btn" onClick={handleGenerate} disabled={isLoadingResearch}>
                {isLoadingResearch ? (
                  <span style={{display: 'flex', alignItems: 'center', justifyContent: 'center', gap: '8px'}}>
                    <span className="spinner"></span> Generating...
                  </span>
                ) : 'Get Research Insights'}
              </button>
            </div>
          </div>
        </div>

        <div className="main-workspace-area">

          <div className="tabs-container">
            <button className={`workspace-tab ${!showSavedListsView ? 'active-tab' : ''}`} onClick={() => { setShowSavedListsView(false); setActiveSavedList(null); setActiveSavedListLeads([]); }}>{responseFormat === 'Supplier Research' ? 'Vendors' : 'Leads'}</button>
            <button className={`workspace-tab ${showSavedListsView ? 'active-tab' : ''}`} onClick={() => { setShowSavedListsView(true); fetchSavedLists(); }}>{responseFormat === 'Supplier Research' ? 'Saved Vendors' : 'Saved Leads'}</button>
            <a href="/market-research/campaigns" className="workspace-link">
              Campaign Dashboard →
            </a>
          </div>

          <div className="workspace-content-box" style={{ display: 'flex', flexDirection: 'column', height: '100%', minHeight: '600px' }}>
             {showSavedListsView ? (
               <div className="saved-lists-container" style={{ padding: '20px', display: 'flex', flexDirection: 'column', flex: 1, height: '100%', overflow: 'hidden' }}>
                 {activeSavedList && (
                    <button 
                      onClick={() => { setActiveSavedList(null); setActiveSavedListLeads([]); fetchSavedLists(); }} 
                       style={{ marginBottom: '20px', background: 'none', border: '1px solid #1E3A5F', padding: '5px 15px', borderRadius: '5px', cursor: 'pointer', color: '#1E3A5F' }}
                    >
                       ← Back to All Lists
                    </button>
                 )}
                 {activeSavedList && (
                   <h2 style={{ fontSize: '24px', fontWeight: 'bold', margin: '0 0 20px 0', color: '#1E3A5F' }}>
                     {activeSavedList.name}
                   </h2>
                 )}
                 {isLoadingSavedLists ? (
                    <div style={{ textAlign: 'center', padding: '40px' }}><span className="spinner"></span> Loading lists...</div>
                 ) : !activeSavedList ? (
                    savedLists.length === 0 ? (
                      <div style={{ textAlign: 'center', padding: '40px', color: '#666' }}>No saved lists found.</div>
                    ) : (
                      <div className="saved-lists-table-scroll">
                        <table className="businesses-table" style={{ width: '100%', borderCollapse: 'collapse' }}>
                          <thead>
                            <tr>
                              <th style={{ textAlign: 'left' }}>List Name</th>
                              <th style={{ textAlign: 'left' }}>Query</th>
                              <th style={{ textAlign: 'center' }}>Results</th>
                              <th style={{ textAlign: 'center' }}>Created</th>
                              <th style={{ textAlign: 'center' }}>Actions</th>
                            </tr>
                          </thead>
                          <tbody>
                            {savedLists.map((list) => (
                              <tr key={list.id}>
                                <td>{list.name}</td>
                                <td>{list.query_used || 'N/A'}</td>
                                <td style={{ textAlign: 'center' }}>{list.lead_count}</td>
                                <td style={{ textAlign: 'center' }}>{list.created_at ? new Date(list.created_at).toLocaleDateString() : 'N/A'}</td>
                                <td>
                                  <div style={{ display: 'flex', gap: '10px', justifyContent: 'center', flexWrap: 'wrap' }}>
                                    <button
                                      onClick={() => loadSavedListDetails(list.id)}
                                      style={{ padding: '8px 16px', border: 'none', borderRadius: '6px', background: '#284A7A', color: '#fff', cursor: 'pointer', fontWeight: 700 }}
                                    >
                                      Open List
                                    </button>
                                    <button
                                      onClick={() => handleDeleteSavedList(list.id)}
                                      disabled={deletingListId === list.id}
                                      style={{ padding: '8px 16px', border: 'none', borderRadius: '6px', background: '#b42318', color: '#fff', cursor: 'pointer', fontWeight: 700, opacity: deletingListId === list.id ? 0.7 : 1 }}
                                    >
                                      {deletingListId === list.id ? 'Deleting...' : 'Delete'}
                                    </button>
                                  </div>
                                </td>
                              </tr>
                            ))}
                          </tbody>
                        </table>
                      </div>
                    )
                 ) : (
                    <div className="saved-list-leads-scroll">
                      <table className="businesses-table" style={{ width: '100%', borderCollapse: 'collapse' }}>
                        <thead>
                          <tr>
                            <th>Business Name</th>
                            <th>Address</th>
                            <th>Phone</th>
                            <th>Website</th>
                            <th>Email</th>
                            <th>LinkedIn</th>
                            <th>Send Email</th>
                            <th style={{ width: '120px' }}>Match</th>
                            <th>Summary</th>
                          </tr>
                        </thead>
                        <tbody>
                          {activeSavedListLeads?.map((business, index) => (
                            <tr key={index}>
                              <td>{business.name || 'N/A'}</td>
                              <td>{business.address || 'N/A'}</td>
                              <td>{business.phone || 'N/A'}</td>
                              <td>{business.website ? <a href={business.website} target="_blank" rel="noopener noreferrer">Visit</a> : 'N/A'}</td>
                              <td>{business.email && business.email !== 'N/A' ? <span>{business.email}</span> : <span style={{ color: '#999' }}>N/A</span>}</td>
                              <td>{business.linkedin ? (business.linkedin !== 'N/A' ? <a href={business.linkedin} target="_blank" rel="noopener noreferrer" style={{ color: '#0d6efd', textDecoration: 'none', fontWeight: 'bold' }}>View Profile</a> : <span style={{ color: '#999', fontStyle: 'italic', fontSize: '0.9em' }}>Not Found</span>) : <span style={{ color: '#999' }}>N/A</span>}</td>
                              <td>{business.email && business.email !== 'N/A' ? <button onClick={() => handleGeneratePersonalizedEmail(business, index)} disabled={!!isGeneratingEmail[index]} style={{ backgroundColor: '#3b82f6', color: 'white', border: 'none', padding: '4px 8px', borderRadius: '4px', cursor: 'pointer', fontSize: '11px', fontWeight: 'bold' }}>{isGeneratingEmail[index] ? 'Drafting...' : 'Draft Email'}</button> : <span style={{ color: '#999', fontSize: '0.9em' }}>-</span>}</td>
                              <td style={{ textAlign: 'center', verticalAlign: 'middle' }}>
                                {business.match_score != null ? (
                                  <div style={{ display: 'flex', flexDirection: 'column', alignItems: 'center' }}>
                                    <div style={{ fontWeight: 700, color: business.match_score >= 70 ? '#0f766e' : (business.match_score >= 40 ? '#f59e0b' : '#9ca3af') }}>{business.match_score}%</div>
                                  </div>
                                ) : <span style={{ color: '#999' }}>-</span>}
                              </td>
                              <td style={{ maxWidth: '320px', whiteSpace: 'pre-line', lineHeight: '1.4' }} title={business.short_summary || business.summary || business.description || 'N/A'}>{business.short_summary || business.summary || business.description || 'N/A'}</td>
                            </tr>
                          ))}
                        </tbody>
                      </table>
                    </div>
                 )}
               </div>
            ) : (
            <div className="ai-assisted" style={{ background: 'transparent', boxShadow: 'none' }}>
              {isLoadingResearch ? (
                  <div style={{ display: 'flex', flexDirection: 'column', alignItems: 'center', justifyContent: 'center', height: '100%', minHeight: '400px' }}>
                    <div className="loader" style={{ border: '4px solid #f3f3f3', borderTop: '4px solid #ff725e', borderRadius: '50%', width: '40px', height: '40px', animation: 'spin 1s linear infinite' }}></div>
                    <p style={{ marginTop: '20px', color: '#666', fontSize: '18px' }}>Scraping and analyzing {getResearchEntityMeta().plural}... please wait...</p>
                    <style>{"@keyframes spin { 0% { transform: rotate(0deg); } 100% { transform: rotate(360deg); } }"}</style>
                  </div>
              ) : (!aiRequirements && !customerResearchResults) ? (
                <div style={{ display: 'flex', flexDirection: 'column', alignItems: 'center', justifyContent: 'center', height: '100%', minHeight: '200px' }}>
                  <div style={{ background: '#EAE1D9', borderRadius: '12px', width: '64px', height: '64px', display: 'flex', alignItems: 'center', justifyContent: 'center', marginBottom: '12px' }}>
                    <svg width="32" height="32" viewBox="0 0 24 24" fill="none" stroke="#8E9BAb" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
                      <path d="M12 2l3.09 6.26L22 9.27l-5 4.87 1.18 6.88L12 17.77l-6.18 3.25L7 14.14 2 9.27l6.91-1.01L12 2z"></path>
                    </svg>
                  </div>
                  <h2 style={{ color: '#0D2644', fontSize: '1.5rem', marginBottom: '12px' }}>Awaiting Configuration</h2>
                  <p style={{ color: '#6C7F99', textAlign: 'center', maxWidth: '400px', fontSize: '0.95rem', lineHeight: '1.5', marginBottom: '16px' }}>
                    Refine the requirements in the bar above to generate structured architectural specifications. Our AI will analyze your context, industry, and region to produce a precise specification.
                  </p>
                  <div style={{ display: 'flex', gap: '16px' }}>
                    <div style={{ background: 'white', padding: '8px 16px', borderRadius: '4px', fontSize: '0.75rem', fontWeight: 'bold', color: '#1E3A5F', boxShadow: '0 2px 4px rgba(0,0,0,0.05)' }}>READY TO ANALYZE</div>
                    <div style={{ background: 'white', padding: '8px 16px', borderRadius: '4px', fontSize: '0.75rem', fontWeight: 'bold', color: '#1E3A5F', boxShadow: '0 2px 4px rgba(0,0,0,0.05)' }}>SECURE END-TO-END</div>
                    <div style={{ background: 'white', padding: '8px 16px', borderRadius: '4px', fontSize: '0.75rem', fontWeight: 'bold', color: '#1E3A5F', boxShadow: '0 2px 4px rgba(0,0,0,0.05)' }}>ENTERPRISE LLM</div>
                  </div>
                </div>
              ) : (
                <>
                  {/* Show Customer Research Results */}
                  {customerResearchResults && (
                      <div className="minimized-customer-research-box">
                        <div className="research-summary-row minimized">
                          <div className="summary-badges">
                            {activeSavedList && (
                              <button 
                                onClick={() => { 
                                  setActiveSavedList(null); 
                                  setActiveSavedListLeads([]); 
                                  setShowSavedListsView(true); 
                                  setShowCustomerResearchTable(false);
                                }} 
                                style={{ marginRight: '12px', padding: '8px 16px', background: 'none', border: '1px solid #1E3A5F', borderRadius: '5px', cursor: 'pointer', color: '#1E3A5F', fontWeight: 600 }}
                              >
                                ← Back to Saved Lists
                              </button>
                            )}
                            <div className="summary-badge">
                              <span className="badge-label">Search</span>
                              <span className="badge-value">{customerResearchResults.query}</span>
                            </div>
                            <div className="summary-badge">
                              <span className="badge-label">Location</span>
                              <span className="badge-value">{customerResearchResults.location}</span>
                            </div>
                            <div className="summary-badge">
                              <span className="badge-label">Industry</span>
                              <span className="badge-value">{customerResearchResults.industry}</span>
                            </div>
                            <div className="summary-badge results-badge">
                              <span className="badge-label">Results</span>
                              <span className="badge-value">{customerResearchResults.totalResults}</span>
                            </div>
                          </div>
                          
                          <div style={{ marginLeft: 'auto', display: 'flex', gap: '8px', alignItems: 'center' }}>
                              <div className="summary-badge emails-badge">
                                <span className="badge-label">Extracted</span>
                                <span className="badge-value">
                                  {customerResearchResults.businesses ? customerResearchResults.businesses.filter(b => b.email && b.email !== 'N/A').length : 0}/100
                                </span>
                              </div>
                              <button 
                                className="get-emails-button compact"
                                onClick={handleGetEmails}
                                disabled={isLoadingEmails}
                                style={{ margin: 0, padding: '8px 16px' }}
                              >
                                {isLoadingEmails ? (
                                  <>
                                    <span className="spinner" style={{ marginRight: '6px' }}></span>
                                    Extracting...
                                  </>
                                ) : 'Extract Emails'}
                              </button>
                              <button 
                                className="action-icon-button"
                                onClick={handleCopyToClipboard}
                                title="Copy to Clipboard"
                                aria-label="Copy to Clipboard"
                              >
                                <img src="/assets/icons/copy.png" alt="Copy to Clipboard" />
                              </button>
                              <button 
                                className="action-icon-button"
                                onClick={() => setShowExportModal(true)}
                                title="Export Data"
                                aria-label="Export Data"
                              >
                                <img src="/assets/icons/import-export.png" alt="Export Data" />
                              </button>
                              <button 
                                className="action-icon-button"
                                onClick={() => {
                                  setSaveListMode('create');
                                  setSelectedAppendProjectId('');
                                  setShowSaveListModal(true);
                                }}
                                title={responseFormat === 'Supplier Research' ? 'Save Vendors List' : 'Save Leads List'}
                                aria-label={responseFormat === 'Supplier Research' ? 'Save Vendors List' : 'Save Leads List'}
                              >
                                <img src="/assets/icons/document.png" alt={responseFormat === 'Supplier Research' ? 'Save Vendors' : 'Save Leads'} />
                              </button>
                              <button 
                                className="action-icon-button"
                                onClick={() => { setScoreQueryText(''); setShowScoreModal(true); }}
                                title={responseFormat === 'Supplier Research' ? 'Score Vendors' : 'Score Leads'}
                                aria-label={responseFormat === 'Supplier Research' ? 'Score Vendors' : 'Score Leads'}
                              >
                                <img src="/assets/icons/bar-chart.png" alt={responseFormat === 'Supplier Research' ? 'Score Vendors' : 'Score Leads'} />
                              </button>
                              <button 
                                className="action-icon-button"
                                onClick={() => { prepareSupplierEmailDraft(null); setShowEmailModal(true); }}
                                title="Send Emails"
                                aria-label="Send Emails"
                              >
                                <img src="/assets/icons/mail.png" alt="Send Emails" />
                              </button>
                              <button 
                                className="action-icon-button"
                                onClick={() => { setShowCustomerResearchTable(true); setMinimizedCustomerResearch(false); }}
                                title="Maximize"
                                aria-label="Maximize Table"
                              >
                                <img src="/assets/icons/maximize.png" alt="Maximize" />
                              </button>
                          </div>
                        </div>
                        <div className="minimized-content-scroll">
                          {customerResearchResults.businesses && customerResearchResults.businesses.length > 0 ? (
                            <div className="table-wrapper minimized-table-wrapper">
                              <table className="businesses-table">
                                <thead>
                                  <tr>
                                      <th>Business Name</th>
                                      <th>Address</th>
                                      <th>Phone</th>
                                      <th>Website</th>
                                      <th>Email</th>
                                      <th>LinkedIn</th>
                                      <th>Send Email</th>
                                      <th style={{ width: '120px' }}>Match</th>
                                      <th>Summary</th>
                                    </tr>
                                </thead>
                                <tbody>
                                  {customerResearchResults.businesses.map((business, index) => (
                                    <tr key={index}>
                                      <td>{business.name || 'N/A'}</td>
                                      <td>{business.address || 'N/A'}</td>
                                      <td>{business.phone || 'N/A'}</td>
                                      <td>
                                        {business.website ? (
                                          <a href={business.website} target="_blank" rel="noopener noreferrer">
                                            Visit
                                          </a>
                                        ) : (
                                          'N/A'
                                        )}
                                      </td>
                                      <td>
                                        {business.email && business.email !== 'N/A' ? ( 
                                          <span>{business.email}</span>
                                        ) : (
                                          <button
                                            className="extract-email-button"
                                            onClick={() => handleExtractEmailForBusiness(business, index)}
                                            disabled={!!extractingEmailRows[index]}     
                                          >
                                            {extractingEmailRows[index] ? 'Extracting...' : 'Extract Email'}
                                          </button>
                                        )}
                                      </td>
                                      <td>
                                        {business.linkedin ? (
                                          business.linkedin !== 'N/A' ? (
                                            <a href={business.linkedin} target="_blank" rel="noopener noreferrer" className="table-link">
                                              View Profile
                                            </a>
                                          ) : (
                                            <span className="table-na">Not Found</span>
                                          )
                                        ) : (
                                          <button
                                            className="table-btn-secondary"
                                            onClick={() => handleExtractLinkedInForBusiness(business, index)}
                                            disabled={!!extractingLinkedInRows[index]}
                                          >
                                            {extractingLinkedInRows[index] ? 'Extracting...' : 'Extract LinkedIn'}
                                          </button>
                                        )}
                                      </td>
                                      <td>
                                        {business.email && business.email !== 'N/A' ? (
                                          <button
                                            className="table-btn-primary"
                                            onClick={() => handleGeneratePersonalizedEmail(business, index)}
                                            disabled={!!isGeneratingEmail[index]}
                                          >
                                            {isGeneratingEmail[index] ? 'Drafting...' : 'Draft Email'}
                                          </button>
                                        ) : (
                                          <span className="table-muted">-</span>
                                        )}
                                      </td>
                                      <td className="match-cell">
                                        {business.match_score != null ? (
                                          <span className={`match-score ${business.match_score >= 70 ? 'match-score--high' : business.match_score >= 40 ? 'match-score--medium' : 'match-score--low'}`}>
                                            {business.match_score}%
                                          </span>
                                        ) : <span className="table-muted">-</span>}
                                      </td>
                                      <td className="summary-cell" title={business.short_summary || business.summary || business.description || 'N/A'}>
                                        {business.short_summary || business.summary || business.description || 'N/A'}
                                      </td>
                                    </tr>
                                  ))}
                                </tbody>
                              </table>
                            </div>
                          ) : (
                            <div className="no-results">No businesses found matching your search criteria.</div>
                          )}
                        </div>
                      </div>
                  )}

                  {/* Show AI Requirements */}
                  {aiRequirements.length > 0 && (
                    <div className="requirements-list-section">
                      <h3>Generated Requirements</h3>
                      <ul>
                        {aiRequirements.map((requirement, index) => (
                          <li key={index}>{requirement}</li>
                        ))}
                      </ul>
                    </div>
                  )}

                  {aiRequirements.length === 0 && !customerResearchResults && (
                    <div className="empty-state-container">
                      <div className="empty-state-icon">🔍</div>
                      <h3 className="empty-state-title">Ready to Research</h3>
                      <p className="empty-state-description">
                        Describe your product or service, select an industry and region, then click "Get Research Insights" to discover leads, competitors, and market opportunities.
                      </p>
                      <div className="empty-state-steps">
                        <div className="empty-state-step">
                          <span className="step-number">1</span>
                          <span className="step-text">Enter project context</span>
                        </div>
                        <div className="empty-state-step">
                          <span className="step-number">2</span>
                          <span className="step-text">Select industry & region</span>
                        </div>
                        <div className="empty-state-step">
                          <span className="step-number">3</span>
                          <span className="step-text">Get AI-powered insights</span>
                        </div>
                      </div>
                      {isDemoMode && (
                        <button
                          className="demo-example-btn"
                          onClick={() => {
                            // Load demo data and persist to sessionStorage
                            setOverview(DEMO_MOCK_DATA.overview);
                            setIndustries(DEMO_MOCK_DATA.industries);
                            setCountries(DEMO_MOCK_DATA.countries);
                            setResponseFormat(DEMO_MOCK_DATA.responseFormat);
                            setCustomerResearchResults(DEMO_MOCK_DATA.results);
                            setShowCustomerResearchTable(true);
                            // Persist demo state
                            saveDemoState({
                              overview: DEMO_MOCK_DATA.overview,
                              industries: DEMO_MOCK_DATA.industries,
                              countries: DEMO_MOCK_DATA.countries,
                              responseFormat: DEMO_MOCK_DATA.responseFormat,
                              results: DEMO_MOCK_DATA.results,
                              savedLists: DEMO_MOCK_DATA.savedLists,
                              isDemo: true
                            });
                          }}
                        >
                          ✨ Try Example (Demo Mode)
                        </button>
                      )}
                    </div>
                  )}
                </>
              )}
            </div>
          )}
          </div>

          {/* Google Business Integration Modal */}
          {showIntegrationModal && (
            <div className="popup-overlay">
              <div className="popup-content integration-modal">
                <h3>{googleBusinessConnected ? 'Reconnect Google Business Account' : 'Connect Google Business Account'}</h3>
                <div className="integration-form">
                  <div className="form-group">
                    <label>Client ID</label>
                    <input
                      type="text"
                      name="clientId"
                      value={googleBusinessForm.clientId}
                      onChange={handleGoogleBusinessInputChange}
                      placeholder="Enter Client ID"
                    />
                  </div>
                  {/* Client Secret is configured server-side in .env for security */}
                  <div className="form-group">
                    <label>Redirect URI</label>
                    <input
                      type="text"
                      name="redirectUri"
                      value={googleBusinessForm.redirectUri}
                      onChange={handleGoogleBusinessInputChange}
                      placeholder="Enter Redirect URI"
                    />
                  </div>
                </div>
                <div className="modal-buttons">
                  <button className="connect-submit-button" onClick={handleGoogleBusinessConnect}>
                    Connect
                  </button>
                  <button className="close-popup-button" onClick={() => setShowIntegrationModal(false)}>
                    Cancel
                  </button>
                </div>
              </div>
            </div>
          )}

          {/* Popup for Export Options */}
          {showPopup && (
            <div className="popup-overlay">
              <div className="popup-content">
                <h3>Export Options</h3>
                <div className="export-icons">
                  <img src="/assets/icons/gmail.png" alt="Gmail" title="Gmail" />
                  <img src="/assets/icons/word.png" alt="Word" title="Word" />
                  <img src="/assets/icons/pdf.png" alt="PDF" title="PDF" />
                  <img src="/assets/icons/canva.png" alt="Canva" title="Canva" />
                  <img src="/assets/icons/figma.png" alt="Figma" title="Figma" />
                  <img src="/assets/icons/powerpoint.png" alt="PowerPoint" title="PowerPoint" />
                </div>
                <button className="close-popup-button" onClick={closePopup}>
                  Close
                </button>
              </div>
            </div>
          )}

          {/* Customer Research Results Table */}

          {showCustomerResearchTable && customerResearchResults && !minimizedCustomerResearch && (
              <div className="popup-overlay">
                <div className="popup-content customer-research-table" style={{ position: 'relative' }}>
                  <button 
                    onClick={() => { setMinimizedCustomerResearch(true); setShowCustomerResearchTable(false); }}
                    style={{ position: 'absolute', top: '10px', right: '15px', background: 'none', border: 'none', fontSize: '24px', cursor: 'pointer', color: '#666', zIndex: 100 }}
                    title="Minimize Table"
                  >
                    &times;
                  </button>
                  <div className="research-summary-row">
                  <span><strong>Search:</strong> {customerResearchResults.query}</span>
                  <span><strong>Location:</strong> {customerResearchResults.location}</span>
                  <span><strong>Industry:</strong> {customerResearchResults.industry}</span>
                  <span><strong>Total Results:</strong> {customerResearchResults.totalResults}</span>
                </div>

                {isLoadingResearch ? (
                  <div className="loading">Loading businesses...</div>
                ) : customerResearchResults.businesses && customerResearchResults.businesses.length > 0 ? (
                  <div className="table-wrapper">
                    <table className="businesses-table">
                      <thead>
                        <tr>
                          <th>Business Name</th>
                          <th>Address</th>
                          <th>Phone</th>
                          <th>Website</th>
                          <th>Email</th>
                          <th>LinkedIn</th>
                          <th>Send Email</th>
                          <th>Match</th>
                          <th>Summary</th>
                        </tr>
                      </thead>
                                <tbody>
                                  {customerResearchResults.businesses.map((business, index) => (
                                    <tr key={index}>
                                      <td>{business.name || 'N/A'}</td>
                                      <td>{business.address || 'N/A'}</td>
                                      <td>{business.phone || 'N/A'}</td>
                                      <td>
                                        {business.website ? (
                                          <a href={business.website} target="_blank" rel="noopener noreferrer">
                                            Visit
                                          </a>
                                        ) : (
                                          'N/A'
                                        )}
                                      </td>
                                      <td>
                                        {business.email && business.email !== 'N/A' ? ( 
                                          <span>{business.email}</span>
                                        ) : (
                                          <button
                                            className="extract-email-button"
                                            onClick={() => handleExtractEmailForBusiness(business, index)}
                                            disabled={!!extractingEmailRows[index]}     
                                          >
                                            {extractingEmailRows[index] ? 'Extracting...' : 'Extract Email'}
                                          </button>
                                        )}
                                      </td>
                                      <td>
                                        {business.linkedin ? (
                                          business.linkedin !== 'N/A' ? (
                                            <a href={business.linkedin} target="_blank" rel="noopener noreferrer" style={{ color: '#0d6efd', textDecoration: 'none', fontWeight: 'bold' }}>
                                              View Profile
                                            </a>
                                          ) : (
                                            <span style={{ color: '#999', fontStyle: 'italic', fontSize: '0.9em' }}>Not Found</span>
                                          )
                                        ) : (
                                          <button
                                            className="extract-email-button"
                                            style={{ background: '#0a66c2', color: 'white', border: 'none' }}
                                            onClick={() => handleExtractLinkedInForBusiness(business, index)}
                                            disabled={!!extractingLinkedInRows[index]}
                                          >
                                            {extractingLinkedInRows[index] ? 'Extracting...' : 'Extract LinkedIn'}
                                          </button>
                                        )}
                                      </td>
                                      <td>
                                        {business.email && business.email !== 'N/A' ? (
                                          <button
                                            onClick={() => handleGeneratePersonalizedEmail(business, index)}
                                            disabled={!!isGeneratingEmail[index]}
                                            style={{ backgroundColor: '#3b82f6', color: 'white', border: 'none', padding: '4px 8px', borderRadius: '4px', cursor: 'pointer', fontSize: '11px', fontWeight: 'bold' }}
                                          >
                                            {isGeneratingEmail[index] ? 'Drafting...' : 'Draft Email'}
                                          </button>
                                        ) : (
                                          <span style={{ color: '#999', fontSize: '0.9em' }}>-</span>
                                        )}
                                      </td>
                                      <td style={{ textAlign: 'center', verticalAlign: 'middle' }}>
                                        {business.match_score != null ? (
                                          <div style={{ display: 'flex', flexDirection: 'column', alignItems: 'center' }}>
                                            <div style={{ fontWeight: 700, color: business.match_score >= 70 ? '#0f766e' : (business.match_score >= 40 ? '#f59e0b' : '#9ca3af') }}>{business.match_score}%</div>
                                          </div>
                                        ) : <span style={{ color: '#999' }}>-</span>}
                                      </td>
                                      <td style={{ maxWidth: '320px', whiteSpace: 'pre-line', lineHeight: '1.4' }} title={business.short_summary || business.summary || business.description || 'N/A'}>{business.short_summary || business.summary || business.description || 'N/A'}</td>
                                    </tr>
                                  ))}
                                </tbody>
                    </table>
                  </div>
                ) : (
                  <div className="no-results">No businesses found matching your search criteria.</div>
                )}

                <div className="modal-buttons">
                  <button className="minimize-popup-button" onClick={() => { setMinimizedCustomerResearch(true); setShowCustomerResearchTable(false); }}>
                    Minimize
                  </button>
                </div>
              </div>
            </div>
          )}


          {showPromptsPopup && (
            <div className="popup-overlay">
              <div className="popup-content">
                <h3>Previous Prompts</h3>
                <ul>
                  {previousPrompts.map((prompt, index) => (
                    <li key={index}>
                      <strong>Prompt ID:</strong> {prompt.id}
                      <br />
                      <strong>Overview:</strong> {prompt.overview}
                      <br />
                      <strong>Context:</strong> {prompt.context}
                      <br />
                      <strong>Countries:</strong> {prompt.countries}
                      <br />
                      <strong>Industries:</strong> {prompt.industries}
                      <br />
                      <strong>Business Functions:</strong> {prompt.businessFunctions}
                      <br />
                      <strong>Frameworks:</strong> {prompt.analysisFrameworks.join(', ')}
                      <br />
                      <strong>Response Format:</strong> {prompt.responseFormat}
                    </li>
                  ))}
                </ul>
                <button className="close-popup-button" onClick={() => setShowPromptsPopup(false)}>
                  Close
                </button>
              </div>
            </div>
          )}

          {showExportModal && (
            <div className="popup-overlay">
              <div className="popup-content export-options-modal">
                <h3>Export Market Research</h3>
                <div className="export-options-grid">
                  <button onClick={() => handleExport('excel')}>Download Excel (.xlsx)</button>
                  <button onClick={() => handleExport('csv')}>Download CSV (.csv)</button>
                  <button onClick={() => handleExport('pdf')}>Download PDF (.pdf)</button>
                  <button onClick={() => handleExport('json')}>Download JSON (.json)</button>
                  <button onClick={() => handleExport('sheets')}>Open in Google Sheets</button>
                </div>
                <div className="modal-buttons">
                  <button className="close-popup-button" onClick={() => setShowExportModal(false)}>
                    Close
                  </button>
                </div>
              </div>
            </div>
          )}


            {showEmailModal && (
            <div className="popup-overlay">
              <div className="popup-content email-modal-large">
                <div className="email-modal-header">
                  <h3>Draft Email Campaign</h3>
                  <button 
                    className="modal-close-btn"
                    onClick={() => { 
                      setShowEmailModal(false); 
                      setSelectedLead(null);
                      setEmailImages([]);
                      setIsAddingNewCampaign(false);
                    }}
                  >
                    ×
                  </button>
                </div>

                <div className="email-modal-body">
                  {/* Campaign Name - Dropdown with Add New Option */}
                  {!selectedLead && (
                    <div className="input-group">
                      <label>Campaign Name</label>
                      {existingCampaigns.length > 0 && !isAddingNewCampaign ? (
                        <div className="campaign-selector">
                          <select 
                            value={selectedCampaignId} 
                            onChange={(e) => handleCampaignSelect(e.target.value)}
                            className="campaign-dropdown"
                          >
                            <option value="">Select an existing campaign...</option>
                            {existingCampaigns.map(campaign => (
                              <option key={campaign.id} value={campaign.id}>
                                {campaign.name}
                              </option>
                            ))}
                            <option value="new">+ Add New Campaign</option>
                          </select>
                        </div>
                      ) : (
                        <div>
                          <input 
                            type="text" 
                            value={campaignName} 
                            onChange={(e) => setCampaignName(e.target.value)} 
                            placeholder="e.g., Tech Startups Dec 2026"
                            className="campaign-input"
                          />
                          {!isAddingNewCampaign && existingCampaigns.length > 0 && (
                            <button 
                              className="use-existing-btn"
                              onClick={() => setIsAddingNewCampaign(false)}
                            >
                              Use Existing Campaign
                            </button>
                          )}
                        </div>
                      )}
                    </div>
                  )}

                  {/* AI Personalization Checkbox - Properly Aligned */}
                  {!selectedLead && (
                    <div className="checkbox-group">
                      <input 
                        type="checkbox" 
                        checked={useAiBulk} 
                        onChange={(e) => setUseAiBulk(e.target.checked)} 
                        id="useAiBulkCheck"
                        className="checkbox-input"
                      />
                      <label htmlFor="useAiBulkCheck" className="checkbox-label">
                        Use AI Personalization for Bulk Emails
                      </label>
                    </div>
                  )}

                  {/* Subject Field */}
                  <div className="input-group">
                    <label>Subject Line</label>
                    <input 
                      type="text" 
                      value={emailSubject} 
                      onChange={(e) => setEmailSubject(e.target.value)} 
                      placeholder="Email Subject"
                      className="subject-input"
                    />
                  </div>

                  {/* Email Body with Image Support */}
                  <div className="input-group">
                    <div className="body-label-row">
                      <label>Email Body</label>
                      <span className="body-helper-text">You can use {"{"}Company{"}"} for dynamic content</span>
                    </div>
                    <textarea 
                      value={emailBody} 
                      onChange={(e) => setEmailBody(e.target.value)} 
                      placeholder="Type your email body here..." 
                      rows={10}
                      className="body-textarea"
                    ></textarea>

                    {/* Image Upload Section */}
                    <div className="image-upload-section">
                      <label className="image-label">Add Images to Email</label>
                      <div className="image-upload-controls">
                        <input 
                          type="file" 
                          multiple 
                          accept="image/*"
                          onChange={handleImageUpload}
                          className="image-file-input"
                          id="emailImageInput"
                        />
                        <label htmlFor="emailImageInput" className="image-upload-button">
                          Choose Images
                        </label>
                      </div>

                      {emailImages.length > 0 && (
                        <div className="image-gallery">
                          <p className="gallery-title">Selected Images ({emailImages.length}):</p>
                          <div className="image-list">
                            {emailImages.map((img, index) => (
                              <div key={index} className="image-item">
                                <div className="image-preview">
                                  <img src={img.data} alt={img.name} />
                                </div>
                                <div className="image-actions">
                                  <button 
                                    type="button"
                                    className="image-insert-btn"
                                    onClick={() => insertImageIntoBody(index)}
                                    title="Insert into body"
                                  >
                                    Insert
                                  </button>
                                  <button 
                                    type="button"
                                    className="image-remove-btn"
                                    onClick={() => removeEmailImage(index)}
                                    title="Remove image"
                                  >
                                    Remove
                                  </button>
                                </div>
                              </div>
                            ))}
                          </div>
                        </div>
                      )}
                    </div>
                  </div>
                </div>

                {/* Footer with Improved Buttons */}
                <div className="email-modal-footer">
                  <button 
                    className="email-cancel-button" 
                    onClick={() => { 
                      setShowEmailModal(false); 
                      setSelectedLead(null);
                      setEmailImages([]);
                      setIsAddingNewCampaign(false);
                    }}
                  >
                    Cancel
                  </button>
                  <button 
                    className="email-send-button" 
                    onClick={handleSendEmails} 
                    disabled={isSendingEmails}
                  >
                    {isSendingEmails ? 'Sending...' : 'Send Email'}
                  </button>
                </div>
              </div>
            </div>
          )}

          {/* Saved List Modal */}
          {showSaveListModal && (
            <div className="popup-overlay" style={{ zIndex: 3000 }}>
              <div className="popup-content" style={{ width: '400px', display: 'flex', flexDirection: 'column', gap: '15px' }}>
                <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
                  <h2 style={{ margin: 0, color: '#1E3A5F' }}>{`${getResearchEntityMeta().title} List`}</h2>
                  <button onClick={() => { setShowSaveListModal(false); setSaveListMode('create'); setSelectedAppendProjectId(''); }} style={{ background: 'none', border: 'none', fontSize: '20px', cursor: 'pointer' }}>×</button>
                </div>
                <div style={{ display: 'flex', flexDirection: 'column', gap: '8px' }}>
                  <label style={{ fontSize: '14px', fontWeight: 'bold' }}>Action</label>
                  <select
                    value={saveListMode}
                    onChange={(e) => setSaveListMode(e.target.value)}
                    style={{ padding: '10px', borderRadius: '5px', border: '1px solid #ccc', outline: 'none' }}
                  >
                    <option value="create">Create new list</option>
                    <option value="append">Append to existing list</option>
                  </select>
                </div>
                {saveListMode === 'append' ? (
                  <div style={{ display: 'flex', flexDirection: 'column', gap: '8px' }}>
                    <label style={{ fontSize: '14px', fontWeight: 'bold' }}>Select List to Append</label>
                    <select
                      value={selectedAppendProjectId}
                      onChange={(e) => setSelectedAppendProjectId(e.target.value)}
                      style={{ padding: '10px', borderRadius: '5px', border: '1px solid #ccc', outline: 'none' }}
                    >
                      <option value="">Choose a saved list...</option>
                      {savedLists.map((list) => (
                        <option key={list.id} value={list.id}>
                          {list.name}
                        </option>
                      ))}
                    </select>
                    <div style={{ fontSize: '12px', color: '#6b7280' }}>
                      Duplicate {getResearchEntityMeta().plural} are removed automatically when appending.
                    </div>
                  </div>
                ) : (
                <div style={{ display: 'flex', flexDirection: 'column', gap: '8px' }}>
                  <label style={{ fontSize: '14px', fontWeight: 'bold' }}>List Name</label>
                  <input type="text" value={saveListName} onChange={e => setSaveListName(e.target.value)} placeholder="E.g. NY Dentists Campaign" style={{ padding: '10px', borderRadius: '5px', border: '1px solid #ccc', outline: 'none' }} autoFocus />
                </div>
                )}
                <div style={{ display: 'flex', justifyContent: 'flex-end', gap: '10px', marginTop: '10px' }}>
                  <button onClick={() => { setShowSaveListModal(false); setSaveListMode('create'); setSelectedAppendProjectId(''); }} style={{ padding: '8px 15px', background: '#ccc', border: 'none', borderRadius: '5px', cursor: 'pointer', color: '#333' }}>Cancel</button>
                  <button onClick={handleSaveList} disabled={isSavingList || (saveListMode === 'append' && !selectedAppendProjectId)} style={{ padding: '8px 15px', background: '#1E3A5F', border: 'none', borderRadius: '5px', cursor: 'pointer', color: 'white' }}>
                    {isSavingList ? 'Saving...' : (saveListMode === 'append' ? `Append ${getResearchEntityMeta().title}` : `Save ${getResearchEntityMeta().title} List`)}
                  </button>
                </div>
              </div>
            </div>
          )}

          {/* Score Leads Modal */}
          {showScoreModal && (
            <div className="popup-overlay" style={{ zIndex: 3100 }}>
              <div className="popup-content" style={{ width: '520px', display: 'flex', flexDirection: 'column', gap: '12px' }}>
                <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
                  <h2 style={{ margin: 0, color: '#1E3A5F' }}>{`Score ${getResearchEntityMeta().title}`}</h2>
                  <button onClick={() => { setShowScoreModal(false); setScoreQueryText(''); }} style={{ background: 'none', border: 'none', fontSize: '20px', cursor: 'pointer' }}>×</button>
                </div>
                <div style={{ display: 'flex', flexDirection: 'column', gap: '8px' }}>
                  <label style={{ fontSize: '14px', fontWeight: 'bold' }}>Describe what you're trying to sell or match</label>
                  <textarea value={scoreQueryText} onChange={(e) => setScoreQueryText(e.target.value)} rows={4} placeholder="e.g. We sell fleet telematics hardware and software to logistics companies; looking for mid-size fleet operators in North America" style={{ padding: '10px', borderRadius: '6px', border: '1px solid #ccc', outline: 'none' }} />
                  <div style={{ fontSize: '12px', color: '#6b7280' }}>A concise description helps rank {getResearchEntityMeta().plural}. Results will add a Match score and a two-line summary for each company.</div>
                </div>
                <div style={{ display: 'flex', justifyContent: 'flex-end', gap: '10px', marginTop: '6px' }}>
                  <button onClick={() => { setShowScoreModal(false); setScoreQueryText(''); }} style={{ padding: '8px 15px', background: '#ccc', border: 'none', borderRadius: '5px', cursor: 'pointer', color: '#333' }}>Cancel</button>
                  <button onClick={handleScoreLeads} disabled={isScoring} style={{ padding: '8px 15px', background: '#1E3A5F', border: 'none', borderRadius: '5px', cursor: 'pointer', color: 'white' }}>{isScoring ? 'Scoring...' : `Score ${getResearchEntityMeta().title}`}</button>
                </div>
              </div>
            </div>
          )}

        </div>   
      </div>
    </div>
  );
}

export default RequirementsGathering;
