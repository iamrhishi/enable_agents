import React, { useState, useEffect, useCallback } from 'react';
import { useNavigate, useSearchParams } from 'react-router-dom';
import Header from '../core/Header';
import { BackButton, Textarea, Select, ProjectSelector, LiveModeHint, AgentOutcomesStrip, ProjectGate, EmptyState } from '../components';
import '../styles/DataInsights.css';
import { PDFDocument } from 'pdf-lib';
import { API_CONFIG } from '../config/apiConfig';
import { showToast } from '../core/toast';
import { useSelectedProjectId } from '../hooks/useSelectedProjectId';

// Document Intelligence API
const DOC_API = `${API_CONFIG.API_URL}/api/document-intelligence`;

// Allowed file types (must match backend)
const ALLOWED_EXTENSIONS = ['pdf', 'docx', 'doc', 'txt', 'xlsx', 'xls', 'csv'];

// Storage key for state persistence
const STATE_KEY = 'dataInsightsState';

// Demo documents with rich metadata (DDA-style)
const DEMO_DOCUMENTS = [
  {
    id: 'doc-1',
    name: 'Q3 Financial Report.pdf',
    type: 'pdf',
    size: '2.4 MB',
    uploadedAt: '2026-07-10',
    category: 'financial',
    pages: 24,
    status: 'completed',
    entities: 45,
    relationships: 28,
    confidence: 0.92,
  },
  {
    id: 'doc-2',
    name: 'Customer Survey Results.xlsx',
    type: 'xlsx',
    size: '1.1 MB',
    uploadedAt: '2026-07-12',
    category: 'research',
    pages: null,
    status: 'completed',
    entities: 32,
    relationships: 15,
    confidence: 0.88,
  },
  {
    id: 'doc-3',
    name: 'Product Roadmap 2026.pdf',
    type: 'pdf',
    size: '3.8 MB',
    uploadedAt: '2026-07-14',
    category: 'strategy',
    pages: 18,
    status: 'completed',
    entities: 28,
    relationships: 22,
    confidence: 0.95,
  },
  {
    id: 'doc-4',
    name: 'Market Analysis Report.pdf',
    type: 'pdf',
    size: '5.2 MB',
    uploadedAt: '2026-07-15',
    category: 'research',
    pages: 42,
    status: 'completed',
    entities: 56,
    relationships: 31,
    confidence: 0.91,
  },
];

// Demo extracted entities per document
const DEMO_ENTITIES = {
  'doc-1': [
    { id: 'e1', name: 'Revenue', type: 'metric', confidence: 0.95, value: '$4.2M' },
    { id: 'e2', name: 'Q3 2026', type: 'period', confidence: 0.98 },
    { id: 'e3', name: 'Enterprise Segment', type: 'segment', confidence: 0.92, value: '62%' },
    { id: 'e4', name: 'Gross Margin', type: 'metric', confidence: 0.94, value: '68%' },
    { id: 'e5', name: 'Operating Expenses', type: 'metric', confidence: 0.91, value: '$1.8M' },
    { id: 'e6', name: 'YoY Growth', type: 'trend', confidence: 0.89, value: '+18%' },
    { id: 'e7', name: 'Net Profit', type: 'metric', confidence: 0.93, value: '$1.0M' },
    { id: 'e8', name: 'New ARR', type: 'metric', confidence: 0.87, value: '$800K' },
  ],
  'doc-2': [
    { id: 'e9', name: 'NPS Score', type: 'metric', confidence: 0.96, value: '72' },
    { id: 'e10', name: 'CSAT Rating', type: 'metric', confidence: 0.94, value: '4.5/5' },
    { id: 'e11', name: 'Response Rate', type: 'metric', confidence: 0.92, value: '34%' },
    { id: 'e12', name: 'Product Quality', type: 'attribute', confidence: 0.88, value: '4.7/5' },
    { id: 'e13', name: 'Support Response', type: 'attribute', confidence: 0.85, value: '3.8/5' },
    { id: 'e14', name: 'Detractors', type: 'metric', confidence: 0.91, value: '8%' },
  ],
  'doc-3': [
    { id: 'e15', name: 'AI Assistant', type: 'feature', confidence: 0.97 },
    { id: 'e16', name: 'Mobile App v2', type: 'feature', confidence: 0.95 },
    { id: 'e17', name: 'Enterprise SSO', type: 'feature', confidence: 0.93 },
    { id: 'e18', name: 'Q4 Launch', type: 'milestone', confidence: 0.96 },
    { id: 'e19', name: 'Tech Debt Reduction', type: 'goal', confidence: 0.90, value: '15%' },
    { id: 'e20', name: 'Engineering Team', type: 'resource', confidence: 0.88, value: '12 FTEs' },
  ],
  'doc-4': [
    { id: 'e21', name: 'TAM', type: 'metric', confidence: 0.94, value: '$12B' },
    { id: 'e22', name: 'Market Growth', type: 'trend', confidence: 0.92, value: '24% CAGR' },
    { id: 'e23', name: 'Competitor A', type: 'company', confidence: 0.96, value: '32% share' },
    { id: 'e24', name: 'Competitor B', type: 'company', confidence: 0.95, value: '18% share' },
    { id: 'e25', name: 'Our Position', type: 'metric', confidence: 0.91, value: '8% share' },
    { id: 'e26', name: 'Target Segment', type: 'segment', confidence: 0.89, value: 'Mid-Market' },
    { id: 'e27', name: 'Growth Opportunity', type: 'trend', confidence: 0.87, value: '+45%' },
    { id: 'e28', name: 'Key Region', type: 'geography', confidence: 0.93, value: 'North America' },
  ],
};

// Demo knowledge graph data
const DEMO_GRAPHS = {
  'doc-1': {
    nodes: [
      { id: 'n1', label: 'Revenue', type: 'metric', x: 200, y: 150 },
      { id: 'n2', label: 'Q3 2026', type: 'period', x: 350, y: 80 },
      { id: 'n3', label: 'Enterprise', type: 'segment', x: 100, y: 80 },
      { id: 'n4', label: '$4.2M', type: 'value', x: 200, y: 250 },
      { id: 'n5', label: '+18% YoY', type: 'growth', x: 350, y: 200 },
      { id: 'n6', label: 'Gross Margin', type: 'metric', x: 50, y: 200 },
      { id: 'n7', label: '68%', type: 'value', x: 50, y: 280 },
    ],
    edges: [
      { source: 'n1', target: 'n4', label: 'equals' },
      { source: 'n1', target: 'n2', label: 'period' },
      { source: 'n3', target: 'n1', label: 'drives' },
      { source: 'n1', target: 'n5', label: 'growth' },
      { source: 'n6', target: 'n7', label: 'equals' },
      { source: 'n6', target: 'n1', label: 'impacts' },
    ],
  },
  'doc-2': {
    nodes: [
      { id: 'n1', label: 'NPS', type: 'metric', x: 200, y: 100 },
      { id: 'n2', label: '72', type: 'value', x: 200, y: 200 },
      { id: 'n3', label: 'Satisfaction', type: 'metric', x: 350, y: 150 },
      { id: 'n4', label: '4.5/5', type: 'value', x: 350, y: 250 },
      { id: 'n5', label: 'Product', type: 'category', x: 100, y: 180 },
    ],
    edges: [
      { source: 'n1', target: 'n2', label: 'score' },
      { source: 'n3', target: 'n4', label: 'rating' },
      { source: 'n5', target: 'n1', label: 'influences' },
      { source: 'n5', target: 'n3', label: 'influences' },
    ],
  },
  'doc-4': {
    nodes: [
      { id: 'n1', label: 'TAM', type: 'metric', x: 200, y: 100 },
      { id: 'n2', label: '$12B', type: 'value', x: 200, y: 200 },
      { id: 'n3', label: 'Competitor A', type: 'segment', x: 80, y: 150 },
      { id: 'n4', label: '32%', type: 'value', x: 80, y: 250 },
      { id: 'n5', label: 'Us', type: 'segment', x: 320, y: 150 },
      { id: 'n6', label: '8%', type: 'value', x: 320, y: 250 },
      { id: 'n7', label: 'Growth', type: 'growth', x: 200, y: 300 },
    ],
    edges: [
      { source: 'n1', target: 'n2', label: 'size' },
      { source: 'n3', target: 'n4', label: 'share' },
      { source: 'n5', target: 'n6', label: 'share' },
      { source: 'n3', target: 'n1', label: 'in' },
      { source: 'n5', target: 'n1', label: 'in' },
      { source: 'n5', target: 'n7', label: 'potential' },
    ],
  },
};

// Demo insights with key facts and sources
const DEMO_INSIGHTS = {
  'doc-1': {
    summary: 'Q3 Financial Performance Analysis',
    keyFacts: [
      { fact: 'Revenue reached $4.2M', confidence: 0.95, source: 'Page 3' },
      { fact: 'YoY growth of 18%', confidence: 0.92, source: 'Page 5' },
      { fact: 'Enterprise segment contributed 62%', confidence: 0.89, source: 'Page 8' },
      { fact: 'Operating expenses reduced by 5%', confidence: 0.91, source: 'Page 12' },
    ],
    recommendations: [
      'Expand enterprise sales team for high ROI',
      'Invest in customer success to improve retention',
      'Accelerate automation initiatives',
    ],
    sources: [
      { page: 3, text: 'Total Q3 revenue of $4.2 million...', relevance: 0.95 },
      { page: 5, text: 'Compared to Q3 2025, revenue increased by 18%...', relevance: 0.92 },
      { page: 8, text: 'Enterprise customers now represent 62% of total revenue...', relevance: 0.89 },
    ],
  },
  'doc-2': {
    summary: 'Customer Satisfaction Insights',
    keyFacts: [
      { fact: 'NPS score of 72 (up 8 points)', confidence: 0.96, source: 'Summary' },
      { fact: 'Product quality rated 4.7/5', confidence: 0.94, source: 'Section 2' },
      { fact: 'Support response time needs improvement', confidence: 0.88, source: 'Section 4' },
    ],
    recommendations: [
      'Prioritize mobile app improvements',
      'Enhance API documentation',
      'Reduce support response time',
    ],
    sources: [
      { page: 1, text: 'Net Promoter Score reached 72...', relevance: 0.96 },
      { page: 4, text: 'Average support response time of 4.2 hours...', relevance: 0.88 },
    ],
  },
  'doc-3': {
    summary: 'Product Strategy Overview',
    keyFacts: [
      { fact: '24 features planned for 2026', confidence: 0.97, source: 'Roadmap' },
      { fact: '8 features launching in Q4', confidence: 0.95, source: 'Q4 Section' },
      { fact: '15% tech debt reduction target', confidence: 0.90, source: 'Goals' },
    ],
    recommendations: [
      'Prioritize AI assistant feature launch',
      'Allocate resources for mobile v2',
      'Address tech debt before scaling',
    ],
    sources: [
      { page: 2, text: 'Q4 priorities include AI assistant launch...', relevance: 0.97 },
      { page: 5, text: 'Technical debt reduction target of 15%...', relevance: 0.90 },
    ],
  },
  'doc-4': {
    summary: 'Market Landscape & Competitive Analysis',
    keyFacts: [
      { fact: 'Total addressable market is $12B', confidence: 0.94, source: 'Page 4' },
      { fact: 'Market growing at 24% CAGR', confidence: 0.92, source: 'Page 6' },
      { fact: 'Top competitor holds 32% market share', confidence: 0.96, source: 'Page 12' },
      { fact: 'Our current position is 8% market share', confidence: 0.91, source: 'Page 15' },
      { fact: 'Mid-market segment shows highest growth potential', confidence: 0.89, source: 'Page 22' },
    ],
    recommendations: [
      'Focus expansion on mid-market segment',
      'Differentiate on AI capabilities vs competitors',
      'Increase North America sales presence',
      'Target 15% market share by 2027',
    ],
    sources: [
      { page: 4, text: 'The total addressable market for enterprise solutions reached $12 billion...', relevance: 0.94 },
      { page: 12, text: 'Competitor A maintains market leadership with 32% share...', relevance: 0.96 },
      { page: 22, text: 'Mid-market companies represent the fastest-growing segment at 45% growth...', relevance: 0.89 },
    ],
  },
};

// Suggested prompts
const SUGGESTED_PROMPTS = [
  'What are the key findings?',
  'Summarize the main metrics',
  'What are the recommendations?',
  'Identify potential risks',
  'Compare to previous period',
];

function DataInsights() {
  const navigate = useNavigate();
  const selectedProjectId = useSelectedProjectId();
  const [searchParams, setSearchParams] = useSearchParams();

  // Load persisted state
  const loadPersistedState = useCallback(() => {
    try {
      const saved = sessionStorage.getItem(STATE_KEY);
      return saved ? JSON.parse(saved) : {};
    } catch {
      return {};
    }
  }, []);

  // Demo mode detection
  const [isDemoMode, setIsDemoMode] = useState(() => {
    return localStorage.getItem('enableAgentsMode') !== 'live';
  });

  // Main view tabs
  const [activeView, setActiveView] = useState(() => {
    return searchParams.get('view') || loadPersistedState().activeView || 'library';
  });

  // Documents
  const [documents, setDocuments] = useState([]);
  const [selectedDocument, setSelectedDocument] = useState(null);

  // Upload state
  const [files, setFiles] = useState([]);
  const [isBulkMode, setIsBulkMode] = useState(false);
  const [isUploading, setIsUploading] = useState(false);

  // Query state
  const [inputPrompt, setInputPrompt] = useState('');
  const [insightsEngine, setInsightsEngine] = useState('Contextual Insights with RAG');

  // Results
  const [currentInsight, setCurrentInsight] = useState(null);
  const [conversationHistory, setConversationHistory] = useState([]);
  const [isLoading, setIsLoading] = useState(false);

  // Analysis view sub-tab
  const [analysisTab, setAnalysisTab] = useState('insights');

  // Persist state
  useEffect(() => {
    const state = {
      activeView,
      selectedDocId: selectedDocument?.id || null,
    };
    sessionStorage.setItem(STATE_KEY, JSON.stringify(state));

    // Update URL
    const params = new URLSearchParams(searchParams);
    if (activeView !== 'library') {
      params.set('view', activeView);
    } else {
      params.delete('view');
    }
    if (selectedDocument?.id) {
      params.set('doc', selectedDocument.id);
    } else {
      params.delete('doc');
    }
    setSearchParams(params, { replace: true });
  }, [activeView, selectedDocument, searchParams, setSearchParams]);

  // Listen for mode changes
  useEffect(() => {
    const handleModeChange = () => {
      const newMode = localStorage.getItem('enableAgentsMode') !== 'live';
      if (isDemoMode !== newMode) {
        setIsDemoMode(newMode);
        setCurrentInsight(null);
        setConversationHistory([]);
        setSelectedDocument(null);
      }
    };
    window.addEventListener('storage', handleModeChange);
    return () => window.removeEventListener('storage', handleModeChange);
  }, [isDemoMode]);

  // Load documents when project selected
  useEffect(() => {
    if (!selectedProjectId) {
      setDocuments([]);
      setSelectedDocument(null);
      setCurrentInsight(null);
      return;
    }

    if (isDemoMode) {
      setDocuments(DEMO_DOCUMENTS);
    } else {
      // Live mode - fetch from backend
      fetchDocuments();
    }
  }, [isDemoMode, selectedProjectId]);

  // Fetch documents from backend (live mode)
  const fetchDocuments = async () => {
    if (!selectedProjectId) return;

    try {
      const userEmail = localStorage.getItem('userEmail') || '';
      const res = await fetch(`${DOC_API}/documents?project_id=${selectedProjectId}`, {
        headers: { 'X-User-Id': userEmail },
      });

      if (res.ok) {
        const data = await res.json();
        const docs = (data.documents || []).map(doc => ({
          id: doc.document_id,
          name: doc.file_name,
          type: doc.file_type,
          size: formatFileSize(doc.file_size),
          uploadedAt: doc.created_at?.split('T')[0] || 'Unknown',
          category: doc.document_type || 'other',
          pages: doc.page_count,
          status: doc.status,
          entities: doc.entity_count || 0,
          relationships: 0, // Not tracked separately in backend
          confidence: 0.85, // Default confidence
          processingProgress: doc.processing_progress,
          processingStage: doc.processing_stage,
        }));
        setDocuments(docs);
      }
    } catch (err) {
      console.error('Failed to fetch documents:', err);
      showToast('Failed to load documents', 'error');
    }
  };

  // Helper to format file size
  const formatFileSize = (bytes) => {
    if (!bytes) return 'Unknown';
    if (bytes < 1024) return `${bytes} B`;
    if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(1)} KB`;
    return `${(bytes / (1024 * 1024)).toFixed(1)} MB`;
  };

  // Poll for document processing status
  const pollDocumentStatus = async (documentId) => {
    const userEmail = localStorage.getItem('userEmail') || '';
    let attempts = 0;
    const maxAttempts = 60; // 5 minutes max

    const poll = async () => {
      try {
        const res = await fetch(`${DOC_API}/status/${documentId}?project_id=${selectedProjectId}`, {
          headers: { 'X-User-Id': userEmail },
        });

        if (res.ok) {
          const status = await res.json();

          // Update document in list
          setDocuments(prev => prev.map(d =>
            d.id === documentId
              ? {
                  ...d,
                  status: status.status,
                  entities: status.entity_count || 0,
                  processingProgress: status.processing_progress,
                  processingStage: status.processing_stage,
                }
              : d
          ));

          if (status.status === 'completed') {
            showToast(`Document processed: ${status.chunk_count} chunks, ${status.entity_count} entities`, 'success');
            return;
          } else if (status.status === 'failed') {
            showToast(`Processing failed: ${status.error_message}`, 'error');
            return;
          }
        }

        attempts++;
        if (attempts < maxAttempts) {
          setTimeout(poll, 5000); // Poll every 5 seconds
        }
      } catch (err) {
        console.error('Status poll failed:', err);
      }
    };

    poll();
  };

  // Restore selected document from URL
  useEffect(() => {
    const docId = searchParams.get('doc');
    if (docId && documents.length > 0 && !selectedDocument) {
      const doc = documents.find(d => d.id === docId);
      if (doc) {
        handleSelectDocument(doc);
      }
    }
  }, [documents, searchParams, selectedDocument]);

  // Helpers
  const getFileIcon = (type) => {
    const icons = {
      pdf: '/assets/icons/pdf.png',
      xlsx: '/assets/icons/data-discovery.png',
      xls: '/assets/icons/data-discovery.png',
      csv: '/assets/icons/data-discovery.png',
      txt: '/assets/icons/document.png',
      doc: '/assets/icons/word.png',
      docx: '/assets/icons/word.png',
      json: '/assets/icons/database.png',
    };
    return icons[type] || '/assets/icons/document.png';
  };

  const getStatusColor = (status) => {
    const colors = { completed: 'success', processing: 'warning', failed: 'error' };
    return colors[status] || 'default';
  };

  const getConfidenceColor = (confidence) => {
    if (confidence >= 0.9) return 'high';
    if (confidence >= 0.7) return 'medium';
    return 'low';
  };

  const getTotalStats = () => {
    const completed = documents.filter(d => d.status === 'completed');
    return {
      totalDocs: documents.length,
      completedDocs: completed.length,
      processingDocs: documents.filter(d => d.status === 'processing').length,
      totalEntities: completed.reduce((sum, d) => sum + (d.entities || 0), 0),
      totalRelationships: completed.reduce((sum, d) => sum + (d.relationships || 0), 0),
      avgConfidence: completed.length > 0
        ? (completed.reduce((sum, d) => sum + (d.confidence || 0), 0) / completed.length * 100).toFixed(0)
        : 0,
    };
  };

  // Handlers
  const handleSelectDocument = (doc) => {
    setSelectedDocument(doc);
    setActiveView('analysis');
    setAnalysisTab('insights');

    // Load demo insights if available
    if (isDemoMode && DEMO_INSIGHTS[doc.id]) {
      setCurrentInsight(DEMO_INSIGHTS[doc.id]);
    } else {
      setCurrentInsight(null);
    }
  };

  const handleFileUpload = (e) => {
    let selectedFiles = Array.from(e.target.files);
    if (isBulkMode) {
      if (selectedFiles.length > 10) {
        showToast('Maximum 10 files allowed', 'warning');
        selectedFiles = selectedFiles.slice(0, 10);
      }
    } else {
      selectedFiles = selectedFiles.slice(0, 1);
    }
    setFiles(selectedFiles);
  };

  const handleUploadFiles = async () => {
    if (files.length === 0) return;

    // Validate file types
    const invalidFiles = files.filter(f => {
      const ext = f.name.split('.').pop()?.toLowerCase();
      return !ALLOWED_EXTENSIONS.includes(ext);
    });

    if (invalidFiles.length > 0) {
      showToast(`Unsupported file type. Allowed: ${ALLOWED_EXTENSIONS.join(', ')}`, 'error');
      return;
    }

    setIsUploading(true);

    if (isDemoMode) {
      setTimeout(() => {
        const newDocs = files.map((file, idx) => ({
          id: `doc-new-${Date.now()}-${idx}`,
          name: file.name,
          type: file.name.split('.').pop().toLowerCase(),
          size: `${(file.size / 1024 / 1024).toFixed(1)} MB`,
          uploadedAt: new Date().toISOString().split('T')[0],
          category: 'other',
          pages: null,
          status: 'processing',
          entities: 0,
          relationships: 0,
          confidence: 0,
        }));
        setDocuments([...newDocs, ...documents]);
        setFiles([]);
        setIsUploading(false);
        showToast(`${files.length} file(s) uploaded - processing started`, 'info');

        // Simulate processing completion after 3 seconds
        setTimeout(() => {
          setDocuments(prev => prev.map(d =>
            newDocs.find(n => n.id === d.id)
              ? { ...d, status: 'completed', entities: Math.floor(Math.random() * 30) + 10, relationships: Math.floor(Math.random() * 20) + 5, confidence: 0.85 + Math.random() * 0.1 }
              : d
          ));
          showToast('Document processing completed', 'success');
        }, 3000);
      }, 1000);
      return;
    }

    // Live mode - upload to backend API
    const userEmail = localStorage.getItem('userEmail') || '';
    const uploadedDocIds = [];

    try {
      for (const file of files) {
        const formData = new FormData();
        formData.append('file', file);
        formData.append('project_id', selectedProjectId);

        const response = await fetch(`${DOC_API}/upload`, {
          method: 'POST',
          headers: { 'X-User-Id': userEmail },
          body: formData,
        });

        if (!response.ok) {
          const err = await response.json();
          throw new Error(err.error || 'Upload failed');
        }

        const result = await response.json();
        uploadedDocIds.push(result.document_id);

        // Add to documents list immediately with processing status
        const newDoc = {
          id: result.document_id,
          name: result.file_name,
          type: file.name.split('.').pop()?.toLowerCase() || 'unknown',
          size: formatFileSize(result.file_size),
          uploadedAt: new Date().toISOString().split('T')[0],
          category: 'other',
          pages: null,
          status: 'processing',
          entities: 0,
          relationships: 0,
          confidence: 0,
        };

        setDocuments(prev => [newDoc, ...prev]);
      }

      setFiles([]);
      showToast(`${files.length} file(s) uploaded - processing started`, 'success');

      // Start polling for each document
      uploadedDocIds.forEach(docId => pollDocumentStatus(docId));

    } catch (error) {
      console.error('Upload error:', error);
      showToast(`Upload failed: ${error.message}`, 'error');
    } finally {
      setIsUploading(false);
    }
  };

  const handleAskQuestion = async () => {
    if (!selectedDocument || !inputPrompt.trim()) {
      showToast('Please select a document and enter a question', 'warning');
      return;
    }

    setIsLoading(true);

    // Add to conversation
    const userMessage = { role: 'user', content: inputPrompt, timestamp: new Date() };
    setConversationHistory(prev => [...prev, userMessage]);

    if (isDemoMode) {
      setTimeout(() => {
        const baseInsight = DEMO_INSIGHTS[selectedDocument.id] || DEMO_INSIGHTS['doc-1'];
        const aiResponse = {
          role: 'assistant',
          content: `Based on **${selectedDocument.name}**:\n\n${baseInsight.keyFacts.map(f => `• ${f.fact}`).join('\n')}\n\n**Recommendations:**\n${baseInsight.recommendations.map(r => `• ${r}`).join('\n')}`,
          timestamp: new Date(),
          sources: baseInsight.sources,
          confidence: 0.92,
        };
        setConversationHistory(prev => [...prev, aiResponse]);
        setInputPrompt('');
        setIsLoading(false);
      }, 1500);
      return;
    }

    // Live mode - use document intelligence chat API
    const userEmail = localStorage.getItem('userEmail') || '';

    try {
      const response = await fetch(`${DOC_API}/chat`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
          'X-User-Id': userEmail,
        },
        body: JSON.stringify({
          query: inputPrompt,
          document_ids: [selectedDocument.id],
          project_id: selectedProjectId,
          use_entity_boost: true,
        }),
      });

      if (!response.ok) {
        const err = await response.json();
        throw new Error(err.error || 'Chat failed');
      }

      const data = await response.json();

      // Format sources from backend response
      const formattedSources = (data.sources || []).map(s => ({
        page: s.page_number || 1,
        text: s.content?.substring(0, 100) || '',
        relevance: s.score || 0.8,
      }));

      const aiResponse = {
        role: 'assistant',
        content: data.answer,
        timestamp: new Date(),
        sources: formattedSources,
        confidence: 0.85,
        chunkCount: data.chunk_count,
      };

      setConversationHistory(prev => [...prev, aiResponse]);
      setInputPrompt('');
    } catch (error) {
      console.error('Chat error:', error);
      showToast(`Failed to get response: ${error.message}`, 'error');

      // Add error message to conversation
      const errorResponse = {
        role: 'assistant',
        content: `Sorry, I couldn't process your question. Error: ${error.message}`,
        timestamp: new Date(),
        sources: [],
        confidence: 0,
      };
      setConversationHistory(prev => [...prev, errorResponse]);
    } finally {
      setIsLoading(false);
    }
  };

  // State for live mode entities
  const [liveEntities, setLiveEntities] = useState([]);

  // Fetch entities for document (live mode)
  const fetchDocumentEntities = async (documentId) => {
    if (isDemoMode) return;

    const userEmail = localStorage.getItem('userEmail') || '';
    try {
      const res = await fetch(`${DOC_API}/status/${documentId}?project_id=${selectedProjectId}`, {
        headers: { 'X-User-Id': userEmail },
      });

      if (res.ok) {
        const data = await res.json();
        // Backend returns entities with the status - format them
        if (data.entities) {
          const formatted = Object.entries(data.entities).flatMap(([type, names]) =>
            names.map((name, idx) => ({
              id: `${type}-${idx}`,
              name,
              type,
              confidence: 0.8,
            }))
          );
          setLiveEntities(formatted);
        }
      }
    } catch (err) {
      console.error('Failed to fetch entities:', err);
    }
  };

  // Get entities for selected document
  const getDocumentEntities = () => {
    if (!selectedDocument) return [];
    if (isDemoMode) {
      return DEMO_ENTITIES[selectedDocument.id] || [];
    }
    return liveEntities;
  };

  // Fetch entities when document is selected (live mode)
  useEffect(() => {
    if (selectedDocument && !isDemoMode) {
      fetchDocumentEntities(selectedDocument.id);
    }
  }, [selectedDocument, isDemoMode]);

  // Get graph for selected document
  const getDocumentGraph = () => {
    if (!selectedDocument || !isDemoMode) return null;
    return DEMO_GRAPHS[selectedDocument.id] || null;
  };

  const stats = getTotalStats();

  return (
    <div className="data-insights-page">
      <Header />

      <div className="agent-page-header">
        <div className="agent-header-left">
          <BackButton />
          <div className="agent-header-content">
            <div className="agent-title-row">
              <h1>Data Insights</h1>
            </div>
            <p className="text-muted">
              {selectedDocument
                ? `Analyzing: ${selectedDocument.name}`
                : 'AI-powered document analysis with knowledge extraction'}
            </p>
          </div>
        </div>
        <div className="agent-header-right">
          <ProjectSelector agentKey="dataInsights" />
        </div>
      </div>

      <AgentOutcomesStrip
        items={[
          { iconSrc: '/assets/icons/bar-chart.png', title: 'Smart Q&A', description: 'Ask questions, get sourced answers.' },
          { iconSrc: '/assets/icons/data-discovery.png', title: 'Entity extraction', description: 'Auto-extract key facts and metrics.' },
          { iconSrc: '/assets/icons/performance.png', title: 'Knowledge graph', description: 'Visualize relationships in your data.' },
        ]}
      />

      <LiveModeHint
        requireProject
        message="Choose a project, then upload documents or switch to Demo for sample data."
      />

      <ProjectGate agentLabel="Data Insights workspace">
        <div className="di-container">
          {/* Overall Stats Strip */}
          {documents.length > 0 && (
            <div className="di-overall-stats">
              <div className="di-stat-item">
                <div className="di-stat-icon">
                  <img src="/assets/icons/bar-chart.png" alt="" width={20} height={20} />
                </div>
                <div className="di-stat-content">
                  <span className="di-stat-value">{stats.totalDocs}</span>
                  <span className="di-stat-label">Documents</span>
                </div>
              </div>
              <div className="di-stat-item">
                <div className="di-stat-icon di-stat-icon--success">
                  <img src="/assets/icons/performance.png" alt="" width={20} height={20} />
                </div>
                <div className="di-stat-content">
                  <span className="di-stat-value">{stats.completedDocs}</span>
                  <span className="di-stat-label">Processed</span>
                </div>
              </div>
              <div className="di-stat-item">
                <div className="di-stat-icon di-stat-icon--accent">
                  <img src="/assets/icons/data-discovery.png" alt="" width={20} height={20} />
                </div>
                <div className="di-stat-content">
                  <span className="di-stat-value">{stats.totalEntities}</span>
                  <span className="di-stat-label">Entities</span>
                </div>
              </div>
              <div className="di-stat-item">
                <div className="di-stat-icon di-stat-icon--primary">
                  <img src="/assets/icons/networking.png" alt="" width={20} height={20} />
                </div>
                <div className="di-stat-content">
                  <span className="di-stat-value">{stats.totalRelationships}</span>
                  <span className="di-stat-label">Relations</span>
                </div>
              </div>
              <div className="di-stat-item">
                <div className="di-stat-icon di-stat-icon--info">
                  <img src="/assets/icons/search-analysis.png" alt="" width={20} height={20} />
                </div>
                <div className="di-stat-content">
                  <span className="di-stat-value">{stats.avgConfidence}%</span>
                  <span className="di-stat-label">Confidence</span>
                </div>
              </div>
            </div>
          )}

          {/* Main View Tabs */}
          <div className="di-main-tabs">
            <button
              className={`di-main-tab ${activeView === 'library' ? 'di-main-tab--active' : ''}`}
              onClick={() => setActiveView('library')}
            >
              <img src="/assets/icons/bar-chart.png" alt="" className="di-tab-icon" />
              Library
            </button>
            <button
              className={`di-main-tab ${activeView === 'analysis' ? 'di-main-tab--active' : ''}`}
              onClick={() => selectedDocument && setActiveView('analysis')}
              disabled={!selectedDocument}
            >
              <img src="/assets/icons/data-discovery.png" alt="" className="di-tab-icon" />
              Analysis
              {selectedDocument && <span className="di-tab-doc">{selectedDocument.name.substring(0, 15)}...</span>}
            </button>
            <button
              className={`di-main-tab ${activeView === 'chat' ? 'di-main-tab--active' : ''}`}
              onClick={() => selectedDocument && setActiveView('chat')}
              disabled={!selectedDocument}
            >
              <img src="/assets/icons/ai-chatbots.png" alt="" className="di-tab-icon" />
              Ask AI
            </button>
          </div>

          {/* Views */}
          {activeView === 'library' && (
            <div className="di-library-view">
              <div className="di-library-layout">
                {/* Upload Panel */}
                <div className="di-upload-panel">
                  <div className="di-panel-header">
                    <h3>Upload Documents</h3>
                    <label className="di-bulk-toggle">
                      <input
                        type="checkbox"
                        checked={isBulkMode}
                        onChange={(e) => { setIsBulkMode(e.target.checked); setFiles([]); }}
                      />
                      Bulk
                    </label>
                  </div>

                  <div className="di-upload-zone">
                    <input
                      type="file"
                      id="file-input"
                      onChange={handleFileUpload}
                      multiple={isBulkMode}
                      accept=".pdf,.csv,.xlsx,.xls,.txt,.json"
                      className="di-file-input"
                    />
                    <label htmlFor="file-input" className="di-upload-label">
                      <div className="di-upload-icon">+</div>
                      <span>Drop files or click to upload</span>
                      <span className="di-upload-formats">PDF, CSV, XLSX, TXT, JSON</span>
                    </label>
                  </div>

                  {files.length > 0 && (
                    <div className="di-pending-files">
                      <div className="di-pending-header">
                        <span>{files.length} file(s) ready</span>
                        <button onClick={() => setFiles([])}>×</button>
                      </div>
                      <ul>{files.map((f, i) => <li key={i}>{f.name}</li>)}</ul>
                      <button className="di-upload-btn" onClick={handleUploadFiles} disabled={isUploading}>
                        {isUploading ? 'Uploading...' : 'Upload & Process'}
                      </button>
                    </div>
                  )}

                  {/* Processing Status */}
                  {stats.processingDocs > 0 && (
                    <div className="di-processing-status">
                      <div className="di-processing-header">
                        <span className="di-processing-spinner" />
                        <span>Processing {stats.processingDocs} document(s)...</span>
                      </div>
                    </div>
                  )}
                </div>

                {/* Documents Grid */}
                <div className="di-documents-panel">
                  <div className="di-panel-header">
                    <h3>Document Library</h3>
                    <span className="di-doc-count">{documents.length} files</span>
                  </div>

                  {documents.length === 0 ? (
                    <EmptyState
                      iconType="document"
                      title="No documents yet"
                      description="Upload documents to start extracting insights with AI."
                    />
                  ) : (
                    <div className="di-doc-grid">
                      {documents.map(doc => (
                        <div
                          key={doc.id}
                          className={`di-doc-card ${selectedDocument?.id === doc.id ? 'di-doc-card--selected' : ''} ${doc.status === 'processing' ? 'di-doc-card--processing' : ''}`}
                          onClick={() => doc.status === 'completed' && handleSelectDocument(doc)}
                        >
                          <div className="di-doc-header">
                            <div className={`di-doc-icon di-doc-icon--${doc.category}`}>
                              <img src={getFileIcon(doc.type)} alt="" />
                            </div>
                            <span className={`di-doc-status di-doc-status--${doc.status}`} />
                          </div>
                          <h4 className="di-doc-name">{doc.name}</h4>
                          <div className="di-doc-meta">
                            <span>{doc.type.toUpperCase()}</span>
                            <span>•</span>
                            <span>{doc.size}</span>
                          </div>
                          {doc.status === 'completed' && (
                            <div className="di-doc-stats">
                              <span>{doc.entities} entities</span>
                              <span>•</span>
                              <span>{Math.round(doc.confidence * 100)}% conf</span>
                            </div>
                          )}
                          {doc.status === 'completed' && (
                            <button className="di-doc-analyze-btn">Analyze</button>
                          )}
                        </div>
                      ))}
                    </div>
                  )}
                </div>
              </div>
            </div>
          )}

          {activeView === 'analysis' && selectedDocument && (
            <div className="di-analysis-view">
              {/* Document Header */}
              <div className="di-analysis-header">
                <div className="di-analysis-doc">
                  <div className={`di-doc-icon di-doc-icon--${selectedDocument.category}`}>
                    <img src={getFileIcon(selectedDocument.type)} alt="" />
                  </div>
                  <div className="di-analysis-doc-info">
                    <h3>{selectedDocument.name}</h3>
                    <span>{selectedDocument.type.toUpperCase()} • {selectedDocument.size} • {selectedDocument.pages} pages</span>
                  </div>
                </div>
                <div className="di-analysis-badges">
                  <span className="di-badge di-badge--entities">{selectedDocument.entities} entities</span>
                  <span className="di-badge di-badge--relations">{selectedDocument.relationships} relations</span>
                  <span className={`di-badge di-badge--confidence di-badge--${getConfidenceColor(selectedDocument.confidence)}`}>
                    {Math.round(selectedDocument.confidence * 100)}% confidence
                  </span>
                </div>
                <button className="di-change-doc" onClick={() => setActiveView('library')}>Change Document</button>
              </div>

              {/* Analysis Sub-tabs */}
              <div className="di-analysis-tabs">
                <button
                  className={`di-analysis-tab ${analysisTab === 'insights' ? 'di-analysis-tab--active' : ''}`}
                  onClick={() => setAnalysisTab('insights')}
                >
                  Key Insights
                </button>
                <button
                  className={`di-analysis-tab ${analysisTab === 'entities' ? 'di-analysis-tab--active' : ''}`}
                  onClick={() => setAnalysisTab('entities')}
                >
                  Entities ({getDocumentEntities().length})
                </button>
                <button
                  className={`di-analysis-tab ${analysisTab === 'graph' ? 'di-analysis-tab--active' : ''}`}
                  onClick={() => setAnalysisTab('graph')}
                >
                  Knowledge Graph
                </button>
                <button
                  className={`di-analysis-tab ${analysisTab === 'sources' ? 'di-analysis-tab--active' : ''}`}
                  onClick={() => setAnalysisTab('sources')}
                >
                  Sources
                </button>
              </div>

              {/* Analysis Content */}
              <div className="di-analysis-content">
                {analysisTab === 'insights' && currentInsight && (
                  <div className="di-insights-panel">
                    <div className="di-insights-summary">
                      <h4>{currentInsight.summary}</h4>
                    </div>

                    <div className="di-key-facts">
                      <h5>Key Facts</h5>
                      <ul>
                        {currentInsight.keyFacts.map((kf, idx) => (
                          <li key={idx} className="di-fact-item">
                            <span className="di-fact-text">{kf.fact}</span>
                            <div className="di-fact-meta">
                              <span className={`di-fact-confidence di-fact-confidence--${getConfidenceColor(kf.confidence)}`}>
                                {Math.round(kf.confidence * 100)}%
                              </span>
                              <span className="di-fact-source">{kf.source}</span>
                            </div>
                          </li>
                        ))}
                      </ul>
                    </div>

                    <div className="di-recommendations">
                      <h5>Recommendations</h5>
                      <ul>
                        {currentInsight.recommendations.map((rec, idx) => (
                          <li key={idx}>{rec}</li>
                        ))}
                      </ul>
                    </div>
                  </div>
                )}

                {analysisTab === 'entities' && (
                  <div className="di-entities-panel">
                    <div className="di-entity-grid">
                      {getDocumentEntities().map(entity => (
                        <div key={entity.id} className={`di-entity-card di-entity-card--${entity.type}`}>
                          <div className="di-entity-header">
                            <span className="di-entity-type">{entity.type}</span>
                            <span className={`di-entity-confidence di-entity-confidence--${getConfidenceColor(entity.confidence)}`}>
                              {Math.round(entity.confidence * 100)}%
                            </span>
                          </div>
                          <div className="di-entity-name">{entity.name}</div>
                          {entity.value && <div className="di-entity-value">{entity.value}</div>}
                        </div>
                      ))}
                    </div>
                  </div>
                )}

                {analysisTab === 'graph' && (
                  <div className="di-graph-panel">
                    {getDocumentGraph() ? (
                      <div className="di-graph-container">
                        <div className="di-graph-header">
                          <h5>Knowledge Graph</h5>
                          <span>{getDocumentGraph().nodes.length} nodes • {getDocumentGraph().edges.length} edges</span>
                        </div>
                        <div className="di-graph-canvas">
                          <svg viewBox="0 0 400 350" className="di-graph-svg">
                            {/* Edges */}
                            {getDocumentGraph().edges.map((edge, idx) => {
                              const source = getDocumentGraph().nodes.find(n => n.id === edge.source);
                              const target = getDocumentGraph().nodes.find(n => n.id === edge.target);
                              if (!source || !target) return null;
                              return (
                                <g key={idx}>
                                  <line
                                    x1={source.x}
                                    y1={source.y}
                                    x2={target.x}
                                    y2={target.y}
                                    className="di-graph-edge"
                                  />
                                  <text
                                    x={(source.x + target.x) / 2}
                                    y={(source.y + target.y) / 2 - 5}
                                    className="di-graph-edge-label"
                                  >
                                    {edge.label}
                                  </text>
                                </g>
                              );
                            })}
                            {/* Nodes */}
                            {getDocumentGraph().nodes.map(node => (
                              <g key={node.id}>
                                <circle
                                  cx={node.x}
                                  cy={node.y}
                                  r={25}
                                  className={`di-graph-node di-graph-node--${node.type}`}
                                />
                                <text x={node.x} y={node.y + 4} className="di-graph-node-label">
                                  {node.label}
                                </text>
                              </g>
                            ))}
                          </svg>
                        </div>
                        <div className="di-graph-legend">
                          <span><span className="di-legend-dot di-legend-dot--metric"></span> Metric</span>
                          <span><span className="di-legend-dot di-legend-dot--value"></span> Value</span>
                          <span><span className="di-legend-dot di-legend-dot--period"></span> Period</span>
                          <span><span className="di-legend-dot di-legend-dot--segment"></span> Segment</span>
                        </div>
                      </div>
                    ) : (
                      <EmptyState
                        iconType="document"
                        title="No graph available"
                        description="Knowledge graph will be generated after document processing."
                      />
                    )}
                  </div>
                )}

                {analysisTab === 'sources' && currentInsight && (
                  <div className="di-sources-panel">
                    <h5>Source References</h5>
                    <div className="di-sources-list">
                      {currentInsight.sources.map((source, idx) => (
                        <div key={idx} className="di-source-card">
                          <div className="di-source-header">
                            <span className="di-source-page">Page {source.page}</span>
                            <span className={`di-source-relevance di-source-relevance--${getConfidenceColor(source.relevance)}`}>
                              {Math.round(source.relevance * 100)}% relevant
                            </span>
                          </div>
                          <p className="di-source-text">"{source.text}"</p>
                        </div>
                      ))}
                    </div>
                  </div>
                )}
              </div>
            </div>
          )}

          {activeView === 'chat' && selectedDocument && (
            <div className="di-chat-view">
              {/* Chat Header */}
              <div className="di-chat-header">
                <div className="di-chat-doc">
                  <div className={`di-doc-icon di-doc-icon--${selectedDocument.category}`}>
                    <img src={getFileIcon(selectedDocument.type)} alt="" />
                  </div>
                  <span>Asking about: <strong>{selectedDocument.name}</strong></span>
                </div>
                <button className="di-change-doc" onClick={() => setActiveView('library')}>Change</button>
              </div>

              {/* Chat Messages */}
              <div className="di-chat-messages">
                {conversationHistory.length === 0 && (
                  <div className="di-chat-welcome">
                    <h4>Ask anything about this document</h4>
                    <p>I'll analyze the content and provide answers with source references.</p>
                    <div className="di-suggested-prompts">
                      {SUGGESTED_PROMPTS.slice(0, 4).map((prompt, idx) => (
                        <button key={idx} onClick={() => setInputPrompt(prompt)}>
                          {prompt}
                        </button>
                      ))}
                    </div>
                  </div>
                )}

                {conversationHistory.map((msg, idx) => (
                  <div key={idx} className={`di-chat-message di-chat-message--${msg.role}`}>
                    <div className="di-message-content">
                      {msg.content.split('\n').map((line, i) => (
                        <p key={i}>{line}</p>
                      ))}
                    </div>
                    {msg.sources && msg.sources.length > 0 && (
                      <div className="di-message-sources">
                        <span className="di-sources-label">Sources:</span>
                        {msg.sources.map((s, i) => (
                          <span key={i} className="di-source-chip">Page {s.page}</span>
                        ))}
                      </div>
                    )}
                    {msg.confidence && (
                      <span className="di-message-confidence">{Math.round(msg.confidence * 100)}% confidence</span>
                    )}
                  </div>
                ))}

                {isLoading && (
                  <div className="di-chat-message di-chat-message--assistant di-chat-message--loading">
                    <div className="di-loading-dots">
                      <span></span><span></span><span></span>
                    </div>
                  </div>
                )}
              </div>

              {/* Chat Input */}
              <div className="di-chat-input">
                <Textarea
                  placeholder="Ask a question about this document..."
                  value={inputPrompt}
                  onChange={(e) => setInputPrompt(e.target.value)}
                  rows={2}
                  onKeyDown={(e) => {
                    if (e.key === 'Enter' && !e.shiftKey) {
                      e.preventDefault();
                      handleAskQuestion();
                    }
                  }}
                />
                <button
                  className="di-send-btn"
                  onClick={handleAskQuestion}
                  disabled={isLoading || !inputPrompt.trim()}
                >
                  {isLoading ? '...' : 'Ask'}
                </button>
              </div>
            </div>
          )}
        </div>
      </ProjectGate>
    </div>
  );
}

export default DataInsights;
