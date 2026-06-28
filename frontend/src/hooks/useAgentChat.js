import { useState, useRef, useEffect, useCallback } from 'react';
import { API_CONFIG } from '../config/apiConfig';

/**
 * useAgentChat — shared state and helpers for agent chat pages.
 *
 * @param {string} welcomeText   - First message shown when the agent loads.
 * @param {string} folderName    - Server-side folder used by saveJSONToFile.
 * @param {string} agentId       - Unique ID for this agent (used for localStorage key).
 *
 * Returns:
 *   messages, inputMessage, setInputMessage, isLoading,
 *   messagesEndRef, addMessage, clearChat, checkExistingFile, saveJSONToFile
 */
export function useAgentChat(welcomeText, folderName, agentId = folderName) {
  const storageKey = `enableAgents_chat_${agentId}`;
  const [showClearConfirm, setShowClearConfirm] = useState(false);

  // Initialize messages from localStorage or with welcome message
  const [messages, setMessages] = useState(() => {
    try {
      const saved = localStorage.getItem(storageKey);
      if (saved) {
        const parsed = JSON.parse(saved);
        if (Array.isArray(parsed) && parsed.length > 0) {
          return parsed;
        }
      }
    } catch (e) {
      console.error('Error loading chat history:', e);
    }
    // Default welcome message
    return [{
      id: 1,
      text: welcomeText,
      sender: 'agent',
      timestamp: new Date().toLocaleTimeString(),
      format: 'markdown',
    }];
  });

  const [inputMessage, setInputMessage] = useState('');
  const [isLoading, setIsLoading] = useState(false);
  const messagesEndRef = useRef(null);

  // Check if there's existing chat history on mount
  useEffect(() => {
    try {
      const saved = localStorage.getItem(storageKey);
      if (saved) {
        const parsed = JSON.parse(saved);
        // If there's more than just the welcome message, ask to continue or clear
        if (Array.isArray(parsed) && parsed.length > 1) {
          setShowClearConfirm(true);
        }
      }
    } catch (e) {
      // Ignore errors
    }
  }, [storageKey]);

  // Save messages to localStorage whenever they change
  useEffect(() => {
    try {
      localStorage.setItem(storageKey, JSON.stringify(messages));
    } catch (e) {
      console.error('Error saving chat history:', e);
    }
  }, [messages, storageKey]);

  // Auto-scroll to the latest message
  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, [messages]);

  const addMessage = useCallback((text, sender, data = null, format = 'markdown') => {
    setMessages((prev) => [
      ...prev,
      {
        id: Date.now(),
        text,
        sender,
        timestamp: new Date().toLocaleTimeString(),
        data,
        format,
      },
    ]);
  }, []);

  const clearChat = useCallback(() => {
    setMessages([{
      id: 1,
      text: welcomeText,
      sender: 'agent',
      timestamp: new Date().toLocaleTimeString(),
      format: 'markdown',
    }]);
    setShowClearConfirm(false);
  }, [welcomeText]);

  const continueChat = useCallback(() => {
    setShowClearConfirm(false);
  }, []);

  const checkExistingFile = async (fileName, newFileSize) => {
    try {
      const response = await fetch(`${API_CONFIG.API_URL}/check_existing_file`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ file_name: fileName, new_file_size: newFileSize }),
      });
      return await response.json();
    } catch (error) {
      console.error('Error checking existing file:', error);
      return { exists: false, should_skip: false };
    }
  };

  const saveJSONToFile = async (jsonData, originalFileName) => {
    try {
      const jsonFileName = originalFileName.replace(/\.(csv|xlsx|xls)$/i, '.json');
      const response = await fetch(`${API_CONFIG.API_URL}/save_json_file`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ data: jsonData, file_name: jsonFileName, folder_name: folderName }),
      });
      return await response.json();
    } catch (error) {
      console.error('Error saving JSON file:', error);
      return null;
    }
  };

  return {
    messages,
    inputMessage,
    setInputMessage,
    isLoading,
    setIsLoading,
    messagesEndRef,
    addMessage,
    clearChat,
    continueChat,
    showClearConfirm,
    checkExistingFile,
    saveJSONToFile,
  };
}
