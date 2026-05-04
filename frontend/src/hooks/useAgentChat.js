import { useState, useRef, useEffect } from 'react';
import { API_CONFIG } from '../config/apiConfig';

/**
 * useAgentChat — shared state and helpers for agent chat pages.
 *
 * @param {string} welcomeText   - First message shown when the agent loads.
 * @param {string} folderName    - Server-side folder used by saveJSONToFile.
 *
 * Returns:
 *   messages, inputMessage, setInputMessage, isLoading,
 *   messagesEndRef, addMessage, checkExistingFile, saveJSONToFile
 */
export function useAgentChat(welcomeText, folderName) {
  const [messages, setMessages] = useState([
    {
      id: 1,
      text: welcomeText,
      sender: 'agent',
      timestamp: new Date().toLocaleTimeString(),
      format: 'markdown',
    },
  ]);
  const [inputMessage, setInputMessage] = useState('');
  const [isLoading, setIsLoading] = useState(false);
  const messagesEndRef = useRef(null);

  // Auto-scroll to the latest message
  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, [messages]);

  const addMessage = (text, sender, data = null, format = 'markdown') => {
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
  };

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
    checkExistingFile,
    saveJSONToFile,
  };
}
