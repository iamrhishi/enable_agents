import React from 'react';
import ReactMarkdown from 'react-markdown';

/**
 * MessageContent — renders a single chat message body.
 * Supports 'markdown' (default) and 'html' formats.
 */
function MessageContent({ message }) {
  if (message.format === 'html') {
    return (
      <div
        className="message-text"
        dangerouslySetInnerHTML={{ __html: message.text }}
      />
    );
  }
  return (
    <div className="message-text">
      <ReactMarkdown>{message.text}</ReactMarkdown>
    </div>
  );
}

export default MessageContent;
