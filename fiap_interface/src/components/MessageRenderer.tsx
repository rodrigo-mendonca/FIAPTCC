import React from 'react';
import ReactMarkdown from 'react-markdown';
import remarkGfm from 'remark-gfm';
import { Prism as SyntaxHighlighter } from 'react-syntax-highlighter';
import { tomorrow } from 'react-syntax-highlighter/dist/esm/styles/prism';
import './MessageRenderer.css';

interface MessageRendererProps {
  content: string;
  isStreaming?: boolean;
  darkMode?: boolean;
  isTyping?: boolean;
}

const MessageRenderer: React.FC<MessageRendererProps> = ({ content, isStreaming = false, darkMode = false, isTyping = false }) => {
  // Se está digitando e não há conteúdo ainda, mostra indicador de digitação
  if (isTyping && !content) {
    return (
      <div className={`message-renderer typing ${darkMode ? 'dark-mode' : ''}`}>
        <span className="typing-indicator">
          <span className="dot"></span>
          <span className="dot"></span>
          <span className="dot"></span>
        </span>
      </div>
    );
  }

  return (
    <div className={`message-renderer ${isStreaming ? 'streaming' : ''} ${darkMode ? 'dark-mode' : ''}`}>
      <ReactMarkdown
        remarkPlugins={[remarkGfm]}
        components={{
          code({ node, inline, className, children, ...props }: any) {
            const match = /language-(\w+)/.exec(className || '');
            const language = match ? match[1] : '';
            
            if (!inline && language) {
              return (
                <div className="code-block">
                  <div className="code-header">
                    <span className="code-language">{language}</span>
                    <button 
                      className="copy-button"
                      onClick={() => navigator.clipboard.writeText(String(children).replace(/\n$/, ''))}
                    >
                      Copiar
                    </button>
                  </div>
                  <SyntaxHighlighter
                    style={tomorrow}
                    language={language}
                    PreTag="div"
                    {...props}
                  >
                    {String(children).replace(/\n$/, '')}
                  </SyntaxHighlighter>
                </div>
              );
            }
            
            return (
              <code className="inline-code" {...props}>
                {children}
              </code>
            );
          },
          p({ children }) {
            return <p className="paragraph">{children}</p>;
          },
          h1({ children }) {
            return <h1 className="heading h1">{children}</h1>;
          },
          h2({ children }) {
            return <h2 className="heading h2">{children}</h2>;
          },
          h3({ children }) {
            return <h3 className="heading h3">{children}</h3>;
          },
          strong({ children }) {
            return <strong className="bold">{children}</strong>;
          },
          em({ children }) {
            return <em className="italic">{children}</em>;
          },
          ul({ children }) {
            return <ul className="list unordered">{children}</ul>;
          },
          ol({ children }) {
            return <ol className="list ordered">{children}</ol>;
          },
          li({ children }) {
            return <li className="list-item">{children}</li>;
          },
          blockquote({ children }) {
            return <blockquote className="blockquote">{children}</blockquote>;
          },
          a({ href, children }) {
            return (
              <a 
                href={href} 
                className="link" 
                target="_blank" 
                rel="noopener noreferrer"
              >
                {children}
              </a>
            );
          },
          table({ children }) {
            return <table className="table">{children}</table>;
          },
          thead({ children }) {
            return <thead className="table-header">{children}</thead>;
          },
          tbody({ children }) {
            return <tbody className="table-body">{children}</tbody>;
          },
          tr({ children }) {
            return <tr className="table-row">{children}</tr>;
          },
          th({ children }) {
            return <th className="table-cell header">{children}</th>;
          },
          td({ children }) {
            return <td className="table-cell">{children}</td>;
          }
        }}
      >
        {content}
      </ReactMarkdown>
      {isStreaming && <span className="cursor">▋</span>}
    </div>
  );
};

export default MessageRenderer;
