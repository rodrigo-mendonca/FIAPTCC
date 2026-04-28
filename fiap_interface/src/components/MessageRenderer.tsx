import React, { useState, useEffect, useRef } from 'react';
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

const MessageRenderer: React.FC<MessageRendererProps> = ({
  content,
  isStreaming = false,
  darkMode = false,
  isTyping = false,
}) => {
  const [showCode, setShowCode] = useState(false);
  const [iframeHeight, setIframeHeight] = useState(300); // altura inicial razoável
  const iframeRef = useRef<HTMLIFrameElement>(null);

  // Função para ajustar automaticamente a altura do iframe
  const adjustIframeHeight = () => {
    if (!iframeRef.current) return;

    try {
      const iframe = iframeRef.current;
      const doc = iframe.contentDocument || iframe.contentWindow?.document;

      if (doc && doc.body) {
        const newHeight = doc.body.scrollHeight + 40; // +40px de margem de segurança
        setIframeHeight(Math.max(newHeight, 200));
      }
    } catch (e) {
      console.warn('Não foi possível ajustar altura do iframe:', e);
    }
  };

  // ==================== DETECÇÃO MELHORADA DE HTML ====================
    const isRealHtml = (text: string): boolean => {
      if (!text || typeof text !== 'string') return false;
      const trimmed = text.replace(/```html/g, '').trim();

      // Detecta conteúdo que claramente é HTML (começa com tag ou contém tags comuns)
      return (
        /^<[!a-z]/i.test(trimmed) ||                    // começa com <tag ou <!DOCTYPE
        /<\/?[a-z][\s\S]*>/i.test(trimmed) ||           // contém qualquer tag
        trimmed.includes('<canvas') ||
        trimmed.includes('<div') ||
        trimmed.includes('<table') ||
        trimmed.includes('<script')
      ) && !trimmed.startsWith('```html');
    };


  // Atualiza o iframe quando o conteúdo HTML muda
  useEffect(() => {
    const isHtml = isRealHtml(content);

    if (isHtml && !isStreaming && !showCode && iframeRef.current) {
      const iframe = iframeRef.current;
      const doc = iframe.contentDocument || iframe.contentWindow?.document;

      if (doc) {
        doc.open();
        doc.write(`
          <!DOCTYPE html>
          <html>
          <head>
            <meta charset="UTF-8">
            <style>
              body {
                margin: 0;
                padding: 16px;
                font-family: system-ui, -apple-system, sans-serif;
                line-height: 1.6;
                color: ${darkMode ? '#e2e8f0' : '#1f2937'};
                background: ${darkMode ? '#1e293b' : '#ffffff'};
              }
              canvas { max-width: 100% !important; height: auto !important; display: block; margin: 20px 0; }
              table { width: 100%; }
            </style>
            <script src="https://cdn.jsdelivr.net/npm/chart.js@4.4.1/dist/chart.umd.min.js"></script>
          </head>
          <body>
            <div class="message-renderer ${darkMode ? 'dark-mode' : ''}">
              ${content.replace(/```html/g, '').trim()}
            </div>

            <script>
              // Ajusta altura automaticamente após carregar gráficos
              setTimeout(() => {
                window.parent.postMessage({ type: 'resize', height: document.body.scrollHeight + 40 }, '*');
              }, 300);
            </script>
          </body>
          </html>
        `);
        doc.close();
      }
    }
  }, [content, isStreaming, showCode, darkMode]);

  // Escuta mensagens do iframe (para ajuste de altura mais preciso)
  useEffect(() => {
    const handleMessage = (event: MessageEvent) => {
      if (event.data && event.data.type === 'resize') {
        setIframeHeight(event.data.height);
      }
    };

    window.addEventListener('message', handleMessage);
    return () => window.removeEventListener('message', handleMessage);
  }, []);

  if (isTyping && !content) {
    return <div className={`message-renderer typing ${darkMode ? 'dark-mode' : ''}`}>...</div>;
  }

  
  const isHtml = isRealHtml(content);

  if (isHtml && !isStreaming) {
    return (
      <div className={`message-renderer ${darkMode ? 'dark-mode' : ''}`}>
        {showCode ? (
          /* ... seu bloco de código HTML igual ... */
          <div className="html-code-container">
            <div className="code-header">
              <span className="code-language">HTML</span>
              <button className="copy-button" onClick={() => navigator.clipboard.writeText(content)}>
                Copiar
              </button>
            </div>
            <SyntaxHighlighter style={tomorrow} language="html" PreTag="div" wrapLines>
              {content}
            </SyntaxHighlighter>
            <button className="toggle-code-button" onClick={() => setShowCode(false)}>
              Ver renderizado
            </button>
          </div>
        ) : (
          <div>
            <iframe
              ref={iframeRef}
              title="rendered-html"
              sandbox="allow-scripts allow-same-origin"
              style={{
                width: '100%',           // ← Garante 100% da largura
                maxWidth: '100%',        // ← Importante para evitar overflow
                height: `${iframeHeight}px`,
                borderRadius: '8px',
                border: '0px solid #e2e8f0',
                background: darkMode ? '#1e293b' : '#ffffff',
                display: 'block',        // ← Evita espaços extras
                boxSizing: 'border-box',
              }}
              onLoad={adjustIframeHeight}
            />
            <button
              className="toggle-code-button"
              onClick={() => setShowCode(true)}
              style={{ marginTop: '10px' }}
            >
              Ver código HTML
            </button>
          </div>
        )}
      </div>
    );
  }

  // Modo Markdown normal (mantido igual)
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
                    <span className="code-language">{language.toUpperCase()}</span>
                    <button
                      className="copy-button"
                      onClick={() => navigator.clipboard.writeText(String(children).replace(/\n$/, ''))}
                    >
                      Copiar
                    </button>
                  </div>
                  <SyntaxHighlighter style={tomorrow} language={language} PreTag="div" {...props}>
                    {String(children).replace(/\n$/, '')}
                  </SyntaxHighlighter>
                </div>
              );
            }
            return <code className="inline-code" {...props}>{children}</code>;
          },
          // ... (seus outros componentes p, h1, h2, etc. permanecem iguais)
          p: ({ children }) => <p className="paragraph">{children}</p>,
          h1: ({ children }) => <h1 className="heading h1">{children}</h1>,
          h2: ({ children }) => <h2 className="heading h2">{children}</h2>,
          h3: ({ children }) => <h3 className="heading h3">{children}</h3>,
          strong: ({ children }) => <strong className="bold">{children}</strong>,
          em: ({ children }) => <em className="italic">{children}</em>,
          ul: ({ children }) => <ul className="list unordered">{children}</ul>,
          ol: ({ children }) => <ol className="list ordered">{children}</ol>,
          li: ({ children }) => <li className="list-item">{children}</li>,
          blockquote: ({ children }) => <blockquote className="blockquote">{children}</blockquote>,
          a: ({ href, children }) => (
            <a href={href} className="link" target="_blank" rel="noopener noreferrer">
              {children}
            </a>
          ),
        }}
      >
        {content}
      </ReactMarkdown>

      {isStreaming && <span className="cursor">▋</span>}
    </div>
  );
};

export default MessageRenderer;