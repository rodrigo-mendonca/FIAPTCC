import React, { useRef, useEffect, useMemo, ReactNode, ReactElement } from 'react';
import ReactMarkdown from 'react-markdown';
import remarkGfm from 'remark-gfm';
import embed from 'vega-embed';
import { Wrench, Loader2 } from 'lucide-react'; // Opcional: use ícones de sua preferência
import { useTheme } from '@mui/material/styles';


// ── Componente de Badge de Status ──────────────────────────────────────────

const ToolStatusBadge: React.FC<{ type: 'executing' | 'completed' }> = ({ type }) => {
  const isExecuting = type === 'executing';
  
  return (
    <div style={{
      display: 'inline-flex',
      alignItems: 'center',
      gap: '10px',
      padding: '4px 16px',
      margin: '8px 0',
      borderRadius: '8px',
      fontSize: '0.9rem',
      fontWeight: 500,
      backgroundColor: isExecuting ? '#f0f7ff' : '#ecfdf5',
      border: `1px solid ${isExecuting ? '#cfe2ff' : '#d1fae5'}`,
      color: isExecuting ? '#0056b3' : '#065f46',
      transition: 'all 0.4s ease',
      width: 'fit-content', // Ajusta ao texto, mas você pode usar 200 se preferir fixo
      animation: 'fadeIn 0.3s ease-out'
    }}>
      {isExecuting ? (
        <Loader2 size={16} className="animate-spin" />
      ) : (
        <Wrench size={16} style={{ color: '#10b981' }} />
      )}
      <span>{isExecuting ? 'Executando consulta' : 'Consulta concluída'}</span>
    </div>
  );
};

const flattenChildren = (children: ReactNode): string => {
  return React.Children.toArray(children).reduce((text: string, child: ReactNode) => {
    // 1. Se for string ou número, anexa diretamente
    if (typeof child === 'string' || typeof child === 'number') {
      return text + child;
    }

    // 2. Verifica se é um elemento React válido para acessar as props
    if (React.isValidElement(child)) {
      const element = child as ReactElement<any>;
      if (element.props && element.props.children) {
        return text + flattenChildren(element.props.children);
      }
    }

    return text;
  }, '');
};

export const MarkdownRenderer: React.FC<{ content: string; isStreaming: boolean }> = ({ content, isStreaming }) => {
  const theme = useTheme();
const isDark = theme.palette.mode === 'dark';

const markdownStyles = `
  .markdown-container { 
    font-family: ${theme.typography.fontFamily}; 
    line-height: 1.6; 
    color: ${theme.palette.text.primary};
  }

  .markdown-container table { 
    border-collapse: collapse; 
    width: 100%; 
    margin: 16px 0; 
    display: block; 
    overflow-x: auto; 
  }

  .markdown-container th, .markdown-container td { 
    border: 1px solid ${theme.palette.divider}; 
    padding: 8px 12px; 
  }

  .markdown-container th { 
    background-color: ${isDark ? theme.palette.grey[900] : theme.palette.grey[100]}; 
  }
  
  .typing-cursor {
    display: inline-block;
    width: 6px;
    height: 15px;
    background-color: ${theme.palette.primary.main};
    margin-left: 4px;
    animation: blink 0.8s infinite;
    vertical-align: middle;
  }

  @keyframes spin { from { transform: rotate(0deg); } to { transform: rotate(360deg); } }
  .animate-spin { animation: spin 1s linear infinite; display: inline-block; }

  .loading-bar-progress {
    width: 100%;
    height: 100%;
    background: linear-gradient(90deg, transparent, ${theme.palette.divider}, transparent);
    animation: loading-progress 1.5s infinite;
  }
`;

  // 1. Processamento da string para garantir que o Markdown reconheça os blocos
  const processedContent = useMemo(() => {
    if (!content) return '';

    let text = content
      .replace(/\\n/g, '\n')
      .replace(/([^\n])(```)/g, '$1\n\n$2');

    // LÓGICA GLOBAL: Se o texto final já contém o "Concluído", 
    // removemos versões anteriores de "Executando" para limpar o histórico da mensagem atual.
    if (text.includes('[Consulta concluída]')) {
      text = text.replace(/\[Executando consulta\]/g, '');
    }

    return text;
  }, [content]);

  return (
    <div className="markdown-container">
      <style>{markdownStyles}</style>
      <ReactMarkdown
        remarkPlugins={[remarkGfm]}
        components={{
          // Customização do parágrafo para os Status Badges (usando a lógica anterior)
          p({ children }) {
            const text = flattenChildren(children);
            const parts = text.split(/(\[Executando consulta\]|\[Consulta concluída\])/g);
            
            if (parts.length > 1) {
              return (
                <div style={{ 
                  display: 'flex', 
                  flexDirection: 'column', 
                  gap: '4px',           // Gap reduzido
                  marginBottom: '16px',
                  border: 'none',       // Garante que não haja borda no container
                  outline: 'none' 
                }}>
                  {parts.map((part, i) => {
                    if (part === '[Executando consulta]') return <ToolStatusBadge key={i} type="executing" />;
                    if (part === '[Consulta concluída]') return <ToolStatusBadge key={i} type="completed" />;
                    
                    // Remove espaços em branco que criam "linhas vazias" entre os badges
                    const cleanText = part.trim();
                    return cleanText ? <span key={i} style={{ fontSize: '1rem' }}>{cleanText}</span> : null;
                  })}
                </div>
              );
            }
            return <p style={{ marginBottom: '16px', border: 'none' }}>{children}</p>;
          },

          // Customização do bloco de código corrigindo o erro de Tipagem
          code(props) {
            // No react-markdown v9, 'inline' não existe mais formalmente nas props.
            // O padrão agora é: se tem classe 'language-', é bloco. Se não tem, é inline.
            const { children, className, node, ...rest } = props;
            
            const match = /language-(\w+)/.exec(className || '');
            const lang = match ? match[1] : '';
            const codeValue = String(children).replace(/\n$/, '');

            // Se não houver className, o react-markdown costuma tratar como inline code
            const isInline = !className;

            // Lógica de Detecção Vega-Lite (Blindada)
            const isVega = lang === 'vega-lite' || 
                           (!isInline && codeValue.includes('vega.github.io/schema/vega-lite'));

            if (!isInline && isVega) {
              return <VegaLiteBlock spec={codeValue} />;
            }

            if (isInline) {
              return (
                <code 
                  style={{ padding: '2px 4px', borderRadius: '4px' }} 
                  {...rest}
                >
                  {children}
                </code>
              );
            }

            return (
              <pre style={{ padding: '1rem', borderRadius: '8px', overflowX: 'auto' }}>
                <code className={className} {...rest}>
                  {children}
                </code>
              </pre>
            );
          }
        }}
      >
        {processedContent}
      </ReactMarkdown>
      {isStreaming && <span className="typing-cursor" />}
    </div>
  );
};

// Componente VegaLite (igual ao anterior)
const VegaLiteBlock: React.FC<{ spec: string }> = ({ spec }) => {
  const containerRef = useRef<HTMLDivElement>(null);
  const theme = useTheme();
  const isDark = theme.palette.mode === 'dark';
  const [isValid, setIsValid] = React.useState(false);

  // 1. Estabiliza a especificação. Só muda quando o JSON está completo.
  const stableSpec = useMemo(() => {
    const clean = spec.trim();
    if (!clean.endsWith('}')) return null; // JSON incompleto, não processa

    try {
      const parsed = JSON.parse(clean);
      return {
        ...parsed,
        width: 'container',
        height: 350, // Altura aumentada
        autosize: { type: 'fit', contains: 'padding' }
      };
    } catch (e) {
      return null; // JSON malformado
    }
  }, [spec]);

  // 2. Efeito de renderização vinculado apenas à especificação estável
  useEffect(() => {
    if (!containerRef.current || !stableSpec) {
      setIsValid(false);
      return;
    }

    const render = async () => {
      try {
        // Limpa renderizações anteriores antes de novo embed
        containerRef.current!.innerHTML = '';
        
        await embed(containerRef.current!, stableSpec, {
          actions: false,
          theme: isDark ? 'dark' : 'quartz',
          width: 'container' as any,
          renderer: 'svg',
          defaultStyle: true
        });
        
        setIsValid(true);
      } catch (e) {
        setIsValid(false);
      }
    };

    render();
  }, [stableSpec, isDark]);


  const renderMemo = useMemo(() => (<div style={{ 
      margin: '1.5rem 0', 
      minHeight: '400px', 
      position: 'relative', 
      width: '100%',
      display: 'block'
    }}>
      {/* Container do Gráfico: Visibility mantém o espaço para o cálculo de largura */}
      <div 
        ref={containerRef} 
        style={{ 
          visibility: isValid ? 'visible' : 'hidden',
          opacity: isValid ? 1 : 0,
          width: '100%',
          minHeight: '400px',
          transition: 'opacity 0.4s ease'
        }} 
      />

      {/* Loader: Respeitando o Modo Escuro do MUI */}
      {!isValid && (
        <div style={{
          position: 'absolute',
          top: 0, left: 0, right: 0, bottom: 0,
          display: 'flex', flexDirection: 'column',
          alignItems: 'center', justifyContent: 'center',
          backgroundColor: theme.palette.background.paper,
          border: `2px dashed ${theme.palette.divider}`,
          borderRadius: '12px', 
          color: theme.palette.text.secondary, 
          gap: '12px',
          zIndex: 1
        }}>
          <span style={{ fontSize: '0.9rem', fontWeight: 600 }}>Montando gráfico...</span>
          <div style={{ 
            width: '60%', 
            height: '6px', 
            backgroundColor: theme.palette.divider, 
            borderRadius: '3px', 
            overflow: 'hidden' 
          }}/>
        </div>
      )}
    </div>), [isValid, theme.palette.background.paper, theme.palette.divider, theme.palette.text.secondary]);

  return (
    renderMemo
  );
};

export default MarkdownRenderer;