import React, { useState, useRef, useEffect } from 'react';
import {
  Box,
  Card,
  CardContent,
  Typography,
  TextField,
  Button,
  Avatar,
  useTheme,
  Chip,
  IconButton,
} from '@mui/material';
import {
  Help as HelpIcon,
  Chat as ChatIcon,
  Send as SendIcon,
  SmartToy as BotIcon,
  Person as PersonIcon,
  DeleteSweep as ClearChatIcon,
  Refresh as RefreshIcon,
  Code as CodeIcon,
  ChevronLeft as ChevronLeftIcon,
  ChevronRight as ChevronRightIcon,
} from '@mui/icons-material';
import { useCollection } from '../contexts/CollectionContext';
import CollectionSelector from './CollectionSelector';
import MarkdownRenderer from './MarkdownRenderer';

// ── Tipos ─────────────────────────────────────────────────────────────────────

interface Message {
  id: string;
  content: string;
  sender: 'user' | 'bot';
  timestamp: Date;
}

export interface ChatConfig {
  title: string;
  icon: React.ReactElement;
  streamEndpoint: string;
  placeholder: string;
  description: string;
  emptyStateMessage: string;
  suggestions: string[];
  suggestionsTitle: string;
  suggestionsDescription: string;
  isSQL?: boolean;
  isAluno?: boolean;
}

interface ConfigurableChatProps {
  config: ChatConfig;
  darkMode?: boolean;
  showCollectionSelector?: boolean;
  containerHeight?: string | number;
  hideSuggestions?: boolean;
}

// ── Componente principal de chat ──────────────────────────────────────────────

export const ConfigurableChat: React.FC<ConfigurableChatProps> = ({
  config,
  darkMode = false,
  showCollectionSelector = false,
  containerHeight,
  hideSuggestions = false,
}) => {
  const [messages, setMessages] = useState<Message[]>([]);
  const [inputMessage, setInputMessage] = useState('');
  const [streamingMessageId, setStreamingMessageId] = useState<string | null>(null);
  const [isTyping, setIsTyping] = useState(false);
  const [isClearing, setIsClearing] = useState(false);
  const messagesEndRef = useRef<HTMLDivElement>(null);
  const theme = useTheme();
  const { selectedCollection } = useCollection();
  const [alunoType, setAlunoType] = useState<string>('business_rules');
  const [registering, setRegistering] = useState(false);
  const [suggestionsOpen, setSuggestionsOpen] = useState(true);

  const scrollToBottom = () => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  };

  useEffect(() => {
    scrollToBottom();
  }, [messages]);

  // Limpa o chat quando a coleção muda
  useEffect(() => {
    setMessages([]);
    setInputMessage('');
  }, [selectedCollection]);


  useEffect(() => {
    if (config.isAluno && messages.length === 0) {
      setMessages([{
        id: 'aluno_init',
        content: 'Vou aprender com base no que você informar. Antes de prosseguir, verifiquei se você está falando de dois assuntos ao mesmo tempo — se estiver, por favor escolha somente 1 tema. Escolha um tipo: Regra de negócio, Base de dados ou Serviço. Se precisar, pedirei mais informações. Quando tudo estiver ok, por favor revise e confirme se devo registrar.',
        sender: 'bot',
        timestamp: new Date(),
      }]);
    }
  }, [config.isAluno, messages.length]);

  const lastBotMessage = [...messages]
    .reverse()
    .find((m) => m.sender === 'bot' && m.content?.trim().length > 0);

  const readyToRegister = Boolean(
    lastBotMessage &&
    /pronto para registrar|revisar e confirmar|confirme se devo registrar|revise e confirme|posso registrar|devo registrar|informações suficientes|quando estiver tudo certo|já posso registrar|pronto para salvar/i.test(
      lastBotMessage.content
    )
  );

  const clearChat = async () => {
    setIsClearing(true);
    try { setMessages([]); } finally { setIsClearing(false); }
  };

  const sendMessage = async () => {
    if (!inputMessage.trim() || isTyping) return;

    const userMessage: Message = {
      id: Date.now().toString(),
      content: inputMessage,
      sender: 'user',
      timestamp: new Date(),
    };

    setMessages((prev) => [...prev, userMessage]);
    const currentInput = inputMessage;
    setInputMessage('');
    setIsTyping(true);

    const botMessageId = (Date.now() + 1).toString();
    setMessages((prev) => [
      ...prev,
      { id: botMessageId, content: '', sender: 'bot', timestamp: new Date() },
    ]);
    setStreamingMessageId(botMessageId);

    try {
      let messageToSend = currentInput;
      if (config.isAluno) {
        const typeLabel =
          alunoType === 'business_rules' ? 'Regra de negócio'
          : alunoType === 'database_struct' ? 'Base de dados'
          : 'Serviço';
        messageToSend = `INSTRUÇÕES AO ASSISTENTE-ALUNO: Você é um aluno que está aprendendo. O usuário está ensinando SOBRE: ${typeLabel}. Seu objetivo é extrair e construir um objeto JSON completo desse tipo com todos os campos necessários. Se faltarem informações, faça PERGUNTAS DIRETAS e ESPECÍFICAS ao usuário para obter os campos que faltam. Não invente valores. Quando você tiver todas as informações necessárias, responda primeiro a linha: "PRONTO_PARA_REGISTRAR" seguida do JSON completo. Pergunte apenas uma coisa por vez. Seja objetivo e claro. Agora segue a entrada do usuário:\n${currentInput}`;
      }

      const context = messages.slice(-10).map((msg) => ({
        role: msg.sender === 'user' ? 'user' : 'assistant',
        content: msg.content,
      }));

      const fullURL = `${API_URL}${config.streamEndpoint}?collection_name=${encodeURIComponent(selectedCollection)}`;

      const response = await fetch(fullURL, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json', Accept: 'text/event-stream', 'Cache-Control': 'no-cache' },
        body: JSON.stringify({ message: messageToSend, context }),
        mode: 'cors',
        credentials: 'same-origin',
      });

      if (!response.ok) throw new Error(`Erro na API: ${response.status}`);

      const reader = response.body?.getReader();
      if (!reader) throw new Error('Response body não disponível');

      let accumulatedContent = '';
      try {
        while (true) {
          const { done, value } = await reader.read();
          if (done) break;
          const chunk = new TextDecoder().decode(value);
          for (const line of chunk.split('\n')) {
            if (line.startsWith('data: ')) {
              const jsonStr = line.slice(6).trim();
              if (jsonStr && jsonStr !== '[DONE]') {
                try {
                  const data = JSON.parse(jsonStr);
                  if (data.content) {
                    accumulatedContent += data.content;
                    setMessages((prev) =>
                      prev.map((msg) =>
                        msg.id === botMessageId ? { ...msg, content: accumulatedContent } : msg
                      )
                    );
                  }
                } catch { /* chunk parcial */ }
              }
            }
          }
        }
      } finally {
        reader.releaseLock();
      }
    } catch (error) {
      console.error('Erro ao enviar mensagem:', error);
      setMessages((prev) =>
        prev.map((msg) =>
          msg.id === botMessageId
            ? { ...msg, content: 'Erro na comunicação com a API. Verifique se o servidor está rodando.' }
            : msg
        )
      );
    } finally {
      setIsTyping(false);
      setStreamingMessageId(null);
    }
  };

  const handleRegister = async () => {
    const lastBot = [...messages].reverse().find((m) => m.sender === 'bot' && m.content?.trim().length > 0);
    if (!lastBot) { alert('Nenhuma informação do bot encontrada para registrar.'); return; }

    const containsMultiple =
      (lastBot.content || '').split(/[\.\n]/).filter(Boolean).length > 1 &&
      /\band\b|\be\b|,/i.test(lastBot.content);
    if (containsMultiple) {
      const ok = window.confirm('Parece que você está falando sobre mais de um tema. Deseja continuar e registrar apenas este conteúdo, ou prefere escolher um único tema antes de registrar? (Clique Cancelar para escolher)');
      if (!ok) return;
    }

    let parsed = null;
    try { parsed = JSON.parse(lastBot.content); } catch {
      const ok = window.confirm('O conteúdo do bot não está em JSON válido. Deseja registrar como texto livre?');
      if (!ok) return;
    }

    setRegistering(true);
    try {
      const payload: any = { collection_name: selectedCollection, type: alunoType };
      if (parsed) payload['item'] = parsed;
      else payload['text'] = lastBot.content;

      const resp = await fetch(`${API_URL}/vectordb/add-item`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(payload),
      });
      if (!resp.ok) {
        const err = await resp.json().catch(() => null);
        throw new Error(err?.detail || 'Erro ao registrar item');
      }
      setMessages([{
        id: Date.now().toString() + '_regok',
        content: 'Ensinamento registrado com sucesso. Agora você já pode falar sobre outro assunto.',
        sender: 'bot',
        timestamp: new Date(),
      }]);
    } catch (err: any) {
      alert('Falha ao registrar item: ' + (err?.message || err));
    } finally {
      setRegistering(false);
    }
  };

  const handleKeyPress = (e: React.KeyboardEvent) => {
    if (e.key === 'Enter' && !e.shiftKey) { e.preventDefault(); sendMessage(); }
  };

  const renderBotContent = (message: Message) => {
    const isStreaming = streamingMessageId === message.id;
    if (config.isSQL && message.content) {
      return (
        <Box>
          <Typography variant="body2" fontWeight="bold" sx={{ mb: 1 }}>SQL Gerado:</Typography>
          <pre style={{ background: '#2d2d2d', color: '#ffffff', padding: '12px', borderRadius: '6px', fontSize: '0.9rem', fontFamily: 'Monaco, Consolas, monospace', whiteSpace: 'pre-wrap', overflow: 'auto', margin: 0 }}>
            {message.content}
          </pre>
        </Box>
      );
    }
    return <MarkdownRenderer content={message.content} isStreaming={isStreaming} />;
  };

  const cardHeight = containerHeight ?? 'calc(100vh - 120px)';

  return (
    <Box sx={{
      display: 'flex',
      flexDirection: 'row',
      gap: (!hideSuggestions && suggestionsOpen) ? 3 : 1,
      height: cardHeight,
      alignItems: 'stretch',
    }}>
      <Box sx={{ flex: 1, minWidth: 0, display: 'flex', flexDirection: 'column' }}>
        <Card sx={{ flex: 1, display: 'flex', flexDirection: 'column', overflow: 'hidden' }}>

          {/* ── Header ── */}
          <Box sx={{
            background: 'linear-gradient(135deg, #2E6DA4 0%, #1A3A5C 100%)',
            color: '#ffffff',
            display: 'flex',
            alignItems: 'center',
            justifyContent: 'space-between',
            px: 2,
            py: 1.5,
            borderRadius: '12px 12px 0 0',
            gap: 1,
          }}>
            {/* Left */}
            <Box sx={{ display: 'flex', alignItems: 'center', gap: 1.5, minWidth: 0 }}>
              <Box sx={{ width: 8, height: 8, borderRadius: '50%', bgcolor: '#27AE60', flexShrink: 0, boxShadow: '0 0 0 2px rgba(39,174,96,.3)' }} />
              <Box sx={{ minWidth: 0 }}>
                <Typography sx={{ fontWeight: 700, fontSize: '.95rem', lineHeight: 1.2 }}>{config.title}</Typography>
                <Typography sx={{ color: 'rgba(255,255,255,.6)', fontSize: '.72rem' }}>
                  {config.description} · {messages.length} mensagens
                </Typography>
              </Box>
            </Box>

            {/* Right */}
            <Box sx={{ display: 'flex', alignItems: 'center', gap: 0.5, flexShrink: 0 }}>
              {showCollectionSelector && <CollectionSelector compact />}
              <IconButton
                onClick={clearChat}
                disabled={isClearing || messages.length === 0}
                title="Limpar conversa"
                sx={{
                  mt:1,
                  color: messages.length > 0 ? 'rgba(255,255,255,.75)' : 'rgba(255,255,255,.3)',
                  '&:hover': { bgcolor: messages.length > 0 ? 'rgba(255,255,255,.12)' : 'transparent' },
                }}
              >
                {isClearing ? <RefreshIcon className="spinning" /> : <ClearChatIcon  />}
              </IconButton>
            </Box>
          </Box>

          {/* ── Messages ── */}
          <CardContent sx={{ p: 0, display: 'flex', flexDirection: 'column', height: '100%', flex: 1, overflow: 'hidden' }}>
            <Box sx={{ flex: 1, overflow: 'auto', p: 2, display: 'flex', flexDirection: 'column', gap: 2 }}>
              {messages.length === 0 && (
                <Box sx={{ display: 'flex', flexDirection: 'column', alignItems: 'center', justifyContent: 'center', height: '100%', textAlign: 'center', opacity: 0.7 }}>
                  <BotIcon sx={{ fontSize: 48, mb: 2, color: 'primary.main' }} />
                  <Typography variant="h6" gutterBottom>{config.description}</Typography>
                  <Typography variant="body2" color="text.secondary">{config.emptyStateMessage}</Typography>
                </Box>
              )}

              {messages.map((message) => (
                <Box key={message.id} sx={{ display: 'flex', justifyContent: message.sender === 'user' ? 'flex-end' : 'flex-start', mb: 1 }}>
                  <Box sx={{ display: 'flex', flexDirection: message.sender === 'user' ? 'row-reverse' : 'row', alignItems: 'flex-start', gap: 1, maxWidth: '95%', width: '100%' }}>
                    <Avatar sx={{ width: 32, height: 32, background: '#64748b', color: '#ffffff' }}>
                      {message.sender === 'user' ? <PersonIcon /> : <BotIcon />}
                    </Avatar>
                    <Box
                      sx={{
                        p: 2, width: '100%', borderRadius: 2,
                        background: message.sender === 'user' ? '#2E6DA4' : darkMode ? '#1e293b' : '#f8fafc',
                        color: message.sender === 'user' ? '#ffffff' : darkMode ? '#ffffff' : '#000000',
                        wordBreak: 'break-word',
                        border: message.sender === 'bot' ? `1px solid ${darkMode ? '#475569' : '#e2e8f0'}` : 'none',
                        boxShadow: '0 1px 4px rgba(0,0,0,.08)',
                      }}
                    >
                      {message.sender === 'bot' ? renderBotContent(message) : <Typography variant="body1">{message.content}</Typography>}
                      <Typography variant="caption" sx={{ display: 'block', mt: 1, opacity: 0.7, textAlign: message.sender === 'user' ? 'right' : 'left' }}>
                        {message.timestamp.toLocaleTimeString()}
                      </Typography>
                    </Box>
                  </Box>
                </Box>
              ))}
              <div ref={messagesEndRef} />
            </Box>

            {/* ── Input ── */}
            <Box sx={{ p: 2, borderTop: `1px solid ${theme.palette.divider}` }}>
              <Box sx={{ display: 'flex', gap: 1, width: '100%' }}>
                <TextField
                  fullWidth multiline maxRows={4}
                  value={inputMessage}
                  onChange={(e) => setInputMessage(e.target.value)}
                  onKeyDown={handleKeyPress}
                  placeholder={config.placeholder}
                  disabled={isTyping}
                  variant="outlined" size="small"
                  sx={{ '& .MuiOutlinedInput-root': { borderRadius: 3 } }}
                />
                <Button
                  variant="contained" onClick={sendMessage}
                  disabled={!inputMessage.trim() || isTyping}
                  sx={{ borderRadius: 3, minWidth: 48, bgcolor: '#2E6DA4', '&:hover': { bgcolor: '#1A3A5C' }, '&:disabled': { bgcolor: '#94a3b8', color: '#ffffff' } }}
                >
                  <SendIcon />
                </Button>
                {config.isAluno && readyToRegister && (
                  <Button variant="contained" color="secondary" onClick={handleRegister} disabled={registering} sx={{ borderRadius: 3, minWidth: 140 }}>
                    Registrar
                  </Button>
                )}
              </Box>
            </Box>

            {config.isAluno && (
              <Box sx={{ p: 2, borderTop: `1px solid ${theme.palette.divider}`, display: 'flex', gap: 2, alignItems: 'center' }}>
                <TextField
                  select SelectProps={{ native: true }} value={alunoType}
                  onChange={(e) => setAlunoType(e.target.value)}
                  size="small" label="Tipo" sx={{ minWidth: 220 }}
                >
                  <option value="business_rules">Regra de negócio</option>
                  <option value="database_struct">Base de dados</option>
                  <option value="system_services">Serviço</option>
                </TextField>
              </Box>
            )}
          </CardContent>
        </Card>
      </Box>

      {/* ── Suggestions panel ── */}
      {!hideSuggestions && (
        suggestionsOpen ? (
          <Box sx={{ width: 400, flexShrink: 0, display: 'flex', flexDirection: 'column' }}>
            <Card sx={{ flex: 1, overflow: 'auto' }}>
              <CardContent>
                <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', mb: 1 }}>
                  <Typography variant="h6" sx={{ fontSize: '.95rem' }}>💡 {config.suggestionsTitle}</Typography>
                  <IconButton size="small" onClick={() => setSuggestionsOpen(false)} title="Ocultar sugestões">
                    <ChevronRightIcon fontSize="small" />
                  </IconButton>
                </Box>
                <Typography variant="body2" color="text.secondary" sx={{ mb: 1.5 }}>
                  {config.suggestionsDescription}
                </Typography>
                <Box sx={{ display: 'flex', flexDirection: 'column', gap: 1 }}>
                  {config.suggestions.map((suggestion, index) => (
                    <Chip
                      key={index} label={suggestion} variant="outlined" clickable
                      onClick={() => setInputMessage(suggestion)} size="small"
                      sx={{ justifyContent: 'flex-start', height: 'auto', py: 1, '& .MuiChip-label': { whiteSpace: 'normal', textAlign: 'left' } }}
                    />
                  ))}
                </Box>
              </CardContent>
            </Card>
          </Box>
        ) : (
          <Box sx={{ display: 'flex', alignItems: 'flex-start', pt: 0.5 }}>
            <IconButton
              onClick={() => setSuggestionsOpen(true)}
              title="Mostrar sugestões"
              sx={{ bgcolor: 'background.paper', border: '1px solid', borderColor: 'divider', borderRadius: 1 }}
            >
              <ChevronLeftIcon fontSize="small" />
            </IconButton>
          </Box>
        )
      )}
    </Box>
  );
};

// ── Configurações dos chats ───────────────────────────────────────────────────

export const chatConfigs: Record<string, ChatConfig> = {
  general: {
    title: '💬 Chat',
    icon: <ChatIcon />,
    streamEndpoint: '/api/chat/general/stream',
    placeholder: 'Digite sua mensagem para conversar...',
    description: 'Dúvidas?',
    emptyStateMessage: 'Converse livremente com a IA sobre qualquer assunto.',
    suggestions: [
      'Qual é o seu nome?', 'Olá! Como você pode me ajudar?', 'Me explique sobre inteligência artificial',
      'Quais são as últimas tendências em tecnologia?', 'Como posso melhorar minha produtividade?',
      'Me conte uma curiosidade interessante', 'Como funciona o machine learning?',
      'Quais são os benefícios da automação empresarial?',
    ],
    suggestionsTitle: 'Sugestões de Conversa',
    suggestionsDescription: 'Clique nas sugestões abaixo para iniciar uma conversa:',
    isSQL: false,
  },
  sql: {
    title: '🔍 Chat SQL',
    icon: <CodeIcon />,
    streamEndpoint: '/api/chat/sql/stream',
    placeholder: 'Ex: Como buscar todos os clientes ativos?',
    description: 'Gerador de SQL',
    emptyStateMessage: 'Faça perguntas para gerar consultas SQL baseadas na estrutura do banco.',
    suggestions: [
      'Liste todos os clientes ativos', 'Busque vendas do último mês', 'Produtos com estoque baixo',
      'Clientes inadimplentes', 'Vendas por funcionário', 'Top 10 produtos mais vendidos',
    ],
    suggestionsTitle: 'Exemplos de Consultas',
    suggestionsDescription: 'Clique nas sugestões abaixo para gerar SQL:',
    isSQL: true,
  },
  help: {
    title: '❓ Dúvidas',
    icon: <HelpIcon />,
    streamEndpoint: '/api/chat/help/stream',
    placeholder: 'Digite a mensagem para iniciar uma conversa...',
    description: 'Assistente de Dúvidas',
    emptyStateMessage: 'Tire dúvidas sobre o sistema comercial e regras de negócio.',
    suggestions: [
      'Liste todas as tabelas disponíveis', 'Verifica na documentação sobre saldo devedor do cliente',
      'Cria um gráfico dos últimos 3 meses de vendas', 'Qual o maior cliente?',
      'Qual o produto mais vendido?', 'Cria uma tabela com os 10 produtos mais vendidos',
      'Como cadastrar um novo cliente?', 'Qual é o fluxo de aprovação de crédito?',
      'Como alterar preços de produtos?', 'Quais são os prazos de entrega padrão?', "Cria um gráfico de linhas com as vendas totais por dia do mês de abril"
    ],
    suggestionsTitle: 'Exemplos de Dúvidas',
    suggestionsDescription: 'Clique nas sugestões abaixo para fazer perguntas:',
    isSQL: false,
  },
};

const API_URL = process.env.REACT_APP_API_URL;

// ── ChatTabs: apenas o Chat Dúvidas ──────────────────────────────────────────

const ChatTabs: React.FC<{ darkMode?: boolean }> = ({ darkMode = false }) => (
  <ConfigurableChat
    config={chatConfigs.help}
    darkMode={darkMode}
    showCollectionSelector
  />
);

export default ChatTabs;
