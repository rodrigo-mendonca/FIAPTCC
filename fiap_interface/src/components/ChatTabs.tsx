import React, { useState, useRef, useEffect, useCallback } from 'react';
import { flushSync } from 'react-dom';
import {
  Box,
  Tabs,
  Tab,
  Paper,
  Card,
  CardContent,
  Typography,
  TextField,
  Button,
  Avatar,
  useTheme,
  alpha,
  Chip,
  IconButton
} from '@mui/material';
import {
  Code as CodeIcon,
  Help as HelpIcon,
  Chat as ChatIcon,
  Send as SendIcon,
  SmartToy as BotIcon,
  Person as PersonIcon,
  Delete as DeleteIcon,
  Refresh as RefreshIcon
} from '@mui/icons-material';
import MessageRenderer from './MessageRenderer';
import { useCollection } from '../contexts/CollectionContext';

interface Message {
  id: string;
  content: string;
  sender: 'user' | 'bot';
  timestamp: Date;
}

interface ChatConfig {
  title: string;
  icon: React.ReactElement;
  endpoint: string;
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
}

// Componente de Chat configurável
const ConfigurableChat: React.FC<ConfigurableChatProps> = ({ config, darkMode = false }) => {
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

  const scrollToBottom = () => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  };

  useEffect(() => {
    scrollToBottom();
  }, [messages]);

  // Se for Aluno, insere mensagem inicial explicativa (e verifica se usuário fala sobre dois temas)
  useEffect(() => {
    if (config.isAluno && messages.length === 0) {
      const initMsg: Message = {
        id: 'aluno_init',
        content: 'Vou aprender com base no que você informar. Antes de prosseguir, verifiquei se você está falando de dois assuntos ao mesmo tempo — se estiver, por favor escolha somente 1 tema. Escolha um tipo: Regra de negócio, Base de dados ou Serviço. Se precisar, pedirei mais informações. Quando tudo estiver ok, por favor revise e confirme se devo registrar.',
        sender: 'bot',
        timestamp: new Date()
      };
      setMessages([initMsg]);
    }
  }, [config.isAluno, messages.length]);

  // Heurística para detectar se a IA indicou que tem informações suficientes
  const lastBotMessage = [...messages].reverse().find(m => m.sender === 'bot' && m.content && m.content.trim().length > 0);
  const readyToRegister = Boolean(lastBotMessage && /pronto para registrar|revisar e confirmar|confirme se devo registrar|revise e confirme|posso registrar|devo registrar|informações suficientes|quando estiver tudo certo|já posso registrar|pronto para salvar/i.test(lastBotMessage.content));

  const clearChat = async () => {
    setIsClearing(true);
    try {
      setMessages([]);
      console.log(`${config.title} chat limpo com sucesso`);
    } finally {
      setIsClearing(false);
    }
  };

  const sendMessage = async () => {
    if (!inputMessage.trim() || isTyping) return;

    console.log(`🚀 Iniciando ${config.title} sendMessage...`);
    console.log('📝 Mensagem:', inputMessage);

    const userMessage: Message = {
      id: Date.now().toString(),
      content: inputMessage,
      sender: 'user',
      timestamp: new Date(),
    };

    setMessages(prev => [...prev, userMessage]);
    const currentInput = inputMessage;
    setInputMessage('');
    setIsTyping(true);

    // Cria mensagem do bot (inicialmente vazia para mostrar indicador de digitação)
    const botMessageId = (Date.now() + 1).toString();
    const botMessage: Message = {
      id: botMessageId,
      content: '',
      sender: 'bot',
      timestamp: new Date(),
    };

    setMessages(prev => [...prev, botMessage]);
    setStreamingMessageId(botMessageId);

    try {
      // Se este chat é o 'Aluno', adiciona instruções ao modelo para que ele aja como aluno
      let messageToSend = currentInput;
      if (config.isAluno) {
        const typeLabel = alunoType === 'business_rules' ? 'Regra de negócio' : alunoType === 'database_struct' ? 'Base de dados' : 'Serviço';
        const alunoInstructions = `INSTRUÇÕES AO ASSISTENTE-ALUNO: Você é um aluno que está aprendendo. O usuário está ensinando SOBRE: ${typeLabel}. Seu objetivo é extrair e construir um objeto JSON completo desse tipo com todos os campos necessários. Se faltarem informações, faça PERGUNTAS DIRETAS e ESPECÍFICAS ao usuário para obter os campos que faltam. Não invente valores. Quando você tiver todas as informações necessárias, responda primeiro a linha: "PRONTO_PARA_REGISTRAR" seguida do JSON completo. Pergunte apenas uma coisa por vez. Seja objetivo e claro. Agora segue a entrada do usuário:`;
        messageToSend = `${alunoInstructions}\n${currentInput}`;
      }
      // Prepara contexto das últimas 10 mensagens
      const context = messages.slice(-10).map(msg => ({
        role: msg.sender === 'user' ? 'user' : 'assistant',
        content: msg.content
      }));

      // Use endpoint de streaming com URL completa
      const fullURL = `${process.env.REACT_APP_API_URL}${config.streamEndpoint}?collection_name=${encodeURIComponent(selectedCollection)}`;
      
      const response = await fetch(fullURL, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
          'Accept': 'text/event-stream',
          'Cache-Control': 'no-cache',
        },
        body: JSON.stringify({
          message: messageToSend,
          context: context
        }),
        // Configurações específicas para streaming
        mode: 'cors',
        credentials: 'same-origin',
      });

      if (!response.ok) {
        throw new Error(`Erro na API: ${response.status}`);
      }

      const reader = response.body?.getReader();
      
      if (!reader) {
        throw new Error('Response body não disponível');
      }
      
      let accumulatedContent = '';
      let chunkCount = 0;
      
      try {
        while (true) {
          const { done, value } = await reader.read();
          chunkCount++;
          
          if (done) {
            break;
          }
          
          // Decodificar e processar chunk
          const chunk = new TextDecoder().decode(value);
          
          // Processar cada linha do chunk
          const lines = chunk.split('\n');
          for (const line of lines) {
            if (line.startsWith('data: ')) {
              const jsonStr = line.slice(6).trim();
              if (jsonStr && jsonStr !== '[DONE]') {
                try {
                  const data = JSON.parse(jsonStr);
                  if (data.content) {
                    accumulatedContent += data.content;
                    
                    // Atualização simples com setState
                    setMessages(prev => 
                      prev.map(msg => 
                        msg.id === botMessageId 
                          ? { ...msg, content: accumulatedContent }
                          : msg
                      )
                    );
                  }
                } catch (parseError) {
                  // Ignora erros de parse
                }
              }
            }
          }
        }
      } finally {
        reader.releaseLock();
      }

    } catch (error) {
      console.error('Erro ao enviar mensagem:', error);
      setMessages(prev => 
        prev.map(msg => 
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

  // Função que executa o registro do último item retornado pelo bot
  const handleRegister = async () => {
    const lastBot = [...messages].reverse().find(m => m.sender === 'bot' && m.content && m.content.trim().length>0);
    if (!lastBot) {
      alert('Nenhuma informação do bot encontrada para registrar.');
      return;
    }

    // Detecta se parece que o usuário descreveu mais de um tema (simples heurística)
    const containsMultiple = (lastBot.content || '').split(/[\.\n]/).filter(Boolean).length > 1 && /\band\b|\be\b|,/i.test(lastBot.content);
    if (containsMultiple) {
      const choose = window.confirm('Parece que você está falando sobre mais de um tema. Deseja continuar e registrar apenas este conteúdo, ou prefere escolher um único tema antes de registrar? (Clique Cancelar para escolher)');
      if (!choose) return;
    }

    let parsed = null;
    try {
      parsed = JSON.parse(lastBot.content);
    } catch (e) {
      const confirmTxt = window.confirm('O conteúdo do bot não está em JSON válido. Deseja registrar como texto livre?');
      if (!confirmTxt) return;
    }

    setRegistering(true);
    try {
      const payload: any = {
        collection_name: selectedCollection,
        type: alunoType
      };

      if (parsed) {
        payload['item'] = parsed;
      } else {
        payload['text'] = lastBot.content;
      }

      const resp = await fetch(`${process.env.REACT_APP_API_URL}/vectordb/add-item`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(payload)
      });

      if (!resp.ok) {
        const err = await resp.json().catch(() => null);
        throw new Error(err?.detail || 'Erro ao registrar item');
      }

      const confirmBotMsg: Message = {
        id: Date.now().toString() + '_regok',
        content: 'Ensinamento registrado com sucesso. Agora você já pode falar sobre outro assunto.',
        sender: 'bot',
        timestamp: new Date()
      };
      setMessages([confirmBotMsg]);
    } catch (err: any) {
      console.error(err);
      alert('Falha ao registrar item: ' + (err?.message || err));
    } finally {
      setRegistering(false);
    }
  };

  const handleKeyPress = (e: React.KeyboardEvent) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      sendMessage();
    }
  };

  return (
    <Box sx={{ display: 'flex', flexDirection: { xs: 'column', lg: 'row' }, gap: 3 }}>
      {/* Área Principal de Chat - mesma estrutura do ChatDialog */}
      <Box sx={{ flex: 2 }}>
        <Card sx={{ height: 'calc(100vh - 520px)', display: 'flex', flexDirection: 'column' }}>
          {/* Header - similar ao DialogTitle */}
          <Box
            sx={{
              background: 'linear-gradient(135deg, #ED145B 0%, #C7104A 100%)',
              color: '#ffffff',
              display: 'flex',
              alignItems: 'center',
              justifyContent: 'space-between',
              p: 2,
              borderRadius: '12px 12px 0 0'
            }}
          >
            <Box sx={{ display: 'flex', alignItems: 'center', gap: 2 }}>
              {config.icon}
              <Typography variant="h6">{config.title}</Typography>
              <Chip 
                label={`${messages.length} mensagens`} 
                size="small" 
                sx={{ 
                  backgroundColor: alpha('#ffffff', 0.2),
                  color: '#ffffff'
                }}
              />
            </Box>
            <Box sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
              {messages.length > 0 && (
                <IconButton
                  onClick={clearChat}
                  disabled={isClearing}
                  sx={{ 
                    color: '#ffffff',
                    backgroundColor: alpha('#ffffff', 0.1),
                    '&:hover': { backgroundColor: alpha('#ffffff', 0.2) }
                  }}
                  title="Limpar conversa"
                >
                  {isClearing ? <RefreshIcon className="spinning" /> : <DeleteIcon />}
                </IconButton>
              )}
            </Box>
          </Box>

          {/* Content - similar ao DialogContent */}
          <CardContent 
            sx={{ 
              p: 0, 
              display: 'flex', 
              flexDirection: 'column',
              height: '100%',
              flex: 1
            }}
          >
            <Box 
              sx={{ 
                flex: 1, 
                overflow: 'auto', 
                p: 2,
                display: 'flex',
                flexDirection: 'column',
                gap: 2
              }}
            >
              {messages.length === 0 && (
                <Box sx={{ 
                  display: 'flex', 
                  flexDirection: 'column', 
                  alignItems: 'center', 
                  justifyContent: 'center',
                  height: '100%',
                  textAlign: 'center',
                  opacity: 0.7
                }}>
                  <BotIcon sx={{ fontSize: 48, mb: 2, color: 'primary.main' }} />
                  <Typography variant="h6" gutterBottom>
                    {config.description}
                  </Typography>
                  <Typography variant="body2" color="text.secondary">
                    {config.emptyStateMessage}
                  </Typography>
                </Box>
              )}

              {/* Messages - mesma estrutura do ChatDialog */}
              {messages.map((message) => (
                <Box
                  key={message.id}
                  sx={{
                    display: 'flex',
                    justifyContent: message.sender === 'user' ? 'flex-end' : 'flex-start',
                    mb: 1,
                  }}
                >
                  <Box
                    sx={{
                      display: 'flex',
                      flexDirection: message.sender === 'user' ? 'row-reverse' : 'row',
                      alignItems: 'flex-start',
                      gap: 1,
                      maxWidth: '95%',
                    }}
                  >
                    <Avatar
                      sx={{
                        width: 32,
                        height: 32,
                        background: message.sender === 'user'
                          ? '#64748b'
                          : '#64748b',
                        color: '#ffffff',
                      }}
                    >
                      {message.sender === 'user' ? <PersonIcon /> : <BotIcon />}
                    </Avatar>

                    <Paper
                      elevation={1}
                      sx={{
                        p: 2,
                        borderRadius: 2,
                        background: message.sender === 'user'
                          ? '#ED145B'
                          : darkMode ? '#1e293b' : '#f8fafc',
                        color: message.sender === 'user' 
                          ? '#ffffff' 
                          : darkMode ? '#ffffff' : '#000000',
                        wordBreak: 'break-word',
                        border: message.sender === 'bot' ? `1px solid ${darkMode ? '#475569' : '#e2e8f0'}` : 'none',
                      }}
                    >
                      {message.sender === 'bot' ? (
                        config.isSQL && message.content ? (
                          <Box>
                            <Typography variant="body2" fontWeight="bold" sx={{ mb: 1 }}>
                              SQL Gerado:
                            </Typography>
                            <pre style={{ 
                              background: '#2d2d2d', 
                              color: '#ffffff',
                              padding: '12px', 
                              borderRadius: '6px', 
                              fontSize: '0.9rem',
                              fontFamily: 'Monaco, Consolas, monospace',
                              whiteSpace: 'pre-wrap',
                              overflow: 'auto',
                              margin: 0
                            }}>
                              {message.content}
                            </pre>
                          </Box>
                        ) : (
                          <MessageRenderer 
                            content={message.content} 
                            isStreaming={streamingMessageId === message.id}
                            darkMode={darkMode}
                            isTyping={isTyping && message.content === ''}
                          />
                        )
                      ) : (
                        <Typography variant="body1">
                          {message.content}
                        </Typography>
                      )}
                      
                      <Typography 
                        variant="caption" 
                        sx={{ 
                          display: 'block', 
                          mt: 1, 
                          opacity: 0.7,
                          textAlign: message.sender === 'user' ? 'right' : 'left'
                        }}
                      >
                        {message.timestamp.toLocaleTimeString()}
                      </Typography>
                    </Paper>
                  </Box>
                </Box>
              ))}

              <div ref={messagesEndRef} />
            </Box>

            {/* Input Area - similar ao DialogActions */}
            <Box sx={{ p: 2, borderTop: `1px solid ${theme.palette.divider}` }}>
              <Box sx={{ display: 'flex', gap: 1, width: '100%' }}>
                <TextField
                  fullWidth
                  multiline
                  maxRows={4}
                  value={inputMessage}
                  onChange={(e) => setInputMessage(e.target.value)}
                  onKeyPress={handleKeyPress}
                  placeholder={config.placeholder}
                  disabled={isTyping}
                  variant="outlined"
                  size="small"
                  sx={{
                    '& .MuiOutlinedInput-root': {
                      borderRadius: 3,
                    }
                  }}
                />
                <Button
                  variant="contained"
                  onClick={sendMessage}
                  disabled={!inputMessage.trim() || isTyping}
                  sx={{
                    borderRadius: 3,
                    minWidth: 48,
                    background: '#ED145B',
                    '&:hover': {
                      background: '#C7104A',
                    },
                    '&:disabled': {
                      background: '#94a3b8',
                      color: '#ffffff',
                    }
                  }}
                >
                  <SendIcon />
                </Button>
                {config.isAluno && readyToRegister && (
                  <Button
                    variant="contained"
                    color="secondary"
                    onClick={handleRegister}
                    disabled={registering}
                    sx={{ borderRadius: 3, minWidth: 140 }}
                  >
                    Registrar
                  </Button>
                )}
              </Box>
            </Box>
            {/* Aluno controls: tipo (sem botão Registrar ao lado) */}
            {config.isAluno && (
              <Box sx={{ p: 2, borderTop: `1px solid ${theme.palette.divider}`, display: 'flex', gap: 2, alignItems: 'center' }}>
                <TextField
                  select
                  SelectProps={{ native: true }}
                  value={alunoType}
                  onChange={(e) => setAlunoType(e.target.value)}
                  size="small"
                  label="Tipo"
                  sx={{ minWidth: 220 }}
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

      {/* Painel Lateral de Sugestões */}
      <Box sx={{ flex: 1 }}>
        <Card>
          <CardContent>
            <Typography variant="h6" gutterBottom>
              💡 {config.suggestionsTitle}
            </Typography>
            
            <Typography variant="body2" color="text.secondary" paragraph>
              {config.suggestionsDescription}
            </Typography>

            <Box sx={{ display: 'flex', flexDirection: 'column', gap: 1, mb: 3 }}>
              {config.suggestions.map((suggestion, index) => (
                <Chip
                  key={index}
                  label={suggestion}
                  variant="outlined"
                  clickable
                  onClick={() => setInputMessage(suggestion)}
                  size="small"
                  sx={{ 
                    justifyContent: 'flex-start',
                    height: 'auto',
                    py: 1,
                    '& .MuiChip-label': {
                      whiteSpace: 'normal',
                      textAlign: 'left'
                    }
                  }}
                />
              ))}
            </Box>
          </CardContent>
        </Card>
      </Box>
    </Box>
  );
};

// Configurações para cada tipo de chat
const chatConfigs = {
  general: {
    title: '💬 Chat Geral',
    icon: <ChatIcon />,
    endpoint: '/api/chat',
    streamEndpoint: '/api/chat/stream',
    placeholder: 'Digite sua mensagem para conversar...',
    description: 'Dúvidas?',
    emptyStateMessage: 'Converse livremente com a IA sobre qualquer assunto.',
    suggestions: [
      "Qual é o seu nome?",
      "Olá! Como você pode me ajudar?",
      "Me explique sobre inteligência artificial",
      "Quais são as últimas tendências em tecnologia?",
      "Como posso melhorar minha produtividade?",
      "Me conte uma curiosidade interessante",
      "Como funciona o machine learning?",
      "Quais são os benefícios da automação empresarial?",
      "Me explique sobre computação em nuvem",
      "Como a IA está transformando os negócios?",
      "Quais são as melhores práticas de segurança digital?",
      "Me conte sobre desenvolvimento de software",
      "Como escolher as melhores ferramentas para minha empresa?",
      "Quais são as tendências em análise de dados?",
      "Me explique sobre transformação digital"
    ],
    suggestionsTitle: 'Sugestões de Conversa',
    suggestionsDescription: 'Clique nas sugestões abaixo para iniciar uma conversa:',
    isSQL: false
  },
  sql: {
    title: '🔍 Chat SQL',
    icon: <CodeIcon />,
    endpoint: '/api/chat/sql',
    streamEndpoint: '/api/chat/sql/stream',
    placeholder: 'Ex: Como buscar todos os clientes ativos?',
    description: 'Gerador de SQL',
    emptyStateMessage: 'Faça perguntas para gerar consultas SQL baseadas na estrutura do banco.',
    suggestions: [
      "Liste todos os clientes ativos",
      "Busque vendas do último mês", 
      "Produtos com estoque baixo",
      "Clientes inadimplentes",
      "Vendas por funcionário",
      "Top 10 produtos mais vendidos",
      "Clientes que mais compraram este ano",
      "Vendas por região geográfica",
      "Produtos sem movimento nos últimos 6 meses",
      "Fornecedores com mais pedidos",
      "Média de vendas por vendedor",
      "Clientes com limite de crédito estourado",
      "Campanhas de marketing mais efetivas",
      "Tickets de suporte em aberto",
      "Contratos que vencem nos próximos 30 dias"
    ],
    suggestionsTitle: 'Exemplos de Consultas',
    suggestionsDescription: 'Clique nas sugestões abaixo para gerar SQL:',
    isSQL: true
  },
  help: {
    title: '❓ Chat Dúvidas',
    icon: <HelpIcon />,
    endpoint: '/api/chat',
    streamEndpoint: '/api/chat/stream',
    placeholder: 'Ex: Como funciona o processo de vendas?',
    description: 'Assistente de Dúvidas',
    emptyStateMessage: 'Tire dúvidas sobre o sistema comercial e regras de negócio.',
    suggestions: [
      "O que você sabe?",
      "Como funciona o processo de vendas?",
      "Quais são as regras de desconto?",
      "Como calcular comissões?",
      "O que fazer com clientes inadimplentes?",
      "Como gerar relatórios mensais?",
      "Como cadastrar um novo cliente?",
      "Qual é o fluxo de aprovação de crédito?",
      "Como alterar preços de produtos?",
      "Quais são os prazos de entrega padrão?",
      "Como cancelar uma venda?",
      "Como configurar uma campanha de marketing?",
      "Qual é o processo de devolução de produtos?",
      "Como acompanhar o status de uma entrega?",
      "Quais são os níveis de acesso do sistema?",
      "Como fazer backup dos dados?"
    ],
    suggestionsTitle: 'Exemplos de Dúvidas',
    suggestionsDescription: 'Clique nas sugestões abaixo para fazer perguntas:',
    isSQL: false
  },
  aluno: {
    title: '🧠 Chat Aluno',
    icon: <BotIcon />,
    endpoint: '/api/chat',
    streamEndpoint: '/api/chat/stream',
    placeholder: 'Descreva a informação que você quer que eu aprenda...',
    description: 'Este chat aprenderá com base no que você informar.',
    emptyStateMessage: 'Explique algo que deseja registrar no sistema. Quando estiver pronto, revise e confirme para registrar.',
    suggestions: [
      'Registrar uma regra de negócio sobre descontos',
      'Descrever nova tabela de clientes',
      'Registrar serviço de integração com ERP',
      'Explicar validação para campo CPF',
      'Detalhar campo de endereço na tabela cliente',
      'quero adicionar uma nova regra para novos o clientes, só pode ser cadastrado, se tiver a informação de renda e endereço'
    ],
    suggestionsTitle: 'Exemplos para teste',
    suggestionsDescription: 'Clique em um exemplo para preencher a mensagem e testar o fluxo de registro.',
    isSQL: false,
    isAluno: true
  }
};

// Configuração de visibilidade das abas
const tabsConfig = {
  general: { 
    visible: true, 
    index: 0,
    tabProps: {
      icon: <ChatIcon />, 
      label: "Chat Geral", 
      iconPosition: "start" as const,
      sx: { fontWeight: 'bold' }
    }
  },
  sql: { 
    visible: true, 
    index: 1,
    tabProps: {
      icon: <CodeIcon />, 
      label: "Chat SQL", 
      iconPosition: "start" as const,
      sx: { fontWeight: 'bold' }
    }
  },
  help: { 
    visible: true, 
    index: 2,
    tabProps: {
      icon: <HelpIcon />, 
      label: "Chat Dúvidas", 
      iconPosition: "start" as const,
      sx: { fontWeight: 'bold' }
    }
  }
  ,
  aluno: { 
    visible: true, 
    index: 3,
    tabProps: {
      icon: <BotIcon />, 
      label: "Chat Aluno", 
      iconPosition: "start" as const,
      sx: { fontWeight: 'bold' }
    }
  }
};

const ChatTabs: React.FC<{ darkMode?: boolean }> = ({ darkMode = false }) => {
  const [selectedTab, setSelectedTab] = useState(0);

  // Filtra apenas as abas visíveis
  const visibleTabs = Object.entries(tabsConfig).filter(([_, config]) => config.visible);
  const tabKeys = visibleTabs.map(([key, _]) => key);

  return (
    <Box sx={{ width: '100%', typography: 'body1' }}>
      {/* Tabs de Navegação */}
      <Paper sx={{ mb: 3 }}>
        <Tabs 
          value={selectedTab} 
          onChange={(_, newValue) => setSelectedTab(newValue)}
          variant="fullWidth"
          indicatorColor="primary"
          textColor="primary"
        >
          {visibleTabs.map(([key, config], index) => (
            <Tab 
              key={key}
              {...config.tabProps}
            />
          ))}
        </Tabs>
      </Paper>

      {/* Conteúdo das Tabs */}
      <Box>
        {tabKeys.map((key, index) => (
          <Box
            key={key}
            sx={{ 
              display: selectedTab === index ? 'block' : 'none'
            }}
          >
            <ConfigurableChat 
              config={chatConfigs[key as keyof typeof chatConfigs]} 
              darkMode={darkMode} 
            />
          </Box>
        ))}
      </Box>
    </Box>
  );
};

export default ChatTabs;