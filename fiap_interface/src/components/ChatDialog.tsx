import React, { useState, useRef, useEffect } from 'react';
import {
  Dialog,
  DialogTitle,
  DialogContent,
  DialogActions,
  IconButton,
  Typography,
  Box,
  TextField,
  Button,
  Paper,
  Avatar,
  Slide,
  useTheme,
  alpha,
  Chip
} from '@mui/material';
import {
  Close as CloseIcon,
  Send as SendIcon,
  SmartToy as BotIcon,
  Person as PersonIcon,
  DarkMode as DarkModeIcon,
  LightMode as LightModeIcon,
  Delete as DeleteIcon,
  Refresh as RefreshIcon
} from '@mui/icons-material';
import { TransitionProps } from '@mui/material/transitions';
import MessageRenderer from './MessageRenderer';

const Transition = React.forwardRef(function Transition(
  props: TransitionProps & {
    children: React.ReactElement<any, any>;
  },
  ref: React.Ref<unknown>,
) {
  return <Slide direction="up" ref={ref} {...props} />;
});

interface Message {
  id: string;
  content: string;
  sender: 'user' | 'bot';
  timestamp: Date;
}

interface ChatDialogProps {
  open: boolean;
  onClose: () => void;
  darkMode: boolean;
}

const ChatDialog: React.FC<ChatDialogProps> = ({ open, onClose, darkMode }) => {
  const [messages, setMessages] = useState<Message[]>([]);
  const [inputMessage, setInputMessage] = useState('');
  const [streamingMessageId, setStreamingMessageId] = useState<string | null>(null);
  const [isTyping, setIsTyping] = useState(false);
  const [sessionId, setSessionId] = useState<string | null>(null);
  
  // Debug: Log whenever sessionId changes
  useEffect(() => {
    console.log('🔄 SessionId mudou para:', sessionId);
  }, [sessionId]);
  const [isClearing, setIsClearing] = useState(false);
  const messagesEndRef = useRef<HTMLDivElement>(null);
  const theme = useTheme();
  const API_URL = process.env.REACT_APP_API_URL;

  const scrollToBottom = () => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  };

  useEffect(() => {
    scrollToBottom();
  }, [messages]);

  const clearChat = async () => {
    if (!sessionId) {
      // Se não há sessão, apenas limpa a interface
      setMessages([]);
      setSessionId(null);
      return;
    }

    setIsClearing(true);
    try {
      const response = await fetch(`${API_URL}/chat/clear`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({ session_id: sessionId }),
      });

      if (response.ok) {
        setMessages([]);
        setSessionId(null);
        console.log('Chat limpo com sucesso');
      } else {
        console.error('Erro ao limpar chat:', response.status);
        // Limpa a interface mesmo se o servidor falhar
        setMessages([]);
        setSessionId(null);
      }
    } catch (error) {
      console.error('Erro ao limpar chat:', error);
      // Limpa a interface mesmo se houver erro
      setMessages([]);
      setSessionId(null);
    } finally {
      setIsClearing(false);
    }
  };

  const sendMessage = async () => {
    if (!inputMessage.trim() || isTyping) return;

    console.log('🚀 Iniciando sendMessage...');
    console.log('📝 Mensagem:', inputMessage);
    console.log('🔗 SessionId atual:', sessionId);

    const userMessage: Message = {
      id: Date.now().toString(),
      content: inputMessage,
      sender: 'user',
      timestamp: new Date(),
    };

    setMessages(prev => [...prev, userMessage]);
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
      // Prepara o payload com session_id se disponível
      const requestBody: any = { message: inputMessage };
      if (sessionId) {
        requestBody.session_id = sessionId;
        console.log('🔗 Incluindo session_id na request:', sessionId);
      } else {
        console.log('⚠️ Nenhum session_id disponível para incluir na request');
      }
      
      console.log('📤 Request body completo:', JSON.stringify(requestBody, null, 2));

      const response = await fetch(`${API_URL}/chat`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify(requestBody),
      });

      if (!response.ok) {
        throw new Error(`Erro na API: ${response.status}`);
      }

      const reader = response.body?.getReader();
      const decoder = new TextDecoder();

      if (reader) {
        let buffer = '';
        let endReceived = false;
        
        while (true) {
          const { done, value } = await reader.read();
          if (done) break;

          const chunk = decoder.decode(value, { stream: true });
          buffer += chunk;
          
          // Processa todas as linhas completas no buffer
          const lines = buffer.split('\n');
          buffer = lines.pop() || ''; // Mantém a linha incompleta no buffer

          for (const line of lines) {
            if (line.startsWith('data: ')) {
              try {
                const data = JSON.parse(line.slice(6));
                console.log('📡 Dados recebidos:', data);
                
                if (data.type === 'chunk' && data.content && !endReceived) {
                  // Remove indicador de digitação assim que receber o primeiro chunk
                  if (isTyping) {
                    setIsTyping(false);
                  }
                  
                  setMessages(prev => 
                    prev.map(msg => 
                      msg.id === botMessageId 
                        ? { ...msg, content: msg.content + data.content }
                        : msg
                    )
                  );
                  
                  // Força scroll para baixo após cada chunk
                  setTimeout(() => scrollToBottom(), 10);
                  
                } else if (data.type === 'session_info' && data.session_id) {
                  // Armazena o session_id para próximas mensagens
                  setSessionId(data.session_id);
                  console.log('🎉 Session ID atualizado:', data.session_id);
                  console.log('🎉 Dados completos do session_info:', data);
                  
                } else if (data.type === 'end') {
                  console.log('📝 Recebido type: end - marcando como finalizado');
                  endReceived = true;
                  
                } else if (data.type === 'error') {
                  setMessages(prev => 
                    prev.map(msg => 
                      msg.id === botMessageId 
                        ? { ...msg, content: `Erro: ${data.error}` }
                        : msg
                    )
                  );
                  break;
                }
              } catch (e) {
                console.error('Erro ao parsear JSON:', e);
              }
            }
          }
        }
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

  const handleKeyPress = (e: React.KeyboardEvent) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      sendMessage();
    }
  };

  return (
    <Dialog
      open={open}
      onClose={onClose}
      TransitionComponent={Transition}
      maxWidth="xl"
      fullWidth
      PaperProps={{
        sx: {
          height: '80vh',
          maxHeight: '600px',
          borderRadius: 3,
          background: darkMode 
            ? alpha(theme.palette.background.paper, 0.95)
            : alpha(theme.palette.background.paper, 0.98),
          backdropFilter: 'blur(10px)',
        }
      }}
    >
      <DialogTitle
        sx={{
          background: 'linear-gradient(135deg, #2563eb 0%, #1d4ed8 100%)',
          color: '#ffffff',
          display: 'flex',
          alignItems: 'center',
          justifyContent: 'space-between',
          py: 2,
        }}
      >
        <Box sx={{ display: 'flex', alignItems: 'center', gap: 2 }}>
          <BotIcon />
          <Box>
            <Typography variant="h6">Chat com LMStudio</Typography>
            {sessionId && (
              <Typography variant="caption" sx={{ opacity: 0.8 }}>
                Sessão: {sessionId.substring(0, 8)}...
              </Typography>
            )}
          </Box>
          <Chip 
            icon={darkMode ? <DarkModeIcon /> : <LightModeIcon />}
            label={darkMode ? "Escuro" : "Claro"}
            size="small"
            sx={{ 
              backgroundColor: alpha('#ffffff', 0.2),
              color: '#ffffff',
              '& .MuiChip-icon': { color: '#ffffff' }
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
          <IconButton
            onClick={onClose}
            sx={{ color: '#ffffff' }}
          >
            <CloseIcon />
          </IconButton>
        </Box>
      </DialogTitle>

      <DialogContent 
        sx={{ 
          p: 0, 
          display: 'flex', 
          flexDirection: 'column',
          height: '100%'
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
                Bem-vindo ao Chat!
              </Typography>
              <Typography variant="body2" color="text.secondary">
                Digite sua mensagem abaixo para começar a conversar com a IA.
              </Typography>
            </Box>
          )}

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
                      ? '#2563eb'
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
                      ? '#2563eb'
                      : darkMode ? '#1e293b' : '#f8fafc',
                    color: message.sender === 'user' ? '#ffffff' : 'text.primary',
                    wordBreak: 'break-word',
                    border: message.sender === 'bot' ? `1px solid ${darkMode ? '#475569' : '#e2e8f0'}` : 'none',
                  }}
                >
                  {message.sender === 'bot' ? (
                    <MessageRenderer 
                      content={message.content} 
                      isStreaming={streamingMessageId === message.id}
                      darkMode={darkMode}
                      isTyping={isTyping && message.content === ''}
                    />
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
      </DialogContent>

      <DialogActions sx={{ p: 2, borderTop: `1px solid ${theme.palette.divider}` }}>
        <Box sx={{ display: 'flex', gap: 1, width: '100%' }}>
          <TextField
            fullWidth
            multiline
            maxRows={4}
            value={inputMessage}
            onChange={(e) => setInputMessage(e.target.value)}
            onKeyPress={handleKeyPress}
            placeholder="Digite sua mensagem..."
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
              background: '#2563eb',
              '&:hover': {
                background: '#1d4ed8',
              },
              '&:disabled': {
                background: '#94a3b8',
                color: '#ffffff',
              }
            }}
          >
            <SendIcon />
          </Button>
        </Box>
      </DialogActions>
    </Dialog>
  );
};

export default ChatDialog;
