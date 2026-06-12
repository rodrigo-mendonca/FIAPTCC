import {
  Box,
  Card,
  CardContent,
  Typography,
  TextField,
  Button,
  Alert,
  CircularProgress,
  FormControl,
  InputLabel,
  Select,
  MenuItem,
  SelectChangeEvent,
  Dialog,
  DialogTitle,
  DialogContent,
  DialogActions,
  DialogContentText,
  Tabs,
  Tab,
  Chip,
  LinearProgress,
  List,
  ListItem,
  ListItemIcon,
  ListItemText,
  IconButton
} from '@mui/material';
import {
  Send as SendIcon,
  Delete as DeleteIcon,
  Search as SearchIcon,
  Settings as SettingsIcon,
  CheckCircle as CheckCircleIcon,
  UploadFile as UploadFileIcon,
  Error as ErrorIcon,
  InsertDriveFile as InsertDriveFileIcon
} from '@mui/icons-material';
import React, { useState, useEffect } from 'react';
import { useCollection } from '../contexts/CollectionContext';
import { useNotification } from '../contexts/NotificationContext';

const API_URL = process.env.REACT_APP_API_URL;

// Estado de processamento de um arquivo individual durante o upload em streaming
type FileUploadStatus = {
  state: 'pending' | 'processing' | 'success' | 'error';
  type?: string | null;
  message?: string;
};

const VectorDBTest: React.FC = () => {
  const [question, setQuestion] = useState('');
  const [loading, setLoading] = useState(false);
  const [subTab, setSubTab] = useState(0);
  const [dataSource, setDataSource] = useState('Todos');
  const [confirmDialogOpen, setConfirmDialogOpen] = useState(false);
  const [selectedFiles, setSelectedFiles] = useState<File[]>([]);
  const [uploadingFiles, setUploadingFiles] = useState(false);
  const [uploadError, setUploadError] = useState<string | null>(null);
  const [uploadSuccess, setUploadSuccess] = useState<string | null>(null);
  const [shouldReload, setShouldReload] = useState(false);
  const [isDragging, setIsDragging] = useState(false);
  // Status de cada arquivo durante o upload em streaming
  const [fileStatus, setFileStatus] = useState<{ [key: string]: FileUploadStatus }>({});
  // Contador de progresso geral (arquivos processados / total)
  const [uploadProgress, setUploadProgress] = useState<{ done: number; total: number }>({ done: 0, total: 0 });
  const [queryError, setQueryError] = useState<string | null>(null);
  const [clearError, setClearError] = useState<string | null>(null);
  const [clearMessage, setClearMessage] = useState<string | null>(null);
  const [results, setResults] = useState<any[]>([]);
  const [fileTypes, setFileTypes] = useState<any>(null);
  const { selectedCollection, refreshCollections } = useCollection();
  const { showNotification } = useNotification();

  // Carregar tipos de arquivo suportados
  useEffect(() => {
    const loadFileTypes = async () => {
      try {
        const response = await fetch(`${API_URL}/api/vectordb/file-types`);
        if (response.ok) {
          const data = await response.json();
          setFileTypes(data);
        }
      } catch (error) {
        console.warn('Erro ao carregar tipos de arquivo:', error);
      }
    };
    loadFileTypes();
  }, []);

  const handleSubTabChange = (event: React.SyntheticEvent, newValue: number) => {
    setSubTab(newValue);
  };

  const handleDataSourceChange = (event: SelectChangeEvent) => {
    setDataSource(event.target.value);
  };

  const handleQuery = async () => {
    if (!question.trim()) return;

    setLoading(true);
    setResults([]);
    setQueryError(null);

    console.log('🔍 Fazendo query:', { question, collection: selectedCollection });

    try {
      const response = await fetch(`${API_URL}/vectordb/query`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          question: question,
          collection_name: selectedCollection,
          context: dataSource === 'Todos' ? 'all' : dataSource
        }),
      });

      if (response.ok) {
        const data = await response.json();
        console.log('✅ Resultados recebidos:', data);
        setResults(data.results || []);
        
        // Mostrar notificação com as tabelas encontradas
        if (data.tables_found && data.tables_found.length > 0) {
          showNotification(`Tabelas encontradas: ${data.tables_found.join(', ')}`, 'info');
        }
      } else {
        let errText = response.statusText || `Status ${response.status}`;
        try {
          const body = await response.text();
          if (body) {
            try {
              const json = JSON.parse(body);
              errText = json.detail || json.message || body;
            } catch {
              errText = body;
            }
          }
        } catch {}

        const lower = String(errText).toLowerCase();
        if (lower.includes('lmstudio') || lower.includes('lm-studio') || response.status === 502 || response.status === 503) {
          setQueryError('LMStudio não está disponível ou não possui modelos carregados. Verifique se o LMStudio está aberto e com os modelos carregados.');
        } else {
          setQueryError(`Erro na consulta: ${errText}`);
        }
      }
    } catch (error: any) {
      console.error('Erro:', error);
      const msg = String(error?.message || error || 'Erro ao fazer consulta');
      if (msg.toLowerCase().includes('failed to fetch') || msg.toLowerCase().includes('network')) {
        setQueryError('Não foi possível conectar ao backend. Verifique se a API está rodando.');
      } else {
        setQueryError(msg);
      }
    } finally {
      setLoading(false);
    }
  };

  const handleClearDatabase = () => {
    setConfirmDialogOpen(true);
  };

  const handleConfirmClear = () => {
    // Chama a API para limpar a base de dados
    const doClear = async () => {
      setClearError(null);
      setClearMessage(null);
      try {
        const response = await fetch(`${API_URL}/vectordb/clear`, {
          method: 'POST'
        });

        if (response.ok) {
          const data = await response.json();
          setClearMessage(data.message || 'Base de dados limpa com sucesso');
        } else {
          let errText = response.statusText || `Status ${response.status}`;
          try {
            const body = await response.text();
            if (body) {
              try {
                const json = JSON.parse(body);
                errText = json.detail || json.message || body;
              } catch {
                errText = body;
              }
            }
          } catch {}

          const lower = String(errText).toLowerCase();
          if (lower.includes('lmstudio') || lower.includes('lm-studio') || response.status === 502 || response.status === 503) {
            setClearError('LMStudio não está disponível ou não possui modelos carregados. Verifique se o LMStudio está aberto e com os modelos carregados.');
          } else {
            setClearError(`Erro ao limpar base: ${errText}`);
          }
        }
      } catch (error: any) {
        console.error('Erro ao limpar base:', error);
        const msg = String(error?.message || error || 'Erro ao limpar base');
        if (msg.toLowerCase().includes('failed to fetch') || msg.toLowerCase().includes('network')) {
          setClearError('Não foi possível conectar ao backend. Verifique se a API está rodando.');
        } else {
          setClearError(msg);
        }
      } finally {
        setConfirmDialogOpen(false);
      }
    };

    doClear();
  };

  const handleCancelClear = () => {
    setConfirmDialogOpen(false);
  };

  // Adiciona novos arquivos à seleção (vindos do input ou do drag & drop),
  // ignorando duplicados pelo nome.
  const addFiles = (incoming: File[]) => {
    if (incoming.length === 0) return;
    setSelectedFiles(prev => {
      const existing = new Set(prev.map(f => f.name));
      const merged = [...prev];
      for (const f of incoming) {
        if (!existing.has(f.name)) merged.push(f);
      }
      return merged;
    });
    setUploadError(null);
    setUploadSuccess(null);
    setFileStatus({});
    setUploadProgress({ done: 0, total: 0 });
  };

  const handleFileChange = (event: React.ChangeEvent<HTMLInputElement>) => {
    addFiles(Array.from(event.target.files || []));
  };

  const handleRemoveFile = (name: string) => {
    setSelectedFiles(prev => prev.filter(f => f.name !== name));
  };

  const handleDragOver = (event: React.DragEvent) => {
    event.preventDefault();
    if (!uploadingFiles) setIsDragging(true);
  };

  const handleDragLeave = (event: React.DragEvent) => {
    event.preventDefault();
    setIsDragging(false);
  };

  const handleDrop = (event: React.DragEvent) => {
    event.preventDefault();
    setIsDragging(false);
    if (uploadingFiles) return;
    const dropped = Array.from(event.dataTransfer.files || []).filter(f =>
      /\.(ya?ml|json)$/i.test(f.name)
    );
    if (dropped.length === 0) {
      showNotification('Apenas arquivos YAML, YML ou JSON são aceitos.', 'warning');
      return;
    }
    addFiles(dropped);
  };

  const handleUploadFile = async () => {
    if (selectedFiles.length === 0) {
      showNotification('Selecione pelo menos um arquivo para enviar.', 'warning');
      return;
    }

    setUploadingFiles(true);
    setUploadError(null);
    setUploadSuccess(null);
    // Todos os arquivos começam como pendentes
    setFileStatus(Object.fromEntries(selectedFiles.map(f => [f.name, { state: 'pending' } as FileUploadStatus])));
    setUploadProgress({ done: 0, total: selectedFiles.length });

    try {
      // Verificar conexão com ChromaDB antes de tentar upload
      try {
        const healthResponse = await fetch(`${API_URL}/api/vectordb/health`);
        const health = await healthResponse.json();

        if (!health.connected) {
          const reconnectResponse = await fetch(`${API_URL}/api/vectordb/reconnect`, {
            method: 'POST'
          });
          const reconnectResult = await reconnectResponse.json();

          if (reconnectResult.status !== 'ok') {
            setUploadError('ChromaDB não está disponível. Verifique se o serviço está rodando.');
            showNotification('ChromaDB não disponível', 'error');
            setUploadingFiles(false);
            return;
          }
        }
      } catch (healthError) {
        console.warn('Erro ao verificar saúde do ChromaDB:', healthError);
      }

      // Preparar FormData para upload em lote
      const formData = new FormData();
      for (const file of selectedFiles) {
        formData.append('files', file);
      }
      formData.append('collection_name', selectedCollection);

      const response = await fetch(`${API_URL}/api/vectordb/upload-batch`, {
        method: 'POST',
        body: formData,
      });

      if (!response.ok || !response.body) {
        let errText = response.statusText || `Status ${response.status}`;
        try {
          const body = await response.text();
          if (body) {
            try {
              const json = JSON.parse(body);
              errText = json.detail || json.message || body;
            } catch {
              errText = body;
            }
          }
        } catch {}

        setUploadError(`❌ Erro: ${errText}`);
        showNotification(`Erro ao enviar arquivos: ${errText}`, 'error');
        setUploadingFiles(false);
        return;
      }

      // Consumir o stream (Server-Sent Events) emitido pela API
      const reader = response.body.getReader();
      const decoder = new TextDecoder();
      let buffer = '';
      let completeData: any = null;

      const handleEvent = (data: any) => {
        switch (data.event) {
          case 'start':
            setUploadProgress({ done: 0, total: data.total });
            break;
          case 'file_start':
            setFileStatus(prev => ({ ...prev, [data.filename]: { state: 'processing' } }));
            break;
          case 'file_done':
            setFileStatus(prev => ({
              ...prev,
              [data.filename]: {
                state: data.status === 'success' ? 'success' : 'error',
                type: data.type,
                message: data.message,
              },
            }));
            setUploadProgress(prev => ({ ...prev, done: data.index }));
            break;
          case 'complete':
            completeData = data;
            break;
          default:
            break;
        }
      };

      // eslint-disable-next-line no-constant-condition
      while (true) {
        const { done, value } = await reader.read();
        if (done) break;
        buffer += decoder.decode(value, { stream: true });

        // Eventos SSE são separados por linha em branco
        const parts = buffer.split('\n\n');
        buffer = parts.pop() || '';
        for (const part of parts) {
          const line = part.split('\n').find(l => l.startsWith('data:'));
          if (!line) continue;
          try {
            handleEvent(JSON.parse(line.slice(5).trim()));
          } catch (e) {
            console.warn('Falha ao interpretar evento SSE:', part, e);
          }
        }
      }

      // Processar resultado final
      const result = completeData || { success_count: 0, error_count: 0, results: [] };

      if (result.success_count > 0) {
        const msg = `✅ ${result.success_count} arquivo(s) enviado(s) com sucesso${result.error_count > 0 ? ` (${result.error_count} falharam)` : ''}`;
        setUploadSuccess(msg);
        showNotification(msg, 'success');

        // Atualizar lista de coleções no contexto
        await refreshCollections();

        // Limpar a seleção mantendo o resumo de status visível
        const fileInput = document.querySelector('input[type="file"]') as HTMLInputElement;
        if (fileInput) fileInput.value = '';
        setUploadError(null);

        setTimeout(() => {
          setSelectedFiles([]);
          setFileStatus({});
          setUploadProgress({ done: 0, total: 0 });
          setUploadSuccess(null);
        }, 3000);
      } else if (result.error_count > 0) {
        const errorMessages = Array.isArray(result.results)
          ? result.results.filter((r:any) => r.status === 'error' && r.message)
              .map((r:any) => `${r.filename ? r.filename + ': ' : ''}${r.message}`)
          : [];
        setUploadError(errorMessages.length > 0 ? errorMessages.join('\n') : `❌ Todos os ${result.error_count} arquivo(s) falharam`);
        showNotification(`Erro ao enviar arquivos`, 'error');
      }

    } catch (error: any) {
      console.error('Erro ao enviar arquivos:', error);
      const msg = String(error?.message || error || 'Erro de conexão');
      setUploadError(`❌ Erro: ${msg}`);
      showNotification(`Erro: ${msg}`, 'error');
    } finally {
      setUploadingFiles(false);
    }
  };

  // Quando `shouldReload` for true, executa reload controlado por hook
  useEffect(() => {
    if (!shouldReload) return;

    const timer = setTimeout(() => {
      try {
        window.location.reload();
      } catch (e) {
        console.warn('Reload falhou:', e);
      }
    }, 800);

    return () => clearTimeout(timer);
  }, [shouldReload]);

  return (
    <Box sx={{ width: '100%' }}>
      <Tabs variant="fullWidth" indicatorColor="primary" textColor="primary" value={subTab} onChange={handleSubTabChange} centered>
        <Tab icon={<SearchIcon />} label="Consulta" />
        <Tab icon={<SettingsIcon />} label="Manutenção" />
      </Tabs>

      <Box sx={{ mt: 2 }}>
        {subTab === 0 && (
          <Box>
            <Card sx={{ mb: 3 }}>
              <CardContent>
                <Typography variant="h6" gutterBottom>
                  Fazer Pergunta ao VectorDB
                </Typography>

                {queryError && (
                  <Alert severity="error" sx={{ mb: 2 }}>
                    {queryError}
                  </Alert>
                )}

                <FormControl fullWidth sx={{ mb: 2 }}>
                  <InputLabel>Fonte de Dados</InputLabel>
                  <Select
                    value={dataSource}
                    label="Fonte de Dados"
                    onChange={handleDataSourceChange}
                  >
                    <MenuItem value="Todos">Todos os Dados</MenuItem>
                    <MenuItem value="regras_negocio">Regras de Negócio</MenuItem>
                    <MenuItem value="base_dados">Estrutura do Banco</MenuItem>
                    <MenuItem value="servicos">Serviços do Sistema</MenuItem>
                    <MenuItem value="rotinas_usuario">Rotinas do Usuário</MenuItem>
                  </Select>
                </FormControl>

                <TextField
                  fullWidth
                  multiline
                  rows={3}
                  variant="outlined"
                  label="Digite sua pergunta"
                  value={question}
                  onChange={(e) => setQuestion(e.target.value)}
                  placeholder="Ex: Como funciona o cadastro de clientes?"
                  sx={{ mb: 2 }}
                />

                <Button
                  variant="contained"
                  startIcon={loading ? <CircularProgress size={20} /> : <SendIcon />}
                  disabled={loading || !question.trim()}
                  onClick={handleQuery}
                  fullWidth
                >
                  {loading ? 'Consultando...' : 'Consultar'}
                </Button>
              </CardContent>
            </Card>

            {/* Resultados da Consulta */}
            {results.length > 0 && (
              <Card sx={{ mt: 3 }}>
                <CardContent>
                  <Typography variant="h6" gutterBottom>
                    📊 Resultados da Consulta ({results.length} resultado{results.length !== 1 ? 's' : ''})
                  </Typography>
                  
                  {results.map((result, index) => (
                    <Box key={index} sx={{ mb: 2, p: 2, border: '1px solid #e0e0e0', borderRadius: 1, backgroundColor: '#fafafa' }}>
                      <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'flex-start', mb: 1 }}>
                        <Box>
                          <Typography variant="subtitle2" color="primary" gutterBottom sx={{ mb: 0.5 }}>
                            Resultado {index + 1} - {result.type?.toUpperCase()}
                          </Typography>
                          {result.table && (
                            <Typography variant="body2" sx={{ color: '#1976d2', fontWeight: 'bold', mb: 1 }}>
                              📊 Tabela: <strong>{result.table}</strong>
                            </Typography>
                          )}
                        </Box>
                        <Chip 
                          label={`${(result.relevance || 0).toFixed(1)}% relevante`}
                          size="small"
                          color={(result.relevance || 0) >= 50 ? "success" : (result.relevance || 0) >= 20 ? "warning" : "error"}
                          variant="outlined"
                        />
                      </Box>
                      <Typography variant="body2" color="primary" sx={{ mb: 1 }}>
                        <strong>Tipo:</strong> <Chip color='success' label={result.type} size="small" variant="outlined" sx={{ ml: 1 }} />
                      </Typography>
                      <Typography variant="body2" color="primary">
                        {result.content}
                      </Typography>
                      {result.metadata && (
                        <Box sx={{ mt: 1 }}>
                          {result.metadata.source && (
                            <Typography variant="caption" display="block" color="primary">
                              <strong>Fonte:</strong> {result.metadata.source}
                            </Typography>
                          )}
                          {result.metadata.field_name && (
                            <Typography variant="caption" display="block" color="primary">
                              <strong>Campo:</strong> {result.metadata.field_name}
                            </Typography>
                          )}
                        </Box>
                      )}
                    </Box>
                  ))}
                </CardContent>
              </Card>
            )}
          </Box>
        )}

        {subTab === 1 && (
          <Box>
            {/* Upload de Arquivo */}
            <Card sx={{ mb: 3 }}>
              <CardContent>
                <Typography variant="h6" gutterBottom>
                  📁 Carregar Arquivo de Dados
                </Typography>

                <Typography variant="body2" color="text.secondary" sx={{ mb: 2 }}>
                  Envie arquivos YAML ou JSON. O sistema detecta automaticamente o tipo: regras de negócio, estrutura do banco, serviços ou rotinas do usuário.
                </Typography>


                {uploadError && (
                  <Alert severity="error" sx={{ mb: 2 }}>
                    {uploadError}
                  </Alert>
                )}
                
                {uploadSuccess && (
                  <Alert severity="success" sx={{ mb: 2 }} icon={<CheckCircleIcon />}>
                    {uploadSuccess}
                  </Alert>
                )}

                <Box
                  onDragOver={handleDragOver}
                  onDragLeave={handleDragLeave}
                  onDrop={handleDrop}
                  sx={{
                    mb: 3,
                    p: 2,
                    border: '2px dashed',
                    borderColor: isDragging ? 'primary.main' : '#90caf9',
                    borderRadius: 2,
                    backgroundColor: isDragging ? '#e3f2fd' : '#f5f9ff',
                    transition: 'background-color 0.2s, border-color 0.2s',
                    opacity: uploadingFiles ? 0.6 : 1,
                    pointerEvents: uploadingFiles ? 'none' : 'auto',
                  }}
                >
                  <input
                    type="file"
                    accept=".yaml,.yml,.json"
                    onChange={handleFileChange}
                    id="file-input"
                    multiple
                    disabled={uploadingFiles}
                    style={{ display: 'none' }}
                  />
                  <label htmlFor="file-input" style={{ width: '100%', cursor: uploadingFiles ? 'default' : 'pointer' }}>
                    <Box sx={{ textAlign: 'center', p: 2 }}>
                      <UploadFileIcon sx={{ fontSize: 44, color: isDragging ? 'primary.dark' : '#1976d2', mb: 1 }} />
                      <Typography variant="body1" color="primary" sx={{ fontWeight: 500 }}>
                        {isDragging ? 'Solte os arquivos aqui' : 'Clique para selecionar arquivos ou arraste aqui'}
                      </Typography>
                      <Typography variant="caption" color="text.secondary">
                        Formatos: YAML, YML ou JSON (múltiplos arquivos)
                      </Typography>
                    </Box>
                  </label>
                </Box>

                {uploadingFiles && uploadProgress.total > 0 && (
                  <Box sx={{ mb: 3 }}>
                    <Box sx={{ display: 'flex', justifyContent: 'space-between', mb: 0.5 }}>
                      <Typography variant="caption" color="text.secondary">
                        Processando arquivos…
                      </Typography>
                      <Typography variant="caption" color="text.secondary">
                        {uploadProgress.done} / {uploadProgress.total}
                      </Typography>
                    </Box>
                    <LinearProgress
                      variant="determinate"
                      value={uploadProgress.total > 0 ? (uploadProgress.done / uploadProgress.total) * 100 : 0}
                    />
                  </Box>
                )}

                {selectedFiles.length > 0 && (
                  <Box sx={{ mb: 3 }}>
                    <Typography variant="subtitle2" sx={{ mb: 1 }}>
                      {selectedFiles.length} arquivo(s) selecionado(s)
                    </Typography>
                    <List dense disablePadding sx={{ border: '1px solid #e0e0e0', borderRadius: 1 }}>
                      {selectedFiles.map((file, idx) => {
                        const status = fileStatus[file.name];
                        return (
                          <ListItem
                            key={idx}
                            divider={idx < selectedFiles.length - 1}
                            secondaryAction={
                              uploadingFiles ? undefined : (
                                <IconButton edge="end" size="small" onClick={() => handleRemoveFile(file.name)} aria-label="remover">
                                  <DeleteIcon fontSize="small" />
                                </IconButton>
                              )
                            }
                          >
                            <ListItemIcon sx={{ minWidth: 40 }}>
                              {status?.state === 'processing' && <CircularProgress size={20} />}
                              {status?.state === 'success' && <CheckCircleIcon color="success" fontSize="small" />}
                              {status?.state === 'error' && <ErrorIcon color="error" fontSize="small" />}
                              {(!status || status.state === 'pending') && <InsertDriveFileIcon color="action" fontSize="small" />}
                            </ListItemIcon>
                            <ListItemText
                              primary={file.name}
                              secondary={
                                status?.state === 'error'
                                  ? status.message
                                  : status?.state === 'success'
                                    ? `Importado como ${status.type}`
                                    : status?.state === 'processing'
                                      ? 'Processando…'
                                      : `${(file.size / 1024).toFixed(2)} KB`
                              }
                              slotProps={{
                                primary: { variant: 'body2', sx: { fontWeight: 500 } },
                                secondary: {
                                  variant: 'caption',
                                  color: status?.state === 'error' ? 'error' : 'text.secondary',
                                },
                              }}
                            />
                          </ListItem>
                        );
                      })}
                    </List>
                  </Box>
                )}

                <Button
                  variant="contained"
                  startIcon={uploadingFiles ? <CircularProgress size={20} color="inherit" /> : <SendIcon />}
                  disabled={uploadingFiles || selectedFiles.length === 0}
                  onClick={handleUploadFile}
                  fullWidth
                  size="large"
                >
                  {uploadingFiles
                    ? `Enviando ${uploadProgress.done}/${uploadProgress.total}…`
                    : `Enviar ${selectedFiles.length > 0 ? selectedFiles.length : '0'} arquivo(s) para o ChromaDB`}
                </Button>
              </CardContent>
            </Card>

            {/* Informações sobre tipos de arquivo */}
            {fileTypes && (
              <Card sx={{ mb: 3 }}>
                <CardContent>
                  <Typography variant="h6" gutterBottom>
                    ℹ️ Tipos de Arquivo Aceitos
                  </Typography>
                  <Typography variant="body2" color="text.secondary" sx={{ mb: 2 }}>
                    O sistema detecta automaticamente o tipo baseado no conteúdo:
                  </Typography>
                  <Box sx={{ display: 'grid', gridTemplateColumns: { xs: '1fr', md: '1fr 1fr' }, gap: 2 }}>
                    {Object.entries(fileTypes.types || {}).map(([key, value]: [string, any]) => (
                      <Box key={key} sx={{ p: 1.5, border: '1px solid #e0e0e0', borderRadius: 1 }}>
                        <Typography variant="subtitle2" color="primary">
                          {key}
                        </Typography>
                        <Typography variant="body2" sx={{ mb: 1 }}>
                          {value.description}
                        </Typography>
                        <Typography variant="caption" color="text.secondary">
                          Palavras-chave: {value.keywords.join(', ')}
                        </Typography>
                      </Box>
                    ))}
                  </Box>
                </CardContent>
              </Card>
            )}

            {/* Limpar Base de Dados */}
            <Card sx={{ mb: 3 }}>
              <CardContent>
                <Typography variant="h6" gutterBottom color="error">
                  ⚠️ Limpar Base de Dados
                </Typography>

                <Typography variant="body2" color="text.secondary" sx={{ mb: 2 }}>
                  Esta ação irá remover todos os documentos da base de dados do ChromaDB.
                </Typography>

                <Button
                  variant="outlined"
                  color="error"
                  startIcon={<DeleteIcon />}
                  onClick={handleClearDatabase}
                >
                  Limpar Base de Dados
                </Button>
                {clearError && (
                  <Alert severity="error" sx={{ mt: 2 }}>
                    {clearError}
                  </Alert>
                )}
                {clearMessage && (
                  <Alert severity="success" sx={{ mt: 2 }}>
                    {clearMessage}
                  </Alert>
                )}
              </CardContent>
            </Card>
          </Box>
        )}
      </Box>

      <Dialog
        open={confirmDialogOpen}
        onClose={handleCancelClear}
        aria-labelledby="confirm-dialog-title"
        aria-describedby="confirm-dialog-description"
      >
        <DialogTitle id="confirm-dialog-title">
          Confirmar Limpeza da Base de Dados
        </DialogTitle>
        <DialogContent>
          <DialogContentText id="confirm-dialog-description">
            Tem certeza de que deseja limpar toda a base de dados? Esta ação não pode ser desfeita e todos os documentos serão removidos permanentemente.
          </DialogContentText>
        </DialogContent>
        <DialogActions>
          <Button onClick={handleCancelClear} color="primary">
            Cancelar
          </Button>
          <Button onClick={handleConfirmClear} color="error" variant="contained">
            Confirmar Limpeza
          </Button>
        </DialogActions>
      </Dialog>
    </Box>
  );
};

export default VectorDBTest;