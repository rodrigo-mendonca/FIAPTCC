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
  Tab
} from '@mui/material';
import {
  Send as SendIcon,
  Delete as DeleteIcon,
  Search as SearchIcon,
  Settings as SettingsIcon
} from '@mui/icons-material';
import React, { useState } from 'react';
import { useCollection } from '../contexts/CollectionContext';
const VectorDBTest: React.FC = () => {
  const [question, setQuestion] = useState('');
  const [loading, setLoading] = useState(false);
  const [subTab, setSubTab] = useState(0);
  const [dataSource, setDataSource] = useState('Todos');
  const [confirmDialogOpen, setConfirmDialogOpen] = useState(false);
  const [businessRulesFile, setBusinessRulesFile] = useState<File | null>(null);
  const [databaseStructureFile, setDatabaseStructureFile] = useState<File | null>(null);
  const [systemServicesFile, setSystemServicesFile] = useState<File | null>(null);
  const [uploading, setUploading] = useState(false);
  const [results, setResults] = useState<any[]>([]);
  const { selectedCollection } = useCollection();

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

    try {
      const response = await fetch(`${process.env.REACT_APP_API_URL}/vectordb/query?collection_name=${encodeURIComponent(selectedCollection)}`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          question: question,
          n_results: 5,
          context: dataSource === 'Todos' ? 'all' : dataSource
        }),
      });

      if (response.ok) {
        const data = await response.json();
        setResults(data.results || []);
      } else {
        alert('Erro na consulta: ' + response.statusText);
      }
    } catch (error) {
      console.error('Erro:', error);
      alert('Erro ao fazer consulta');
    } finally {
      setLoading(false);
    }
  };

  const handleClearDatabase = () => {
    setConfirmDialogOpen(true);
  };

  const handleConfirmClear = () => {
    // TODO: Implement actual database clearing logic
    console.log('Database cleared');
    setConfirmDialogOpen(false);
  };

  const handleCancelClear = () => {
    setConfirmDialogOpen(false);
  };

  const handleBusinessRulesFileChange = (event: React.ChangeEvent<HTMLInputElement>) => {
    const file = event.target.files?.[0] || null;
    setBusinessRulesFile(file);
  };

  const handleDatabaseStructureFileChange = (event: React.ChangeEvent<HTMLInputElement>) => {
    const file = event.target.files?.[0] || null;
    setDatabaseStructureFile(file);
  };

  const handleSystemServicesFileChange = (event: React.ChangeEvent<HTMLInputElement>) => {
    const file = event.target.files?.[0] || null;
    setSystemServicesFile(file);
  };

  const handleUploadFiles = async () => {
    const filesToUpload = [
      { file: businessRulesFile, type: 'business_rules' },
      { file: databaseStructureFile, type: 'database_struct' },
      { file: systemServicesFile, type: 'system_services' }
    ].filter(item => item.file !== null);

    if (filesToUpload.length === 0) {
      alert('Selecione pelo menos um arquivo para enviar.');
      return;
    }

    setUploading(true);

    try {
      for (const { file, type } of filesToUpload) {
        if (!file) continue;

        const formData = new FormData();
        formData.append('file', file);
        formData.append('type', type);

        const response = await fetch(`${process.env.REACT_APP_API_URL}/vectordb/upload`, {
          method: 'POST',
          body: formData,
        });

        if (!response.ok) {
          throw new Error(`Erro ao enviar ${type}: ${response.statusText}`);
        }

        const result = await response.json();
        console.log(`Arquivo ${type} enviado com sucesso:`, result);
      }

      alert('Arquivos enviados com sucesso para o ChromaDB!');
      
      // Limpar os arquivos selecionados após o upload
      setBusinessRulesFile(null);
      setDatabaseStructureFile(null);
      setSystemServicesFile(null);
      
      // Resetar os inputs de arquivo
      const inputs = document.querySelectorAll('input[type="file"]') as NodeListOf<HTMLInputElement>;
      inputs.forEach(input => input.value = '');

    } catch (error) {
      console.error('Erro ao enviar arquivos:', error);
      alert('Erro ao enviar arquivos. Verifique o console para mais detalhes.');
    } finally {
      setUploading(false);
    }
  };

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

                <FormControl fullWidth sx={{ mb: 2 }}>
                  <InputLabel>Fonte de Dados</InputLabel>
                  <Select
                    value={dataSource}
                    label="Fonte de Dados"
                    onChange={handleDataSourceChange}
                  >
                    <MenuItem value="Todos">Todos</MenuItem>
                    <MenuItem value="business_rules">Regras de Negócio</MenuItem>
                    <MenuItem value="database_structure">Estrutura do Banco</MenuItem>
                    <MenuItem value="system_services">Serviços do Sistema</MenuItem>
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
                    Resultados da Consulta
                  </Typography>
                  
                  {results.map((result, index) => (
                    <Box key={index} sx={{ mb: 2, p: 2, border: '1px solid #e0e0e0', borderRadius: 1 }}>
                      <Typography variant="subtitle2" color="primary" gutterBottom>
                        Resultado {index + 1} (Relevância: {(result.similarity * 100).toFixed(1)}%)
                      </Typography>
                      <Typography variant="body2" sx={{ mb: 1 }}>
                        <strong>Tipo:</strong> {result.type}
                      </Typography>
                      <Typography variant="body2">
                        {result.content}
                      </Typography>
                      {result.metadata && (
                        <Box sx={{ mt: 1 }}>
                          {result.metadata.table_name && (
                            <Typography variant="caption" display="block">
                              <strong>Tabela:</strong> {result.metadata.table_name}
                            </Typography>
                          )}
                          {result.metadata.source && (
                            <Typography variant="caption" display="block">
                              <strong>Fonte:</strong> {result.metadata.source}
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
            <Card sx={{ mb: 3 }}>
              <CardContent>
                <Typography variant="h6" gutterBottom color="error">
                  ⚠️ Limpar Base de Dados
                </Typography>

                <Typography variant="body2" color="text.secondary" sx={{ mb: 2 }}>
                  Esta ação irá remover todos os documentos da base de dados.
                </Typography>

                <Button
                  variant="outlined"
                  color="error"
                  startIcon={<DeleteIcon />}
                  onClick={handleClearDatabase}
                >
                  Limpar Base de Dados
                </Button>
              </CardContent>
            </Card>

            <Card>
              <CardContent>
                <Typography variant="h6" gutterBottom>
                  📁 Carregar Arquivos JSON
                </Typography>

                <Typography variant="body2" color="text.secondary" sx={{ mb: 2 }}>
                  Selecione arquivos JSON para adicionar à base:
                </Typography>

                <Box sx={{ mb: 2 }}>
                  <Typography variant="subtitle2" gutterBottom>
                    Regras de Negócio
                  </Typography>
                  <input 
                    type="file" 
                    accept=".json" 
                    onChange={handleBusinessRulesFileChange}
                  />
                  {businessRulesFile && (
                    <Typography variant="caption" color="primary" sx={{ ml: 1 }}>
                      {businessRulesFile.name}
                    </Typography>
                  )}
                </Box>

                <Box sx={{ mb: 2 }}>
                  <Typography variant="subtitle2" gutterBottom>
                    Estrutura do Banco
                  </Typography>
                  <input 
                    type="file" 
                    accept=".json" 
                    onChange={handleDatabaseStructureFileChange}
                  />
                  {databaseStructureFile && (
                    <Typography variant="caption" color="primary" sx={{ ml: 1 }}>
                      {databaseStructureFile.name}
                    </Typography>
                  )}
                </Box>

                <Box sx={{ mb: 2 }}>
                  <Typography variant="subtitle2" gutterBottom>
                    Serviços do Sistema
                  </Typography>
                  <input 
                    type="file" 
                    accept=".json" 
                    onChange={handleSystemServicesFileChange}
                  />
                  {systemServicesFile && (
                    <Typography variant="caption" color="primary" sx={{ ml: 1 }}>
                      {systemServicesFile.name}
                    </Typography>
                  )}
                </Box>

                <Button
                  variant="contained"
                  startIcon={uploading ? <CircularProgress size={20} /> : <SendIcon />}
                  disabled={uploading || (!businessRulesFile && !databaseStructureFile && !systemServicesFile)}
                  onClick={handleUploadFiles}
                  fullWidth
                  sx={{ mt: 2 }}
                >
                  {uploading ? 'Enviando...' : 'Enviar Arquivos para ChromaDB'}
                </Button>
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