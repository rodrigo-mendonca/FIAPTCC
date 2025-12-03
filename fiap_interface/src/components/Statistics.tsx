import React, { useState, useEffect } from 'react';
import {
  Box,
  Card,
  CardContent,
  Typography,
  Button,
  Chip,
  Alert
} from '@mui/material';
import {
  Refresh as RefreshIcon,
  Assessment as AssessmentIcon,
  Storage as StorageIcon,
  CheckCircle as CheckCircleIcon
} from '@mui/icons-material';
import { useCollection } from '../contexts/CollectionContext';

interface VectorDBStats {
  total_documentos: number;
  collection_name: string;
  embedding_model: string;
  last_updated: string;
  collections: Array<{
    name: string;
    count: number;
    id: string;
  }>;
}

const Statistics: React.FC = () => {
  const [stats, setStats] = useState<VectorDBStats | null>(null);
  const [loading, setLoading] = useState(false);
  const [lastUpdated, setLastUpdated] = useState<Date | null>(null);
  const [error, setError] = useState<string | null>(null);
  const { selectedCollection } = useCollection();
  const API_URL = process.env.REACT_APP_API_URL;
  const CHROMADB_URL = process.env.REACT_APP_CHROMADB_URL || 'localhost:8200';

  // Carrega estatísticas automaticamente
  useEffect(() => {
    loadStats();
  }, [selectedCollection]);

  // Função para carregar estatísticas
  const loadStats = async () => {
    setLoading(true);
    setError(null);
    try {
      console.log('Fetching stats from: ' + process.env.REACT_APP_API_URL + '/vectordb/stats?collection_name=' + selectedCollection);
      const response = await fetch(process.env.REACT_APP_API_URL + '/vectordb/stats?collection_name=' + encodeURIComponent(selectedCollection));
      if (response.ok) {
        const data = await response.json();
        console.log('Stats data received:', data);
        setStats(data);
        setLastUpdated(new Date());
      } else {
        const errorMsg = `API Error: ${response.status} ${response.statusText}`;
        console.error('Failed to fetch stats:', response.status, response.statusText);
        setError(errorMsg);
        setStats(null);
      }
    } catch (err) {
      const errorMsg = err instanceof Error ? err.message : 'Erro de conexão';
      console.error('Erro ao carregar estatísticas:', err);
      setError(errorMsg);
      setStats(null);
    } finally {
      setLoading(false);
    }
  };

  return (
    <Box sx={{ width: '100%' }}>
      <Box sx={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', mb: 3 }}>
        <Box sx={{ display: 'flex', alignItems: 'center', gap: 2 }}>
          <Typography variant="h4" gutterBottom sx={{ mb: 0 }}>
            📊 Estatísticas do Sistema
          </Typography>
        </Box>
        <Box sx={{ display: 'flex', alignItems: 'center', gap: 2 }}>
          <Button
            variant="outlined"
            startIcon={<RefreshIcon />}
            onClick={() => loadStats()}
            disabled={loading}
            sx={{ minWidth: 120 }}
          >
            {loading ? 'Carregando...' : 'Atualizar'}
          </Button>
        </Box>
      </Box>
      {lastUpdated && (
        <Box sx={{ mb: 3 }}>
          <Chip
            label={`Última atualização: ${lastUpdated.toLocaleTimeString()}`}
            size="small"
            color="primary"
            variant="outlined"
          />
        </Box>
      )}

      {error && (
        <Alert severity="error" sx={{ mb: 3 }}>
          <Typography variant="body2">
            <strong>Erro ao carregar estatísticas:</strong> {error}
          </Typography>
          <Typography variant="caption" display="block" sx={{ mt: 1 }}>
            Verifique se a API está rodando em {API_URL?.replace('http://', '').replace('https://', '') || 'localhost:8000'}
          </Typography>
        </Alert>
      )}

      {/* Status dos Serviços e Configurações - Juntos */}
      <Card sx={{ mb: 3 }}>
        <CardContent>
          <Typography variant="h6" gutterBottom>
            🔍 Status dos Serviços e Configurações
          </Typography>

          <Typography variant="body2" color="text.secondary" paragraph>
            Monitore o status dos componentes principais e visualize as configurações do sistema
          </Typography>

          <Box sx={{ display: 'flex', flexDirection: 'column', gap: 3 }}>
            {/* Status dos Serviços */}
            <Box>
              <Typography variant="subtitle1" gutterBottom sx={{ fontWeight: 'bold' }}>
                Status dos Serviços
              </Typography>
              <Box sx={{ display: 'flex', flexWrap: 'wrap', gap: 2 }}>
                <Chip
                  icon={<CheckCircleIcon />}
                  label="ChromaDB"
                  color="success"
                  variant="filled"
                  size="medium"
                  sx={{
                    minWidth: 140,
                    py: 2,
                    fontSize: '1rem',
                    fontWeight: 'bold',
                    '& .MuiChip-icon': { fontSize: '1.2rem' }
                  }}
                />
                <Chip
                  icon={<CheckCircleIcon />}
                  label="LMStudio"
                  color="success"
                  variant="filled"
                  size="medium"
                  sx={{
                    minWidth: 140,
                    py: 2,
                    fontSize: '1rem',
                    fontWeight: 'bold',
                    '& .MuiChip-icon': { fontSize: '1.2rem' }
                  }}
                />
                <Chip
                  icon={<CheckCircleIcon />}
                  label="API Python"
                  color="success"
                  variant="filled"
                  size="medium"
                  sx={{
                    minWidth: 140,
                    py: 2,
                    fontSize: '1rem',
                    fontWeight: 'bold',
                    '& .MuiChip-icon': { fontSize: '1.2rem' }
                  }}
                />
              </Box>
            </Box>

            {/* Configurações do Sistema */}
            <Box>
              <Typography variant="subtitle1" gutterBottom sx={{ fontWeight: 'bold' }}>
                Configurações do Sistema
              </Typography>
              <Box sx={{ display: 'grid', gridTemplateColumns: { xs: '1fr', md: '1fr 1fr' }, gap: 2 }}>
                <Box>
                  <Typography variant="body2" color="text.secondary">
                    ChromaDB
                  </Typography>
                  <Typography variant="body1" fontWeight="medium">
                    {CHROMADB_URL}
                  </Typography>
                </Box>

                <Box>
                  <Typography variant="body2" color="text.secondary">
                    LMStudio
                  </Typography>
                  <Typography variant="body1" fontWeight="medium">
                    192.168.50.30:1234
                  </Typography>
                </Box>

                <Box>
                  <Typography variant="body2" color="text.secondary">
                    Modelo de Embedding
                  </Typography>
                  <Typography variant="body1" fontWeight="medium">
                    text-embedding-nomic-embed-text-v1.5
                  </Typography>
                </Box>

                <Box>
                  <Typography variant="body2" color="text.secondary">
                    API Python
                  </Typography>
                  <Typography variant="body1" fontWeight="medium">
                    {API_URL?.replace('http://', '').replace('https://', '') || 'localhost:8000'}
                  </Typography>
                </Box>
              </Box>
            </Box>
          </Box>
        </CardContent>
      </Card>

      <Box sx={{ display: 'flex', flexDirection: 'column', gap: 3 }}>
        {/* Estatísticas da Coleção VectorDB e Lista de Coleções */}
        <Box sx={{ display: 'flex', flexDirection: { xs: 'column', md: 'row' }, gap: 3 }}>
          <Box sx={{ flex: 1 }}>
            <Card sx={{ height: '100%' }}>
              <CardContent>
                <Box sx={{ display: 'flex', alignItems: 'center', gap: 1, mb: 2 }}>
                  <StorageIcon color="primary" />
                  <Typography variant="h6">
                    ChromaDB - {selectedCollection}
                  </Typography>
                </Box>

                {stats ? (
                  <Box sx={{ display: 'flex', flexDirection: 'column', gap: 2 }}>
                    <Box>
                      <Typography variant="body2" color="text.secondary">
                        Total de Documentos
                      </Typography>
                      <Typography variant="h4" color="primary.main" fontWeight="bold">
                        {stats.total_documentos != null ? stats.total_documentos.toLocaleString() : 'N/A'}
                      </Typography>
                    </Box>

                    <Box>
                      <Typography variant="body2" color="text.secondary">
                        Modelo de Embedding
                      </Typography>
                      <Typography variant="body1" fontWeight="medium">
                        {stats.embedding_model || 'N/A'}
                      </Typography>
                    </Box>

                    <Box>
                      <Typography variant="body2" color="text.secondary">
                        Última Atualização
                      </Typography>
                      <Typography variant="body1" fontWeight="medium">
                        {stats.last_updated || 'N/A'}
                      </Typography>
                    </Box>
                  </Box>
                ) : (
                  <Typography variant="body2" color="text.secondary">
                    {loading ? 'Carregando estatísticas...' : 'Dados não disponíveis'}
                  </Typography>
                )}
              </CardContent>
            </Card>
          </Box>

          {/* Lista de Coleções */}
          <Box sx={{ flex: 1 }}>
            <Card sx={{ height: '100%' }}>
              <CardContent>
                <Box sx={{ display: 'flex', alignItems: 'center', gap: 1, mb: 2 }}>
                  <AssessmentIcon color="primary" />
                  <Typography variant="h6">
                    Todas as Coleções
                  </Typography>
                </Box>

                {stats && stats.collections ? (
                  <Box sx={{ display: 'flex', flexDirection: 'column', gap: 1 }}>
                    {stats.collections.map((collection, index) => (
                      <Box key={collection.id || index} sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', p: 1, border: '1px solid', borderColor: 'divider', borderRadius: 1 }}>
                        <Typography variant="body2" fontWeight="medium">
                          {collection.name}
                        </Typography>
                        <Chip
                          label={`${collection.count} docs`}
                          size="small"
                          color={collection.name === stats.collection_name ? 'primary' : 'default'}
                          variant={collection.name === stats.collection_name ? 'filled' : 'outlined'}
                        />
                      </Box>
                    ))}
                    {stats.collections.length === 0 && (
                      <Typography variant="body2" color="text.secondary" sx={{ textAlign: 'center', py: 2 }}>
                        Nenhuma coleção encontrada
                      </Typography>
                    )}
                  </Box>
                ) : (
                  <Typography variant="body2" color="text.secondary" sx={{ textAlign: 'center', py: 2 }}>
                    Carregando coleções...
                  </Typography>
                )}
              </CardContent>
            </Card>
          </Box>
        </Box>
      </Box>
    </Box>
  );
};

export default Statistics;