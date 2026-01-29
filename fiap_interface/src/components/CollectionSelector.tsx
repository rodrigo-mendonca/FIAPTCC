import React, { useState } from 'react';
import {
  Box,
  FormControl,
  InputLabel,
  Select,
  MenuItem,
  SelectChangeEvent,
  Chip,
  Typography,
  Paper,
  Dialog,
  DialogTitle,
  DialogContent,
  DialogActions,
  TextField,
  Button,
  Alert,
  IconButton
} from '@mui/material';
import { Storage as StorageIcon, Add as AddIcon, Delete as DeleteIcon } from '@mui/icons-material';
import { useCollection } from '../contexts/CollectionContext';
import { useNotification } from '../contexts/NotificationContext';

const CollectionSelector: React.FC = () => {
  const { selectedCollection, setSelectedCollection, availableCollections, createCollection, deleteCollection } = useCollection();
  const { showNotification } = useNotification();
  const [openDialog, setOpenDialog] = useState(false);
  const [openDeleteDialog, setOpenDeleteDialog] = useState(false);
  const [newCollectionName, setNewCollectionName] = useState('');
  const [isCreating, setIsCreating] = useState(false);
  const [isDeleting, setIsDeleting] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [collectionToDelete, setCollectionToDelete] = useState<string | null>(null);

  const handleCollectionChange = (event: SelectChangeEvent<string>) => {
    const newCollection = event.target.value;
    
    if (newCollection === '__NEW_COLLECTION__') {
      setOpenDialog(true);
      return;
    }
    
    setSelectedCollection(newCollection);
  };

  const handleCreateCollection = async () => {
    if (!newCollectionName.trim()) {
      setError('Nome da coleção é obrigatório');
      showNotification('Nome da coleção é obrigatório', 'warning');
      return;
    }

    if (newCollectionName.trim().length < 3) {
      setError('Nome da coleção deve ter pelo menos 3 caracteres');
      showNotification('Nome da coleção deve ter pelo menos 3 caracteres', 'warning');
      return;
    }

    setIsCreating(true);
    setError(null);

    try {
      await createCollection(newCollectionName.trim());
      showNotification(`Coleção "${newCollectionName.trim()}" criada com sucesso!`, 'success');
      setOpenDialog(false);
      setNewCollectionName('');
    } catch (err) {
      const errorMsg = err instanceof Error ? err.message : 'Erro ao criar coleção';
      setError(errorMsg);
      showNotification(errorMsg, 'error');
    } finally {
      setIsCreating(false);
    }
  };

  const handleDialogClose = () => {
    setOpenDialog(false);
    setNewCollectionName('');
    setError(null);
  };

  const handleDeleteClick = () => {
    // Allow deletion even if only one collection remains
    setCollectionToDelete(selectedCollection);
    setOpenDeleteDialog(true);
  };

  const handleConfirmDelete = async () => {
    if (!collectionToDelete) return;

    setIsDeleting(true);
    try {
      await deleteCollection(collectionToDelete);
      showNotification(`Coleção "${collectionToDelete}" deletada com sucesso!`, 'success');
      setOpenDeleteDialog(false);
      setCollectionToDelete(null);
    } catch (err) {
      const errorMsg = err instanceof Error ? err.message : 'Erro ao deletar coleção';
      showNotification(errorMsg, 'error');
    } finally {
      setIsDeleting(false);
    }
  };

  const handleDeleteDialogClose = () => {
    setOpenDeleteDialog(false);
    setCollectionToDelete(null);
  };

  return (
    <Paper sx={{ p: 2, mb: 3, background: 'linear-gradient(135deg, #667eea 0%, #764ba2 100%)', color: 'white' }}>
      <Box sx={{ display: 'flex', alignItems: 'center', gap: 2, flexWrap: 'wrap' }}>
        <Box sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
          <StorageIcon />
          <Typography variant="h6">
            Coleção Ativa
          </Typography>
        </Box>

        <FormControl size="small" sx={{ minWidth: 200 }}>
          <InputLabel sx={{ color: 'rgba(255,255,255,0.8)' }}>Selecionar Coleção</InputLabel>
          <Select
            value={selectedCollection}
            label="Selecionar Coleção"
            onChange={handleCollectionChange}
            sx={{
              color: 'white',
              '& .MuiOutlinedInput-notchedOutline': {
                borderColor: 'rgba(255,255,255,0.3)',
              },
              '&:hover .MuiOutlinedInput-notchedOutline': {
                borderColor: 'rgba(255,255,255,0.5)',
              },
              '&.Mui-focused .MuiOutlinedInput-notchedOutline': {
                borderColor: 'white',
              },
            }}
          >
            {availableCollections && availableCollections.length > 0 ? (
              availableCollections.map((collection) => (
                <MenuItem key={collection.id} value={collection.name}>
                  {collection.name} ({collection.count} docs)
                </MenuItem>
              ))
            ) : (
              <MenuItem value="" disabled>
                Nenhuma coleção disponível
              </MenuItem>
            )}
            <MenuItem value="__NEW_COLLECTION__" sx={{ borderTop: '1px solid #ddd', pt: 1, mt: 1 }}>
              <Box sx={{ display: 'flex', alignItems: 'center', gap: 1, color: 'primary.main', fontWeight: 'bold' }}>
                <AddIcon fontSize="small" />
                Nova Coleção
              </Box>
            </MenuItem>
          </Select>
        </FormControl>

        {/* Chip with collection name removed as per UX request */}

        <IconButton
          onClick={handleDeleteClick}
          sx={{
            color: 'rgba(255,255,255,0.8)',
            '&:hover': {
              color: 'white',
              backgroundColor: 'rgba(255,0,0,0.1)',
            },
          }}
          title={'Deletar coleção'}
        >
          <DeleteIcon />
        </IconButton>
      </Box>

      {/* Dialog para criar nova coleção */}
      <Dialog open={openDialog} onClose={handleDialogClose} maxWidth="sm" fullWidth>
        <DialogTitle>Criar Nova Coleção</DialogTitle>
        <DialogContent sx={{ pt: 2 }}>
          {error && (
            <Alert severity="error" sx={{ mb: 2 }}>
              {error}
            </Alert>
          )}
          <TextField
            autoFocus
            fullWidth
            label="Nome da Coleção"
            placeholder="Ex: minha_colecao"
            value={newCollectionName}
            onChange={(e) => {
              setNewCollectionName(e.target.value);
              setError(null);
            }}
            onKeyPress={(e) => {
              if (e.key === 'Enter' && !isCreating) {
                handleCreateCollection();
              }
            }}
            disabled={isCreating}
            helperText="Use apenas letras, números e underscore. Mínimo 3 caracteres."
            sx={{ mt: 1 }}
          />
        </DialogContent>
        <DialogActions>
          <Button onClick={handleDialogClose} disabled={isCreating}>
            Cancelar
          </Button>
          <Button 
            onClick={handleCreateCollection} 
            variant="contained" 
            disabled={isCreating || !newCollectionName.trim()}
          >
            {isCreating ? 'Criando...' : 'Criar'}
          </Button>
        </DialogActions>
      </Dialog>

      {/* Dialog de confirmação de deleção */}
      <Dialog open={openDeleteDialog} onClose={handleDeleteDialogClose} maxWidth="sm" fullWidth>
        <DialogTitle>Deletar Coleção</DialogTitle>
        <DialogContent sx={{ pt: 2 }}>
          <Typography variant="body1" gutterBottom>
            Tem certeza que deseja deletar a coleção <strong>{collectionToDelete}</strong>?
          </Typography>
          <Alert severity="warning" sx={{ mt: 2 }}>
            Esta ação é irreversível e todos os documentos serão perdidos.
          </Alert>
        </DialogContent>
        <DialogActions>
          <Button onClick={handleDeleteDialogClose} disabled={isDeleting}>
            Cancelar
          </Button>
          <Button 
            onClick={handleConfirmDelete} 
            variant="contained" 
            color="error"
            disabled={isDeleting}
          >
            {isDeleting ? 'Deletando...' : 'Deletar'}
          </Button>
        </DialogActions>
      </Dialog>
    </Paper>
  );
};

export default CollectionSelector;