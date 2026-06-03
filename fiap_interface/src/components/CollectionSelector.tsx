import React, { useState } from 'react';
import {
  Box,
  FormControl,
  Select,
  MenuItem,
  SelectChangeEvent,
  Typography,
  Paper,
  Dialog,
  DialogTitle,
  DialogContent,
  DialogActions,
  TextField,
  Button,
  Alert,
  IconButton,
} from '@mui/material';
import { Add as AddIcon, Delete as DeleteIcon } from '@mui/icons-material';
import { useCollection } from '../contexts/CollectionContext';
import { useNotification } from '../contexts/NotificationContext';

interface Props {
  compact?: boolean;
}

const CollectionSelector: React.FC<Props> = ({ compact = false }) => {
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
    const val = event.target.value;
    if (val === '__NEW_COLLECTION__') { setOpenDialog(true); return; }
    setSelectedCollection(val);
  };

  const handleCreateCollection = async () => {
    if (!newCollectionName.trim()) { setError('Nome da coleção é obrigatório'); return; }
    if (newCollectionName.trim().length < 3) { setError('Nome deve ter pelo menos 3 caracteres'); return; }
    setIsCreating(true); setError(null);
    try {
      await createCollection(newCollectionName.trim());
      showNotification(`Coleção "${newCollectionName.trim()}" criada com sucesso!`, 'success');
      setOpenDialog(false); setNewCollectionName('');
    } catch (err) {
      const msg = err instanceof Error ? err.message : 'Erro ao criar coleção';
      setError(msg); showNotification(msg, 'error');
    } finally { setIsCreating(false); }
  };

  const handleDialogClose = () => { setOpenDialog(false); setNewCollectionName(''); setError(null); };

  const handleDeleteClick = (target: string) => { setCollectionToDelete(target); setOpenDeleteDialog(true); };

  const handleConfirmDelete = async () => {
    if (!collectionToDelete) return;
    setIsDeleting(true);
    try {
      await deleteCollection(collectionToDelete);
      showNotification(`Coleção "${collectionToDelete}" deletada com sucesso!`, 'success');
      setOpenDeleteDialog(false); setCollectionToDelete(null);
    } catch (err) {
      const msg = err instanceof Error ? err.message : 'Erro ao deletar coleção';
      showNotification(msg, 'error');
    } finally { setIsDeleting(false); }
  };

  const handleDeleteDialogClose = () => { setOpenDeleteDialog(false); setCollectionToDelete(null); };

  // Dialog para criar coleção (shared)
  const createDialog = (
    <Dialog open={openDialog} onClose={handleDialogClose} maxWidth="sm" fullWidth>
      <DialogTitle>Criar Nova Fonte</DialogTitle>
      <DialogContent sx={{ pt: 2 }}>
        {error && <Alert severity="error" sx={{ mb: 2 }}>{error}</Alert>}
        <TextField
          autoFocus fullWidth label="Nome da Fonte" placeholder="Ex: minha_fonte"
          value={newCollectionName}
          onChange={(e) => { setNewCollectionName(e.target.value); setError(null); }}
          onKeyDown={(e) => { if (e.key === 'Enter' && !isCreating) handleCreateCollection(); }}
          disabled={isCreating}
          helperText="Use apenas letras, números e underscore. Mínimo 3 caracteres."
          sx={{ mt: 1 }}
        />
      </DialogContent>
      <DialogActions>
        <Button onClick={handleDialogClose} disabled={isCreating}>Cancelar</Button>
        <Button onClick={handleCreateCollection} variant="contained" disabled={isCreating || !newCollectionName.trim()}>
          {isCreating ? 'Criando...' : 'Criar'}
        </Button>
      </DialogActions>
    </Dialog>
  );

  // Dialog de confirmação de deleção (shared)
  const deleteDialog = (
    <Dialog open={openDeleteDialog} onClose={handleDeleteDialogClose} maxWidth="sm" fullWidth>
      <DialogTitle>Deletar Fonte</DialogTitle>
      <DialogContent sx={{ pt: 2 }}>
        <Typography gutterBottom>Tem certeza que deseja deletar a fonte <strong>{collectionToDelete}</strong>?</Typography>
        <Alert severity="warning" sx={{ mt: 2 }}>Esta ação é irreversível e todos os documentos serão perdidos.</Alert>
      </DialogContent>
      <DialogActions>
        <Button onClick={handleDeleteDialogClose} disabled={isDeleting}>Cancelar</Button>
        <Button onClick={handleConfirmDelete} variant="contained" color="error" disabled={isDeleting}>
          {isDeleting ? 'Deletando...' : 'Deletar'}
        </Button>
      </DialogActions>
    </Dialog>
  );

  // ── Compact: só o Select (sem criar/deletar), para usar no header do chat ────
  if (compact) {
    return (
      <Box sx={{ display: 'flex', flexDirection: 'column', gap: 0.25 }}>
        <Typography sx={{ color: 'rgba(255,255,255,.5)', fontSize: '.62rem', letterSpacing: '.08em', textTransform: 'uppercase', lineHeight: 1 }}>
          Fonte
        </Typography>
        <FormControl size="small">
        <Select
          value={selectedCollection || ''}
          onChange={(e) => setSelectedCollection(e.target.value)}
          displayEmpty
          sx={{
            color: 'white',
            height: 38,
            fontSize: '.85rem',
            minWidth: 190,
            '& fieldset': { borderColor: 'rgba(255,255,255,.35)' },
            '&:hover fieldset': { borderColor: 'rgba(255,255,255,.65)' },
            '&.Mui-focused fieldset': { borderColor: 'white' },
            '& .MuiSelect-icon': { color: 'rgba(255,255,255,.75)' },
            '& .MuiSelect-select': { py: '5px' },
          }}
        >
          {availableCollections && availableCollections.length > 0
            ? availableCollections.map((c) => (
                <MenuItem key={c.id} value={c.name}>{c.name}</MenuItem>
              ))
            : <MenuItem value="" disabled>Carregando...</MenuItem>}
        </Select>
        </FormControl>
      </Box>
    );
  }

  // ── Default: Paper com gradiente do site, inclui deletar ────────────────────
  return (
    <Paper sx={{ p: 2, mb: 3, background: 'linear-gradient(135deg, #2E6DA4 0%, #1A3A5C 100%)', color: 'white' }}>
      <Box sx={{ display: 'flex', alignItems: 'center', gap: 2, flexWrap: 'wrap' }}>
        <FormControl size="small" sx={{ minWidth: 200 }}>
          <Select
            value={selectedCollection || ''}
            onChange={handleCollectionChange}
            displayEmpty
            sx={{
              color: 'white',
              '& fieldset': { borderColor: 'rgba(255,255,255,.3)' },
              '&:hover fieldset': { borderColor: 'rgba(255,255,255,.5)' },
              '&.Mui-focused fieldset': { borderColor: 'white' },
              '& .MuiSelect-icon': { color: 'rgba(255,255,255,.7)' },
            }}
          >
            {availableCollections && availableCollections.length > 0
              ? availableCollections.map((c) => (
                  <MenuItem key={c.id} value={c.name}>{c.name} ({c.count} docs)</MenuItem>
                ))
              : <MenuItem value="" disabled>Nenhuma coleção disponível</MenuItem>}
            <MenuItem
              value="__NEW_COLLECTION__"
              sx={{ borderTop: '1px solid', borderColor: 'divider', mt: 0.5, pt: 0.5 }}
            >
              <Box sx={{ display: 'flex', alignItems: 'center', gap: 0.5, color: 'primary.main', fontWeight: 700, fontSize: '.85rem' }}>
                <AddIcon fontSize="small" /> Nova Coleção
              </Box>
            </MenuItem>
          </Select>
        </FormControl>

        <IconButton
          onClick={() => handleDeleteClick(selectedCollection)}
          title="Deletar coleção"
          sx={{ color: 'rgba(255,255,255,.8)', '&:hover': { color: 'white', bgcolor: 'rgba(255,0,0,.1)' } }}
        >
          <DeleteIcon />
        </IconButton>
      </Box>
      {createDialog}
      {deleteDialog}
    </Paper>
  );
};

export default CollectionSelector;
