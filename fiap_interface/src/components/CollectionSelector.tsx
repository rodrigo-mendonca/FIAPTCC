import React from 'react';
import {
  Box,
  FormControl,
  InputLabel,
  Select,
  MenuItem,
  SelectChangeEvent,
  Chip,
  Typography,
  Paper
} from '@mui/material';
import { Storage as StorageIcon } from '@mui/icons-material';
import { useCollection } from '../contexts/CollectionContext';

const CollectionSelector: React.FC = () => {
  const { selectedCollection, setSelectedCollection, availableCollections } = useCollection();

  const handleCollectionChange = (event: SelectChangeEvent<string>) => {
    const newCollection = event.target.value;
    setSelectedCollection(newCollection);
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
            {availableCollections?.map((collection) => (
              <MenuItem key={collection.id} value={collection.name}>
                {collection.name} ({collection.count} docs)
              </MenuItem>
            )) || (
              <MenuItem value="sistema_comercial">sistema_comercial</MenuItem>
            )}
          </Select>
        </FormControl>

        <Chip
          label={`${selectedCollection} - Ativa`}
          color="success"
          variant="outlined"
          sx={{
            color: 'white',
            borderColor: 'rgba(255,255,255,0.5)',
            '& .MuiChip-label': { fontWeight: 'bold' }
          }}
        />
      </Box>
    </Paper>
  );
};

export default CollectionSelector;