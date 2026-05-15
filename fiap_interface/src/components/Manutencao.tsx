import React, { useState } from 'react';
import { Box, Paper, Tabs, Tab, Typography } from '@mui/material';
import { Storage as StorageIcon, Assessment as AssessmentIcon } from '@mui/icons-material';
import VectorDBTest from './VectorDBTest';
import Statistics from './Statistics';
import CollectionSelector from './CollectionSelector';

interface Props {
  darkMode?: boolean;
}

const Manutencao: React.FC<Props> = ({ darkMode = false }) => {
  const [tab, setTab] = useState(0);
  const titleColor = darkMode ? '#e2eaf4' : '#1A3A5C';

  return (
    <Box>
      <Box sx={{ mb: 2.5 }}>
        <Typography sx={{ fontWeight: 800, fontSize: '1.3rem', color: titleColor }}>
          Manutenção
        </Typography>
        <Typography variant="body2" sx={{ color: 'text.secondary', mt: 0.5 }}>
          Gerenciamento de banco vetorial e monitoramento do sistema
        </Typography>
      </Box>

      {/* Gerenciar coleções — inclui criar e deletar */}
      <CollectionSelector />

      <Paper sx={{ mb: 3 }}>
        <Tabs
          value={tab}
          onChange={(_, v) => setTab(v)}
          indicatorColor="primary"
          textColor="primary"
        >
          <Tab icon={<StorageIcon />} label="Base de Dados" iconPosition="start" sx={{ fontWeight: 600 }} />
          <Tab icon={<AssessmentIcon />} label="Estatísticas" iconPosition="start" sx={{ fontWeight: 600 }} />
        </Tabs>
      </Paper>

      {tab === 0 && <VectorDBTest />}
      {tab === 1 && <Statistics />}
    </Box>
  );
};

export default Manutencao;
