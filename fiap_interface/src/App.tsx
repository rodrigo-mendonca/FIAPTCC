import React, { useState } from 'react';
import { ThemeProvider, createTheme } from '@mui/material/styles';
import CssBaseline from '@mui/material/CssBaseline';
import { 
  Container, 
  Typography, 
  Box, 
  Switch, 
  FormControlLabel,
  Paper,
  GlobalStyles,
  Tabs,
  Tab
} from '@mui/material';
import { Chat, Science, Storage, Assessment } from '@mui/icons-material';
import ChatTabs from './components/ChatTabs';
import Tests from './components/Tests';
import VectorDBTest from './components/VectorDBTest';
import Statistics from './components/Statistics';
import TabPanel from './components/TabPanel';
import { CollectionProvider } from './contexts/CollectionContext';
import CollectionSelector from './components/CollectionSelector';

function App() {
  const [darkMode, setDarkMode] = useState(() => {
    const saved = localStorage.getItem('darkMode');
    return saved ? JSON.parse(saved) : false;
  });
  const [currentTab, setCurrentTab] = useState(0);

  const handleTabChange = (event: React.SyntheticEvent, newValue: number) => {
    setCurrentTab(newValue);
  };

  const handleDarkModeChange = (event: React.ChangeEvent<HTMLInputElement>) => {
    const newDarkMode = event.target.checked;
    setDarkMode(newDarkMode);
    localStorage.setItem('darkMode', JSON.stringify(newDarkMode));
  };

  const theme = createTheme({
    palette: {
      mode: darkMode ? 'dark' : 'light',
      primary: {
        main: '#ED145B', // Rosa/Vermelho personalizado
      },
      secondary: {
        main: '#C7104A', // Tom mais escuro
      },
      background: {
        default: darkMode ? '#121212' : '#f5f5f5',
        paper: darkMode ? '#1d1d1d' : '#ffffff',
      },
    },
    typography: {
      h3: {
        fontWeight: 600,
      },
      h5: {
        fontWeight: 500,
      },
    },
    components: {
      MuiCard: {
        styleOverrides: {
          root: {
            borderRadius: 12,
          },
        },
      },
    },
  });

  return (
    <ThemeProvider theme={theme}>
      <CssBaseline />
      <CollectionProvider>
        <GlobalStyles
          styles={{
            '.spinning': {
              animation: 'spin 1s linear infinite',
            },
            '@keyframes spin': {
              from: { transform: 'rotate(0deg)' },
              to: { transform: 'rotate(360deg)' },
            },
          }}
        />
        <Box sx={{ py: 4, px: 2, minHeight: '100vh' }}>
          <Box sx={{ mb: 2, textAlign: 'center' }}>
            <Typography variant="h3" component="h1" gutterBottom>
              FIAP
            </Typography>
            <FormControlLabel
              control={
                <Switch
                  checked={darkMode}
                  onChange={handleDarkModeChange}
                />
              }
              label={darkMode ? "Modo Escuro" : "Modo Claro"}
              sx={{ mt: 2 }}
            />
          </Box>

          {/* Collection Selector */}
          <CollectionSelector />

          {/* Navigation Tabs */}
          <Paper sx={{ mb: 3 }}>
            <Tabs 
              value={currentTab} 
              onChange={handleTabChange} 
              centered
              textColor="primary"
              indicatorColor="primary"
            >
              <Tab 
                icon={<Chat />} 
                label="Chat" 
                iconPosition="start"
              />
              <Tab 
                icon={<Storage />} 
                label="DataBase" 
                iconPosition="start"
              />
              <Tab 
                icon={<Assessment />} 
                label="Estatísticas" 
                iconPosition="start"
              />
            </Tabs>
          </Paper>

          {/* Tab Content */}
          <TabPanel value={currentTab} index={0} title="Chat" icon={<Chat />}>
            <ChatTabs darkMode={darkMode} />
          </TabPanel>

          <TabPanel value={currentTab} index={1} title="Database" icon={<Storage />}>
            <VectorDBTest />
          </TabPanel>

          <TabPanel value={currentTab} index={2} title="Estatísticas" icon={<Assessment />}>
            <Statistics />
          </TabPanel>
        </Box>
      </CollectionProvider>
    </ThemeProvider>
  );
}

export default App;
