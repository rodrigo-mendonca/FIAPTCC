import { useState, type ReactNode } from 'react';
import { ThemeProvider, createTheme } from '@mui/material/styles';
import CssBaseline from '@mui/material/CssBaseline';
import {
  Box, Drawer, List, ListItem, ListItemButton, ListItemText,
  AppBar, Toolbar, Typography, IconButton, Avatar, GlobalStyles,
  Menu, MenuItem, Fab, Tooltip, Divider, useMediaQuery,
} from '@mui/material';
import {
  DarkMode as DarkModeIcon,
  LightMode as LightModeIcon,
  Chat as ChatIcon,
  Close as CloseIcon,
  Logout as LogoutIcon,
  Menu as MenuIcon,
} from '@mui/icons-material';

import ChatTabs, { ConfigurableChat, chatConfigs } from './components/ChatTabs';
import Dashboard from './components/Dashboard';
import ExplorarMercado from './components/ExplorarMercado';
import Exportar from './components/Exportar';
import Manutencao from './components/Manutencao';
import NotificationCenter from './components/NotificationCenter';
import { CollectionProvider } from './contexts/CollectionContext';
import { NotificationProvider } from './contexts/NotificationContext';

const SIDEBAR_WIDTH = 230;

const NAV_ITEMS = [
  { id: 'dashboard', label: 'Dashboard',           emoji: '📊' },
  { id: 'chat',      label: 'Perguntar aos Dados', emoji: '💬' },
  { id: 'explorar',  label: 'Explorar Mercado',    emoji: '🔍' },
  { id: 'exportar',  label: 'Exportar',            emoji: '📤' },
  { id: 'manutencao', label: 'Manutenção',         emoji: '⚙️', dividerBefore: true },
] as const;

type PageId = typeof NAV_ITEMS[number]['id'];

const PAGE_TITLES: Record<PageId, string> = {
  dashboard:  'Dashboard Geral',
  chat:       'Perguntar aos Dados',
  explorar:   'Explorar Mercado',
  exportar:   'Exportar e Compartilhar',
  manutencao: 'Manutenção',
};

function App() {
  const [darkMode, setDarkMode] = useState(() => {
    const saved = localStorage.getItem('darkMode');
    return saved ? JSON.parse(saved) : false;
  });
  const [currentPage, setCurrentPage] = useState<PageId>('dashboard');
  const isMobile = useMediaQuery('(max-width:900px)');
  const [sidebarOpen, setSidebarOpen] = useState(() => window.innerWidth > 900);

  const [chatOpen, setChatOpen] = useState(false);

  // User avatar menu
  const [userMenuAnchor, setUserMenuAnchor] = useState<null | HTMLElement>(null);

  const toggleDarkMode = () => {
    const next = !darkMode;
    setDarkMode(next);
    localStorage.setItem('darkMode', JSON.stringify(next));
  };

  const theme = createTheme({
    palette: {
      mode: darkMode ? 'dark' : 'light',
      primary:    { main: '#2E6DA4' },
      secondary:  { main: '#4A9FD4' },
      background: {
        default: darkMode ? '#0f1923' : '#F4F7FB',
        paper:   darkMode ? '#1a2535' : '#FFFFFF',
      },
    },
    typography: {
      fontFamily: "'Segoe UI', system-ui, sans-serif",
    },
    components: {
      MuiCard: { styleOverrides: { root: { borderRadius: 12 } } },
    },
  });

  const PAGES: { id: PageId; element: ReactNode }[] = [
    { id: 'dashboard',  element: <Dashboard darkMode={darkMode} /> },
    { id: 'chat',       element: <ChatTabs darkMode={darkMode} /> },
    { id: 'explorar',   element: <ExplorarMercado darkMode={darkMode} /> },
    { id: 'exportar',   element: <Exportar darkMode={darkMode} /> },
    { id: 'manutencao', element: <Manutencao darkMode={darkMode} /> },
  ];

  return (
    <ThemeProvider theme={theme}>
      <CssBaseline />
      <NotificationProvider>
        <CollectionProvider>
          <NotificationCenter />
          <GlobalStyles
            styles={{
              '.spinning': { animation: 'spin 1s linear infinite' },
              '@keyframes spin': { from: { transform: 'rotate(0deg)' }, to: { transform: 'rotate(360deg)' } },
            }}
          />

          <Box sx={{ display: 'flex', minHeight: '100vh' }}>

            {/* ── Sidebar ── */}
            <Drawer
              variant={isMobile ? 'temporary' : 'permanent'}
              open={sidebarOpen}
              onClose={() => setSidebarOpen(false)}
              ModalProps={{ keepMounted: true }}
              sx={{
                width: isMobile ? 0 : (sidebarOpen ? SIDEBAR_WIDTH : 0),
                flexShrink: 0,
                transition: 'width .22s ease',
                overflow: 'hidden',
                '& .MuiDrawer-paper': {
                  width: SIDEBAR_WIDTH,
                  background: '#1A3A5C',
                  color: 'white',
                  border: 'none',
                  boxShadow: '2px 0 16px rgba(0,0,0,.18)',
                  display: 'flex',
                  flexDirection: 'column',
                  transform: isMobile
                    ? 'none'
                    : (sidebarOpen ? 'translateX(0)' : `translateX(-${SIDEBAR_WIDTH}px)`),
                  transition: 'transform .22s ease',
                },
              }}
            >
              {/* Logo */}
              <Box sx={{ p: '24px 20px 16px', borderBottom: '1px solid rgba(255,255,255,.12)' }}>
                <Typography sx={{ color: 'white', fontWeight: 800, fontSize: '1.22rem', letterSpacing: '-0.5px', lineHeight: 1.2 }}>
                  conta<span style={{ color: '#4A9FD4' }}>comigo</span>.ai
                </Typography>
                <Typography component="span" sx={{ color: 'rgba(255,255,255,.5)', fontSize: '.72rem', display: 'block', mt: '3px' }}>
                  Inteligência de Mercado
                </Typography>
              </Box>

              {/* Nav */}
              <List sx={{ flex: 1, py: 1.5, px: 0 }}>
                {NAV_ITEMS.map((item) => {
                  const active = currentPage === item.id;
                  return (
                    <Box key={item.id}>
                      {'dividerBefore' in item && item.dividerBefore && (
                        <Divider sx={{ borderColor: 'rgba(255,255,255,.1)', my: 1 }} />
                      )}
                      <ListItem disablePadding>
                        <ListItemButton
                          onClick={() => {
                            setCurrentPage(item.id);
                            if (isMobile) setSidebarOpen(false);
                          }}
                          sx={{
                            px: '20px', py: '10px',
                            borderLeft: `3px solid ${active ? '#4A9FD4' : 'transparent'}`,
                            bgcolor: active ? 'rgba(74,159,212,.18)' : 'transparent',
                            transition: '.15s',
                            '&:hover': { bgcolor: 'rgba(255,255,255,.07)' },
                          }}
                        >
                          <ListItemText
                            primary={`${item.emoji}  ${item.label}`}
                            slotProps={{
                              primary: {
                                fontSize: '.88rem',
                                color: active ? 'white' : 'rgba(255,255,255,.65)',
                                fontWeight: active ? 600 : 400,
                              },
                            }}
                          />
                        </ListItemButton>
                      </ListItem>
                    </Box>
                  );
                })}
              </List>

              {/* Footer */}
              <Box sx={{ p: '16px 20px', borderTop: '1px solid rgba(255,255,255,.1)' }}>
                <Typography sx={{ color: 'rgba(255,255,255,.4)', fontSize: '.75rem', lineHeight: 1.7 }}>
                  FIAP · MBA<br />
                  Dados: Receita Federal (RFB)
                </Typography>
              </Box>
            </Drawer>

            {/* ── Main ── */}
            <Box sx={{ flex: 1, display: 'flex', flexDirection: 'column', minWidth: 0 }}>

              {/* Topbar */}
              <AppBar
                position="sticky" elevation={0} color="inherit"
                sx={{ bgcolor: 'background.paper', borderBottom: '1px solid', borderColor: 'divider', boxShadow: '0 1px 8px rgba(0,0,0,.06)' }}
              >
                <Toolbar sx={{ justifyContent: 'space-between', minHeight: '56px !important', px: { xs: '12px !important', md: '28px !important' } }}>
                  <Box sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
                    <IconButton size="small" onClick={() => setSidebarOpen((o) => !o)} sx={{ color: 'text.secondary' }}>
                      <MenuIcon fontSize="small" />
                    </IconButton>
                    <Typography sx={{ fontWeight: 700, fontSize: { xs: '.85rem', md: '1rem' }, color: darkMode ? '#e2eaf4' : '#1A3A5C' }}>
                      {PAGE_TITLES[currentPage]}
                    </Typography>
                  </Box>

                  <Box sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
                    {/* Dark mode toggle */}
                    <IconButton size="small" onClick={toggleDarkMode} sx={{ color: 'text.secondary' }}>
                      {darkMode ? <LightModeIcon fontSize="small" /> : <DarkModeIcon fontSize="small" />}
                    </IconButton>

                    {/* User name + avatar menu */}
                    <Typography variant="body2" sx={{ color: 'text.secondary', ml: 0.5, display: { xs: 'none', sm: 'block' } }}>
                      Contabilizei
                    </Typography>
                    <Avatar
                      onClick={(e) => setUserMenuAnchor(e.currentTarget)}
                      sx={{
                        width: 32, height: 32, bgcolor: '#2E6DA4',
                        fontSize: '.8rem', fontWeight: 700, cursor: 'pointer',
                        '&:hover': { bgcolor: '#1A3A5C' }, transition: '.15s',
                      }}
                    >
                      CT
                    </Avatar>

                    {/* User dropdown menu */}
                    <Menu
                      anchorEl={userMenuAnchor}
                      open={Boolean(userMenuAnchor)}
                      onClose={() => setUserMenuAnchor(null)}
                      anchorOrigin={{ vertical: 'bottom', horizontal: 'right' }}
                      transformOrigin={{ vertical: 'top', horizontal: 'right' }}
                      PaperProps={{ sx: { mt: 1, minWidth: 180, borderRadius: 2, boxShadow: '0 4px 20px rgba(0,0,0,.15)' } }}
                    >
                      <Box sx={{ px: 2, py: 1.5 }}>
                        <Typography sx={{ fontWeight: 700, fontSize: '.9rem' }}>Contabilizei</Typography>
                        <Typography variant="caption" color="text.secondary">teste@contabilizei.com.br</Typography>
                      </Box>
                      <Divider />
                      <MenuItem
                        onClick={() => setUserMenuAnchor(null)}
                        sx={{ gap: 1.5, py: 1.2, color: 'error.main', '&:hover': { bgcolor: 'error.lighter' } }}
                      >
                        <LogoutIcon fontSize="small" />
                        <Typography fontSize=".9rem">Sair</Typography>
                      </MenuItem>
                    </Menu>
                  </Box>
                </Toolbar>
              </AppBar>

              {/* Page content — todas as telas ficam montadas; só a ativa é exibida */}
              <Box sx={{ flex: 1, p: { xs: 1.5, md: 3.5 }, bgcolor: 'background.default' }}>
                {PAGES.map(({ id, element }) => (
                  <Box key={id} sx={{ display: currentPage === id ? 'block' : 'none' }}>
                    {element}
                  </Box>
                ))}
              </Box>
            </Box>
          </Box>

          {/* ── Floating Chat Geral — fica sempre montado; só oculta ao fechar ── */}
          <Box sx={{
            display: chatOpen ? 'block' : 'none',
            position: 'fixed',
            bottom: { xs: 12, md: 88 },
            right: { xs: 12, md: 24 },
            left: { xs: 12, md: 'auto' },
            top: { xs: 12, md: 'auto' },
            zIndex: 1300,
            width: { xs: 'auto', md: 820 },
            height: { xs: 'auto', md: 640 },
            borderRadius: 3, overflow: 'hidden',
            boxShadow: '0 12px 48px rgba(0,0,0,.25)',
            bgcolor: 'background.paper',
          }}>
            <ConfigurableChat
              config={chatConfigs.general}
              darkMode={darkMode}
              containerHeight={isMobile ? '100%' : 640}
              hideSuggestions
            />
          </Box>

          <Tooltip title={chatOpen ? 'Fechar' : 'Chat'} placement="left">
            <Fab
              color="primary"
              onClick={() => setChatOpen((o) => !o)}
              sx={{ position: 'fixed', bottom: 24, right: 24, zIndex: 1301 }}
            >
              {chatOpen ? <CloseIcon /> : <ChatIcon />}
            </Fab>
          </Tooltip>

        </CollectionProvider>
      </NotificationProvider>
    </ThemeProvider>
  );
}

export default App;
