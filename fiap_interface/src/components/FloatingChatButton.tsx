import React from 'react';
import { Fab, Zoom } from '@mui/material';
import { Chat as ChatIcon } from '@mui/icons-material';

interface FloatingChatButtonProps {
  onClick: () => void;
}

const FloatingChatButton: React.FC<FloatingChatButtonProps> = ({ onClick }) => {
  return (
    <Zoom in={true} timeout={300}>
      <Fab
        color="primary"
        aria-label="abrir chat"
        onClick={onClick}
        sx={{
          position: 'fixed',
          bottom: 24,
          right: 24,
          background: '#2563eb',
          '&:hover': {
            background: '#1d4ed8',
            transform: 'scale(1.1)',
          },
          transition: 'all 0.3s ease',
          boxShadow: '0 8px 25px rgba(37, 99, 235, 0.3)',
          zIndex: 1000,
        }}
      >
        <ChatIcon sx={{ fontSize: 28 }} />
      </Fab>
    </Zoom>
  );
};

export default FloatingChatButton;
