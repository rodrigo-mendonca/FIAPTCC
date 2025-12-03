import React from 'react';
import { Box, Typography } from '@mui/material';

interface TabPanelProps {
  children?: React.ReactNode;
  index: number;
  value: number;
  title?: string;
  icon?: React.ReactNode;
}

const TabPanel: React.FC<TabPanelProps> = ({ children, value, index, title, icon, ...other }) => {
  return (
    <div
      role="tabpanel"
      hidden={value !== index}
      id={`tabpanel-${index}`}
      aria-labelledby={`tab-${index}`}
      {...other}
    >
      {value === index && (
        <Box sx={{ py: 3 }}>
          {title && (
            <Box sx={{ display: 'flex', alignItems: 'center', gap: 1, mb: 3 }}>
              {icon}
              <Typography variant="h5" component="h2" fontWeight="bold">
                {title}
              </Typography>
            </Box>
          )}
          {children}
        </Box>
      )}
    </div>
  );
};

export default TabPanel;