import React, { useState } from 'react';
import { Box, Card, CardContent, Typography, Button, Snackbar, Alert } from '@mui/material';
import {
  BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, Cell, ResponsiveContainer,
} from 'recharts';
import { exportChartData } from '../mockData';

const COLORS = ['#2E6DA4', '#4A9FD4', '#27AE60', '#E67E22', '#8E44AD'];
const SHARE_URL = 'https://app.contacomigo.ai/share/sp-q4-2025-aberturas-cnae';

interface Props {
  darkMode?: boolean;
}

const Exportar: React.FC<Props> = ({ darkMode = false }) => {
  const [toast, setToast] = useState('');
  const gridLine = darkMode ? 'rgba(255,255,255,.07)' : '#f0f4f8';
  const tooltipBg = darkMode ? '#1a2535' : '#fff';
  const titleColor = darkMode ? '#e2eaf4' : '#1A3A5C';

  const showToast = (msg: string) => setToast(msg);

  const copyLink = () => {
    navigator.clipboard?.writeText(SHARE_URL).catch(() => {});
    showToast('Link copiado para a área de transferência!');
  };

  const outlinedBtnSx = {
    borderColor: '#D5E3F0',
    color: darkMode ? '#e2eaf4' : '#1A3A5C',
    borderRadius: '8px',
    fontWeight: 600,
    textTransform: 'none',
    '&:hover': { bgcolor: '#2E6DA4', color: 'white', borderColor: '#2E6DA4' },
  } as const;

  return (
    <Box>
      {/* Header */}
      <Box sx={{ mb: 2.5 }}>
        <Typography sx={{ fontWeight: 800, fontSize: '1.3rem', color: titleColor }}>
          Exportar e Compartilhar
        </Typography>
        <Typography variant="body2" sx={{ color: 'text.secondary', mt: 0.5 }}>
          Baixe seus insights ou compartilhe com a equipe
        </Typography>
      </Box>

      {/* Chart + export buttons */}
      <Card sx={{ mb: 2, boxShadow: '0 2px 8px rgba(0,0,0,.05)' }}>
        <CardContent>
          <Typography sx={{ fontWeight: 700, color: titleColor, fontSize: '1rem', mb: '6px' }}>
            📊 Gráfico: Aberturas por Setor – SP (Q4 2025)
          </Typography>
          <Typography sx={{ fontSize: '.85rem', color: 'text.secondary', mb: 2 }}>
            Gerado com base nos filtros aplicados. Pronto para apresentação.
          </Typography>

          <ResponsiveContainer width="100%" height={200}>
            <BarChart data={exportChartData} margin={{ top: 5, right: 10, left: -10, bottom: 5 }}>
              <CartesianGrid strokeDasharray="3 3" stroke={gridLine} />
              <XAxis dataKey="name" tick={{ fontSize: 10 }} />
              <YAxis tick={{ fontSize: 10 }} unit="k" />
              <Tooltip
                formatter={(v: number) => [`${v}k`, 'Aberturas']}
                contentStyle={{ background: tooltipBg, borderRadius: 8, fontSize: '.8rem', border: '1px solid #D5E3F0' }}
              />
              <Bar dataKey="value" radius={[6, 6, 0, 0]}>
                {exportChartData.map((_, idx) => (
                  <Cell key={idx} fill={COLORS[idx % COLORS.length]} />
                ))}
              </Bar>
            </BarChart>
          </ResponsiveContainer>

          <Box sx={{ display: 'flex', gap: 1.5, flexWrap: 'wrap', mt: 2, mb: 2 }}>
            <Button
              variant="contained"
              onClick={() => showToast('PNG baixado com sucesso!')}
              sx={{ bgcolor: '#2E6DA4', '&:hover': { bgcolor: '#1A3A5C' }, borderRadius: '8px', fontWeight: 700, textTransform: 'none' }}
            >
              ⬇ Baixar PNG
            </Button>
            <Button variant="outlined" onClick={() => showToast('PDF gerado com sucesso!')} sx={outlinedBtnSx}>
              📄 Exportar PDF
            </Button>
            <Button variant="outlined" onClick={() => showToast('Tabela exportada para Excel!')} sx={outlinedBtnSx}>
              📊 Exportar Excel
            </Button>
          </Box>

          {/* Share link */}
          <Box sx={{
            bgcolor: darkMode ? 'rgba(74,159,212,.08)' : '#F4F7FB',
            border: '1.5px dashed #4A9FD4',
            borderRadius: '10px',
            p: '14px 18px',
            display: 'flex',
            alignItems: 'center',
            gap: 1.5,
          }}>
            <Typography sx={{ fontSize: '1rem' }}>🔗</Typography>
            <Typography sx={{
              fontSize: '.83rem',
              color: 'text.secondary',
              flex: 1,
              overflow: 'hidden',
              textOverflow: 'ellipsis',
              whiteSpace: 'nowrap',
            }}>
              {SHARE_URL}
            </Typography>
            <Button
              onClick={copyLink}
              size="small"
              sx={{
                bgcolor: '#2E6DA4', color: 'white',
                '&:hover': { bgcolor: '#1A3A5C' },
                borderRadius: '6px', fontSize: '.8rem', fontWeight: 700,
                px: 1.5, whiteSpace: 'nowrap', textTransform: 'none',
              }}
            >
              Copiar link
            </Button>
          </Box>
        </CardContent>
      </Card>

      {/* Team sharing */}
      <Card sx={{ boxShadow: '0 2px 8px rgba(0,0,0,.05)' }}>
        <CardContent>
          <Typography sx={{ fontWeight: 700, color: titleColor, fontSize: '1rem', mb: '6px' }}>
            👥 Compartilhar com a equipe
          </Typography>
          <Typography sx={{ fontSize: '.85rem', color: 'text.secondary', mb: 2 }}>
            Envie o insight diretamente para colegas – sem precisar exportar arquivo.
          </Typography>
          <Box sx={{ display: 'flex', gap: 1.5, flexWrap: 'wrap' }}>
            <Button variant="outlined" onClick={() => showToast('Link copiado! Cole no Slack ou e-mail.')} sx={outlinedBtnSx}>
              💬 Copiar para Slack
            </Button>
            <Button variant="outlined" onClick={() => showToast('E-mail enviado para a equipe!')} sx={outlinedBtnSx}>
              ✉️ Enviar por e-mail
            </Button>
            <Button variant="outlined" onClick={() => showToast('Salvo no Google Drive!')} sx={outlinedBtnSx}>
              ☁️ Salvar no Drive
            </Button>
          </Box>
        </CardContent>
      </Card>

      <Snackbar
        open={!!toast}
        autoHideDuration={2800}
        onClose={() => setToast('')}
        anchorOrigin={{ vertical: 'bottom', horizontal: 'right' }}
      >
        <Alert
          onClose={() => setToast('')}
          severity="success"
          sx={{ bgcolor: '#1A3A5C', color: 'white', '& .MuiAlert-icon': { color: '#4A9FD4' } }}
        >
          {toast}
        </Alert>
      </Snackbar>
    </Box>
  );
};

export default Exportar;
