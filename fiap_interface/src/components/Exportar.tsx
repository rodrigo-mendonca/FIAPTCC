import React, { useState, useEffect, useRef } from 'react';
import { Box, Card, CardContent, Typography, Button, Snackbar, Alert, Skeleton } from '@mui/material';
import {
  BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, Cell, ResponsiveContainer,
} from 'recharts';
import { exportChartData } from '../mockData';
import {
  chartToCanvas, canvasToPngBlob, buildChartPdf, buildCsv, downloadBlob,
} from '../utils/chartExport';

const SHARE_URL = 'https://app.contacomigo.ai/share/sp-q2-2026-aberturas-cnae';
const CHART_TITLE = 'Aberturas por Setor - SP';
const PNG_FILE = 'grafico-aberturas-setor-sp.png';

// Logo oficial (multicolor) do Slack.
const SlackIcon = () => (
  <svg width="16" height="16" viewBox="0 0 122.8 122.8" aria-hidden="true">
    <path d="M25.8 77.6c0 7.1-5.8 12.9-12.9 12.9S0 84.7 0 77.6s5.8-12.9 12.9-12.9h12.9zm6.5 0c0-7.1 5.8-12.9 12.9-12.9s12.9 5.8 12.9 12.9v32.3c0 7.1-5.8 12.9-12.9 12.9s-12.9-5.8-12.9-12.9z" fill="#e01e5a"/>
    <path d="M45.2 25.8c-7.1 0-12.9-5.8-12.9-12.9S38.1 0 45.2 0s12.9 5.8 12.9 12.9v12.9zm0 6.5c7.1 0 12.9 5.8 12.9 12.9s-5.8 12.9-12.9 12.9H12.9C5.8 58.1 0 52.3 0 45.2s5.8-12.9 12.9-12.9z" fill="#36c5f0"/>
    <path d="M97 45.2c0-7.1 5.8-12.9 12.9-12.9s12.9 5.8 12.9 12.9-5.8 12.9-12.9 12.9H97zm-6.5 0c0 7.1-5.8 12.9-12.9 12.9s-12.9-5.8-12.9-12.9V12.9C64.7 5.8 70.5 0 77.6 0s12.9 5.8 12.9 12.9z" fill="#2eb67d"/>
    <path d="M77.6 97c7.1 0 12.9 5.8 12.9 12.9s-5.8 12.9-12.9 12.9-12.9-5.8-12.9-12.9V97zm0-6.5c-7.1 0-12.9-5.8-12.9-12.9s5.8-12.9 12.9-12.9h32.3c7.1 0 12.9 5.8 12.9 12.9s-5.8 12.9-12.9 12.9z" fill="#ecb22e"/>
  </svg>
);

// Logo oficial (roxo) do Microsoft Teams.
const TeamsIcon = () => (
  <svg width="16" height="16" viewBox="0 0 2228.833 2073.333" aria-hidden="true">
    <path fill="#5059c9" d="M1554.637 777.5h575.713c54.391 0 98.483 44.092 98.483 98.483v524.398c0 199.901-162.051 361.952-361.952 361.952h-1.711c-199.901.028-361.975-162-362.004-361.901V828.971c0-28.427 23.045-51.471 51.471-51.471z"/>
    <circle fill="#5059c9" cx="1943.75" cy="440.583" r="233.25"/>
    <circle fill="#7b83eb" cx="1218.083" cy="336.917" r="336.917"/>
    <path fill="#7b83eb" d="M1667.323 777.5H717.01c-53.743 1.33-96.257 45.931-95.01 99.676v598.105c-7.505 322.519 247.657 590.16 570.167 598.053 322.51-7.893 577.671-275.534 570.167-598.053V877.176c1.245-53.745-41.268-98.346-95.011-99.676z"/>
    <path opacity=".1" d="M1244 777.5v838.145c-.258 38.435-23.549 72.964-59.09 87.598a91.856 91.856 0 0 1-35.765 7.257H667.613c-6.738-17.105-12.958-34.21-18.142-51.833a631.287 631.287 0 0 1-27.472-183.49V877.02c-1.246-53.659 41.198-98.19 94.855-99.52z"/>
    <path opacity=".2" d="M1192.167 777.5v889.978a91.802 91.802 0 0 1-7.257 35.765c-14.634 35.541-49.163 58.833-87.598 59.09H691.975c-8.812-17.105-17.105-34.21-24.362-51.833-7.257-17.623-12.958-34.21-18.142-51.833a631.287 631.287 0 0 1-27.472-183.49V877.02c-1.246-53.659 41.198-98.19 94.855-99.52z"/>
    <path opacity=".2" d="M1192.167 777.5v786.312c-.395 52.223-42.632 94.46-94.855 94.855h-448.45a631.287 631.287 0 0 1-27.472-183.49V877.02c-1.246-53.659 41.198-98.19 94.855-99.52z"/>
    <path opacity=".2" d="M1140.333 777.5v786.312c-.395 52.223-42.632 94.46-94.855 94.855H649.472a631.287 631.287 0 0 1-27.472-183.49V877.02c-1.246-53.659 41.198-98.19 94.855-99.52z"/>
    <path opacity=".1" d="M1244 509.522v163.275c-8.812.518-17.105 1.037-25.917 1.037-8.812 0-17.105-.518-25.917-1.037a285.232 285.232 0 0 1-51.833-8.293c-104.963-24.857-191.679-98.469-233.25-198.003a288.04 288.04 0 0 1-16.587-51.833h258.648c52.305.198 94.657 42.55 94.856 94.854z"/>
    <path opacity=".2" d="M1192.167 561.355v111.442a285.232 285.232 0 0 1-51.833-8.293c-104.963-24.857-191.679-98.469-233.25-198.003h190.228c52.305.198 94.657 42.55 94.855 94.854z"/>
    <path opacity=".2" d="M1192.167 561.355v111.442a285.232 285.232 0 0 1-51.833-8.293c-104.963-24.857-191.679-98.469-233.25-198.003h190.228c52.305.198 94.657 42.55 94.855 94.854z"/>
    <path opacity=".2" d="M1140.333 561.355v103.149c-104.963-24.857-191.679-98.469-233.25-198.003h138.395c52.305.198 94.656 42.55 94.855 94.854z"/>
    <path fill="#4b53bc" d="M203.466 466.5h941.067c54.111 0 97.967 43.856 97.967 97.967v941.067c0 54.111-43.856 97.967-97.967 97.967H203.466c-54.111 0-97.967-43.856-97.967-97.967V564.467c0-54.111 43.856-97.967 97.967-97.967z"/>
    <path fill="#fff" d="M921.318 825.301H733.81v510.557H614.241V825.301H427.602V726.475h493.716z"/>
  </svg>
);

interface Props {
  darkMode?: boolean;
}

const Exportar: React.FC<Props> = ({ darkMode = false }) => {
  const [toast, setToast] = useState('');
  const [toastSeverity, setToastSeverity] = useState<'success' | 'error'>('success');
  const [chartData, setChartData] = useState(exportChartData);
  const [loading, setLoading] = useState(false);
  const chartRef = useRef<HTMLDivElement>(null);
  const gridLine = darkMode ? 'rgba(255,255,255,.07)' : '#f0f4f8';
  const tooltipBg = darkMode ? '#1a2535' : '#fff';
  const titleColor = darkMode ? '#e2eaf4' : '#1A3A5C';

  const showToast = (msg: string, severity: 'success' | 'error' = 'success') => {
    setToastSeverity(severity);
    setToast(msg);
  };

  // Rasteriza o gráfico atual em um canvas (fundo branco, 2x de resolução).
  const getCanvas = () => {
    if (!chartRef.current) throw new Error('Gráfico não disponível.');
    return chartToCanvas(chartRef.current, 2);
  };

  // 1. PNG do gráfico
  const handlePng = async () => {
    try {
      const blob = await canvasToPngBlob(await getCanvas());
      downloadBlob(blob, PNG_FILE);
      showToast('PNG baixado com sucesso!');
    } catch (e) {
      showToast('Não foi possível gerar o PNG.', 'error');
    }
  };

  // 2. PDF com o gráfico
  const handlePdf = async () => {
    try {
      const blob = buildChartPdf(await getCanvas(), CHART_TITLE);
      downloadBlob(blob, 'grafico-aberturas-setor-sp.pdf');
      showToast('PDF gerado com sucesso!');
    } catch (e) {
      showToast('Não foi possível gerar o PDF.', 'error');
    }
  };

  // 3. CSV com os dados do gráfico
  const handleCsv = () => {
    try {
      downloadBlob(buildCsv(chartData), 'dados-aberturas-setor-sp.csv');
      showToast('Dados exportados em CSV!');
    } catch (e) {
      showToast('Não foi possível exportar o CSV.', 'error');
    }
  };

  // 4. Tenta abrir o Slack (baixa o PNG para anexar, já que deep links não anexam arquivos)
  const handleSlack = async () => {
    try {
      const blob = await canvasToPngBlob(await getCanvas());
      downloadBlob(blob, PNG_FILE);
      showToast('PNG baixado. Abrindo o Slack para você anexá-lo…');
      // Tenta o app nativo; cai para a versão web se não houver.
      const fallback = window.open('https://app.slack.com/client', '_blank');
      window.location.href = 'slack://open';
      if (!fallback) window.open('https://slack.com', '_blank');
    } catch (e) {
      showToast('Não foi possível abrir o Slack.', 'error');
    }
  };

  // 5. Enviar por e-mail: abre o app de e-mail com corpo padrão (PNG baixado p/ anexo)
  const handleEmail = async () => {
    try {
      const blob = await canvasToPngBlob(await getCanvas());
      downloadBlob(blob, PNG_FILE);
      const subject = encodeURIComponent('Insight: Aberturas por Setor – SP');
      const body = encodeURIComponent(
        'Olá,\n\n' +
        'Segue o gráfico de aberturas de empresas por setor em SP, gerado no ContaComigo.AI.\n' +
        `O arquivo PNG (${PNG_FILE}) foi baixado no seu computador — basta anexá-lo a este e-mail.\n\n` +
        `Link do insight: ${SHARE_URL}\n\n` +
        'Atenciosamente.',
      );
      window.location.href = `mailto:?subject=${subject}&body=${body}`;
      showToast('PNG baixado. Abrindo seu aplicativo de e-mail…');
    } catch (e) {
      showToast('Não foi possível abrir o e-mail.', 'error');
    }
  };

  // 6. Enviar no Teams (baixa o PNG e abre o Teams com mensagem pré-preenchida)
  const handleTeams = async () => {
    try {
      const blob = await canvasToPngBlob(await getCanvas());
      downloadBlob(blob, PNG_FILE);
      showToast('PNG baixado. Abrindo o Teams para você anexá-lo…');
      const msg = encodeURIComponent(
        `Gráfico de aberturas por setor – SP. Veja o insight: ${SHARE_URL}`,
      );
      const teamsUrl = `https://teams.microsoft.com/l/chat/0/0?message=${msg}`;
      const fallback = window.open(teamsUrl, '_blank');
      window.location.href = `msteams:/l/chat/0/0?message=${msg}`;
      if (!fallback) window.open(teamsUrl, '_blank');
    } catch (e) {
      showToast('Não foi possível abrir o Teams.', 'error');
    }
  };

  // Busca os dados do mock e atualiza o estado da página
  const refresh = () => {
    setLoading(true);
    setTimeout(() => {
      setChartData(exportChartData);
      setLoading(false);
    }, 600);
  };

  useEffect(() => { refresh(); }, []);

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
      <Box sx={{ mb: 2.5, display: 'flex', justifyContent: 'space-between', alignItems: 'flex-start', gap: 2, flexWrap: 'wrap' }}>
        <Box>
          <Typography sx={{ fontWeight: 800, fontSize: '1.3rem', color: titleColor }}>
            Exportar e Compartilhar
          </Typography>
          <Typography variant="body2" sx={{ color: 'text.secondary', mt: 0.5 }}>
            Baixe seus insights ou compartilhe com a equipe
          </Typography>
        </Box>
        <Button
          onClick={refresh}
          disabled={loading}
          size="small"
          variant="contained"
          sx={{
            bgcolor: '#2E6DA4', '&:hover': { bgcolor: '#1A3A5C' },
            color: 'white', fontWeight: 700, fontSize: '.72rem',
            textTransform: 'none', borderRadius: '8px',
          }}
        >
          {loading ? 'Atualizando…' : '🔄 Atualizar dados'}
        </Button>
      </Box>

      {/* Chart + export buttons */}
      <Card sx={{ mb: 2, boxShadow: '0 2px 8px rgba(0,0,0,.05)' }}>
        <CardContent>
          <Typography sx={{ fontWeight: 700, color: titleColor, fontSize: '1rem', mb: '6px' }}>
            📊 Gráfico: Aberturas por Setor – SP
          </Typography>
          <Typography sx={{ fontSize: '.85rem', color: 'text.secondary', mb: 2 }}>
            Gerado com base nos filtros aplicados. Pronto para apresentação.
          </Typography>

          {loading ? (
            <Skeleton variant="rounded" height={200} />
          ) : (
          <Box ref={chartRef}>
          <ResponsiveContainer width="100%" height={200}>
            <BarChart data={chartData} margin={{ top: 5, right: 10, left: -10, bottom: 5 }}>
              <CartesianGrid strokeDasharray="3 3" stroke={gridLine} />
              <XAxis dataKey="name" tick={{ fontSize: 10 }} />
              <YAxis tick={{ fontSize: 10 }} unit="k" />
              <Tooltip
                formatter={(v: any, _n: any, item: any) => [`${v}${item.payload.unit}`, item.payload.tooltipLabel]}
                contentStyle={{ background: tooltipBg, borderRadius: 8, fontSize: '.8rem', border: '1px solid #D5E3F0', color: titleColor }}
                labelStyle={{ color: titleColor }}
                itemStyle={{ color: titleColor }}
              />
              <Bar dataKey="value" radius={[6, 6, 0, 0]}>
                {chartData.map((entry, idx) => (
                  <Cell key={idx} fill={entry.color} />
                ))}
              </Bar>
            </BarChart>
          </ResponsiveContainer>
          </Box>
          )}

          <Box sx={{ display: 'flex', gap: 1.5, flexWrap: 'wrap', mt: 2, mb: 2 }}>
            <Button
              variant="contained"
              onClick={handlePng}
              sx={{ bgcolor: '#2E6DA4', '&:hover': { bgcolor: '#1A3A5C' }, borderRadius: '8px', fontWeight: 700, textTransform: 'none' }}
            >
              ⬇ Baixar PNG
            </Button>
            <Button variant="outlined" onClick={handlePdf} sx={outlinedBtnSx}>
              📄 Exportar PDF
            </Button>
            <Button variant="outlined" onClick={handleCsv} sx={outlinedBtnSx}>
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
            <Button variant="outlined" onClick={handleSlack} startIcon={<SlackIcon />} sx={outlinedBtnSx}>
              Enviar no Slack
            </Button>
            <Button variant="outlined" onClick={handleEmail} sx={outlinedBtnSx}>
              ✉️ Enviar por e-mail
            </Button>
            <Button variant="outlined" onClick={handleTeams} startIcon={<TeamsIcon />} sx={outlinedBtnSx}>
              Enviar no Teams
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
          severity={toastSeverity}
          sx={toastSeverity === 'success'
            ? { bgcolor: '#1A3A5C', color: 'white', '& .MuiAlert-icon': { color: '#4A9FD4' } }
            : { bgcolor: '#C0392B', color: 'white', '& .MuiAlert-icon': { color: '#fff' } }}
        >
          {toast}
        </Alert>
      </Snackbar>
    </Box>
  );
};

export default Exportar;
