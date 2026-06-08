import React, { useState, useEffect } from 'react';
import { Box, Card, CardContent, Typography, Button, Skeleton } from '@mui/material';
import {
  BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, Cell,
  PieChart, Pie, Legend, ResponsiveContainer,
} from 'recharts';
import {
  fetchMetrics, fetchSetores, fetchPorte, fetchInsights,
  Metric, BarDatum, Insight,
} from '../services/marketApi';

interface Props {
  darkMode?: boolean;
}

const Dashboard: React.FC<Props> = ({ darkMode = false }) => {
  const gridLine = darkMode ? 'rgba(255,255,255,.07)' : '#f0f4f8';
  const tooltipBg = darkMode ? '#1a2535' : '#fff';
  const titleColor = darkMode ? '#e2eaf4' : '#1A3A5C';

  const [metrics, setMetrics] = useState<Metric[]>([]);
  const [setores, setSetores] = useState<BarDatum[]>([]);
  const [porte, setPorte] = useState<BarDatum[]>([]);
  const [insights, setInsights] = useState<Insight[]>([]);
  const [loading, setLoading] = useState(false);

  // Busca os dados da API e atualiza o estado da página
  const refresh = async () => {
    setLoading(true);
    try {
      const [m, s, p, i] = await Promise.all([
        fetchMetrics(), fetchSetores(), fetchPorte(), fetchInsights(),
      ]);
      setMetrics(m);
      setSetores(s);
      setPorte(p);
      setInsights(i);
    } catch (err) {
      console.error('Erro ao carregar dados do dashboard:', err);
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => { refresh(); }, []);

  return (
    <Box>
      {/* Header */}
      <Box sx={{ mb: 3, display: 'flex', justifyContent: 'space-between', alignItems: 'flex-start', gap: 2, flexWrap: 'wrap' }}>
        <Box>
          <Typography sx={{ fontWeight: 800, fontSize: '1.3rem', color: titleColor }}>
            Visão Geral do Mercado
          </Typography>
          <Typography variant="body2" sx={{ color: 'text.secondary', mt: 0.5 }}>
            Dados de CNPJs ativos – Receita Federal do Brasil
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

      {loading ? (
        <>
          {/* Skeleton: metric cards */}
          <Box sx={{ display: 'grid', gridTemplateColumns: { xs: '1fr', sm: 'repeat(2, 1fr)', lg: 'repeat(4, 1fr)' }, gap: 2, mb: 3 }}>
            {Array.from({ length: 4 }).map((_, i) => (
              <Card key={i} sx={{ overflow: 'hidden', boxShadow: '0 2px 8px rgba(0,0,0,.05)' }}>
                <Skeleton variant="rectangular" height={3} />
                <CardContent sx={{ p: '18px 20px !important' }}>
                  <Skeleton variant="text" width="65%" height={14} />
                  <Skeleton variant="text" width="45%" height={34} sx={{ mt: '6px' }} />
                  <Skeleton variant="text" width="55%" height={12} sx={{ mt: '4px' }} />
                </CardContent>
              </Card>
            ))}
          </Box>

          {/* Skeleton: charts row */}
          <Box sx={{ display: 'grid', gridTemplateColumns: { xs: '1fr', md: '2fr 1fr' }, gap: 2, mb: 3 }}>
            {Array.from({ length: 2 }).map((_, i) => (
              <Card key={i} sx={{ boxShadow: '0 2px 8px rgba(0,0,0,.05)' }}>
                <CardContent>
                  <Skeleton variant="text" width="50%" height={18} />
                  <Skeleton variant="text" width="35%" height={12} sx={{ mb: 1.5 }} />
                  <Skeleton variant="rounded" height={220} />
                </CardContent>
              </Card>
            ))}
          </Box>

          {/* Skeleton: insight cards */}
          <Box sx={{ display: 'grid', gridTemplateColumns: { xs: '1fr', sm: 'repeat(2, 1fr)', lg: 'repeat(3, 1fr)' }, gap: 1.75 }}>
            {Array.from({ length: 3 }).map((_, i) => (
              <Card key={i} sx={{ boxShadow: '0 2px 8px rgba(0,0,0,.04)' }}>
                <CardContent sx={{ p: '16px !important' }}>
                  <Skeleton variant="text" width="70%" height={16} />
                  <Skeleton variant="text" width="100%" height={12} sx={{ mt: '5px' }} />
                  <Skeleton variant="text" width="90%" height={12} />
                  <Skeleton variant="rounded" width={92} height={20} sx={{ mt: 1 }} />
                </CardContent>
              </Card>
            ))}
          </Box>
        </>
      ) : (
      <>
      {/* Metric cards */}
      <Box sx={{ display: 'grid', gridTemplateColumns: { xs: '1fr', sm: 'repeat(2, 1fr)', lg: 'repeat(4, 1fr)' }, gap: 2, mb: 3 }}>
        {metrics.map((m, i) => (
          <Card key={i} sx={{ overflow: 'hidden', boxShadow: '0 2px 8px rgba(0,0,0,.05)', position: 'relative' }}>
            <Box sx={{ height: 3, bgcolor: m.topColor }} />
            <CardContent sx={{ p: '18px 20px !important' }}>
              <Typography sx={{ fontSize: '.75rem', color: 'text.secondary', fontWeight: 600, textTransform: 'uppercase', letterSpacing: '.5px', mb: '6px' }}>
                {m.label}
              </Typography>
              <Typography sx={{ fontSize: '1.7rem', fontWeight: 800, color: titleColor, lineHeight: 1 }}>
                {m.value}
              </Typography>
              <Typography sx={{ fontSize: '.75rem', color: '#27AE60', mt: '4px' }}>
                {m.sub}
              </Typography>
              <Box sx={{ position: 'absolute', right: 14, top: 18, fontSize: '1.5rem', opacity: 1 }}>
                {m.icon}
              </Box>
            </CardContent>
          </Card>
        ))}
      </Box>

      {/* Charts row */}
      <Box sx={{ display: 'grid', gridTemplateColumns: { xs: '1fr', md: '2fr 1fr' }, gap: 2, mb: 3 }}>
        <Card sx={{ boxShadow: '0 2px 8px rgba(0,0,0,.05)' }}>
          <CardContent>
            <Typography sx={{ fontWeight: 700, color: titleColor, fontSize: '.9rem', mb: '4px' }}>
              Aberturas por Setor – São Paulo
            </Typography>
            <Typography sx={{ fontSize: '.75rem', color: 'text.secondary', mb: 1.5 }}>
              Top 8 CNAEs por volume de abertura de empresas
            </Typography>
            <ResponsiveContainer width="100%" height={220}>
              <BarChart data={setores} margin={{ top: 5, right: 10, left: -10, bottom: 5 }}>
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
                  {setores.map((entry, idx) => (
                    <Cell key={idx} fill={entry.color} />
                  ))}
                </Bar>
              </BarChart>
            </ResponsiveContainer>
          </CardContent>
        </Card>

        <Card sx={{ boxShadow: '0 2px 8px rgba(0,0,0,.05)' }}>
          <CardContent>
            <Typography sx={{ fontWeight: 700, color: titleColor, fontSize: '.9rem', mb: '4px' }}>
              Distribuição por Porte
            </Typography>
            <Typography sx={{ fontSize: '.75rem', color: 'text.secondary', mb: 1.5 }}>
              Empresas abertas no último trimestre
            </Typography>
            <ResponsiveContainer width="100%" height={220}>
              <PieChart>
                <Pie
                  data={porte}
                  cx="50%"
                  cy="50%"
                  innerRadius={55}
                  outerRadius={85}
                  dataKey="value"
                  paddingAngle={2}
                  stroke="none"
                >
                  {porte.map((entry, idx) => (
                    <Cell key={idx} fill={entry.color} stroke="none" />
                  ))}
                </Pie>
                <Legend wrapperStyle={{ fontSize: '11px' }} />
                <Tooltip
                  formatter={(v: any, _n: any, item: any) => [`${v}${item.payload.unit}`, item.payload.tooltipLabel]}
                  contentStyle={{ background: tooltipBg, borderRadius: 8, fontSize: '.8rem', border: '1px solid #D5E3F0', color: titleColor }}
                  labelStyle={{ color: titleColor }}
                  itemStyle={{ color: titleColor }}
                />
              </PieChart>
            </ResponsiveContainer>
          </CardContent>
        </Card>
      </Box>

      {/* Insight cards */}
      <Box sx={{ display: 'grid', gridTemplateColumns: { xs: '1fr', sm: 'repeat(2, 1fr)', lg: 'repeat(3, 1fr)' }, gap: 1.75 }}>
        {insights.map((insight, i) => {
          return (
            <Card key={i} sx={{ borderLeft: `4px solid ${insight.borderColor}`, boxShadow: '0 2px 8px rgba(0,0,0,.04)' }}>
              <CardContent sx={{ p: '16px !important' }}>
                <Typography sx={{ fontWeight: 700, color: titleColor, fontSize: '.82rem', mb: '5px' }}>
                  {insight.icon} {insight.title}
                </Typography>
                <Typography sx={{ fontSize: '.78rem', color: 'text.secondary', lineHeight: 1.5 }}>
                  {insight.text}
                </Typography>
                <Box sx={{
                  display: 'inline-block', mt: 1, px: 1, py: '2px',
                  borderRadius: '99px', bgcolor: insight.badgeBg, color: insight.badgeColor,
                  fontSize: '.7rem', fontWeight: 700,
                }}>
                  {insight.badge}
                </Box>
              </CardContent>
            </Card>
          );
        })}
      </Box>
      </>
      )}
    </Box>
  );
};

export default Dashboard;
