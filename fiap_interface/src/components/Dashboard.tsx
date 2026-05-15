import React from 'react';
import { Box, Card, CardContent, Typography, Chip } from '@mui/material';
import {
  BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, Cell,
  PieChart, Pie, Legend, ResponsiveContainer,
} from 'recharts';
import { metricsData, setoresBarData, porteDonutData, insightsData } from '../mockData';

const CHART_COLORS = ['#2E6DA4', '#4A9FD4', '#27AE60', '#E67E22', '#8E44AD', '#1ABC9C', '#E74C3C', '#F39C12'];
const PORTE_COLORS = ['#4A9FD4', '#2E6DA4', '#27AE60', '#E67E22'];

const VARIANT_TOP: Record<string, string> = {
  blue: '#2E6DA4',
  green: '#27AE60',
  orange: '#E67E22',
  teal: '#1ABC9C',
};

const BADGE_STYLE: Record<string, { bg: string; color: string }> = {
  up:   { bg: '#D5F5E3', color: '#27AE60' },
  down: { bg: '#FADBD8', color: '#E74C3C' },
  new:  { bg: '#EBF5FB', color: '#2E6DA4' },
};

const INSIGHT_BORDER: Record<string, string> = {
  green: '#27AE60',
  blue: '#2E6DA4',
  orange: '#E67E22',
};

interface Props {
  darkMode?: boolean;
}

const Dashboard: React.FC<Props> = ({ darkMode = false }) => {
  const gridLine = darkMode ? 'rgba(255,255,255,.07)' : '#f0f4f8';
  const tooltipBg = darkMode ? '#1a2535' : '#fff';
  const titleColor = darkMode ? '#e2eaf4' : '#1A3A5C';

  return (
    <Box>
      {/* Header */}
      <Box sx={{ mb: 3 }}>
        <Box sx={{ display: 'flex', alignItems: 'center', gap: 1, flexWrap: 'wrap' }}>
          <Typography sx={{ fontWeight: 800, fontSize: '1.3rem', color: titleColor }}>
            Visão Geral do Mercado
          </Typography>
          <Chip
            label="Atualizado hoje"
            size="small"
            sx={{ bgcolor: '#2E6DA4', color: 'white', fontWeight: 700, fontSize: '.72rem' }}
          />
        </Box>
        <Typography variant="body2" sx={{ color: 'text.secondary', mt: 0.5 }}>
          Dados de CNPJs ativos – Receita Federal do Brasil
        </Typography>
      </Box>

      {/* Metric cards */}
      <Box sx={{ display: 'grid', gridTemplateColumns: 'repeat(4, 1fr)', gap: 2, mb: 3 }}>
        {metricsData.map((m, i) => (
          <Card key={i} sx={{ overflow: 'hidden', boxShadow: '0 2px 8px rgba(0,0,0,.05)', position: 'relative' }}>
            <Box sx={{ height: 3, bgcolor: VARIANT_TOP[m.variant] }} />
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
              <Box sx={{ position: 'absolute', right: 14, top: 18, fontSize: '1.5rem', opacity: .18 }}>
                {m.icon}
              </Box>
            </CardContent>
          </Card>
        ))}
      </Box>

      {/* Charts row */}
      <Box sx={{ display: 'grid', gridTemplateColumns: '2fr 1fr', gap: 2, mb: 3 }}>
        <Card sx={{ boxShadow: '0 2px 8px rgba(0,0,0,.05)' }}>
          <CardContent>
            <Typography sx={{ fontWeight: 700, color: titleColor, fontSize: '.9rem', mb: '4px' }}>
              Aberturas por Setor – São Paulo (2025 Q1–Q4)
            </Typography>
            <Typography sx={{ fontSize: '.75rem', color: 'text.secondary', mb: 1.5 }}>
              Top 8 CNAEs por volume de abertura de empresas
            </Typography>
            <ResponsiveContainer width="100%" height={220}>
              <BarChart data={setoresBarData} margin={{ top: 5, right: 10, left: -10, bottom: 5 }}>
                <CartesianGrid strokeDasharray="3 3" stroke={gridLine} />
                <XAxis dataKey="name" tick={{ fontSize: 10 }} />
                <YAxis tick={{ fontSize: 10 }} unit="k" />
                <Tooltip
                  formatter={(v: number) => [`${v}k`, 'Novas empresas']}
                  contentStyle={{ background: tooltipBg, borderRadius: 8, fontSize: '.8rem', border: '1px solid #D5E3F0' }}
                />
                <Bar dataKey="value" radius={[6, 6, 0, 0]}>
                  {setoresBarData.map((_, idx) => (
                    <Cell key={idx} fill={CHART_COLORS[idx % CHART_COLORS.length]} />
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
                  data={porteDonutData}
                  cx="50%"
                  cy="50%"
                  innerRadius={55}
                  outerRadius={85}
                  dataKey="value"
                  paddingAngle={2}
                >
                  {porteDonutData.map((_, idx) => (
                    <Cell key={idx} fill={PORTE_COLORS[idx]} />
                  ))}
                </Pie>
                <Legend wrapperStyle={{ fontSize: '11px' }} />
                <Tooltip
                  formatter={(v: number) => [`${v}%`, '']}
                  contentStyle={{ background: tooltipBg, borderRadius: 8, fontSize: '.8rem', border: '1px solid #D5E3F0' }}
                />
              </PieChart>
            </ResponsiveContainer>
          </CardContent>
        </Card>
      </Box>

      {/* Insight cards */}
      <Box sx={{ display: 'grid', gridTemplateColumns: 'repeat(3, 1fr)', gap: 1.75 }}>
        {insightsData.map((insight, i) => {
          const badge = BADGE_STYLE[insight.badgeType];
          return (
            <Card key={i} sx={{ borderLeft: `4px solid ${INSIGHT_BORDER[insight.variant]}`, boxShadow: '0 2px 8px rgba(0,0,0,.04)' }}>
              <CardContent sx={{ p: '16px !important' }}>
                <Typography sx={{ fontWeight: 700, color: titleColor, fontSize: '.82rem', mb: '5px' }}>
                  {insight.icon} {insight.title}
                </Typography>
                <Typography sx={{ fontSize: '.78rem', color: 'text.secondary', lineHeight: 1.5 }}>
                  {insight.text}
                </Typography>
                <Box sx={{
                  display: 'inline-block', mt: 1, px: 1, py: '2px',
                  borderRadius: '99px', bgcolor: badge.bg, color: badge.color,
                  fontSize: '.7rem', fontWeight: 700,
                }}>
                  {insight.badge}
                </Box>
              </CardContent>
            </Card>
          );
        })}
      </Box>
    </Box>
  );
};

export default Dashboard;
