import React, { useState } from 'react';
import {
  Box, Card, CardContent, Typography,
  Select, MenuItem, FormControl, InputLabel, Button, SelectChangeEvent,
} from '@mui/material';
import {
  LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip,
  BarChart, Bar, Cell, ResponsiveContainer,
} from 'recharts';
import { lineEvolutionData, cidadesBarData } from '../mockData';

const COLORS = ['#2E6DA4', '#4A9FD4', '#27AE60', '#E67E22', '#8E44AD'];

const ESTADOS = ['São Paulo', 'Rio de Janeiro', 'Minas Gerais', 'Paraná', 'Todos'];
const PORTES = ['Todos', 'MEI', 'Microempresa (ME)', 'Pequeno Porte (EPP)', 'Grande'];
const PERIODOS = ['Último trimestre', 'Últimos 6 meses', 'Último ano', '2024 completo'];

interface Props {
  darkMode?: boolean;
}

const ExplorarMercado: React.FC<Props> = ({ darkMode = false }) => {
  const [estado, setEstado] = useState('São Paulo');
  const [cnae, setCnae] = useState('Todos os setores');
  const [porte, setPorte] = useState('Todos');
  const [periodo, setPeriodo] = useState('Último trimestre');
  const [chartSub, setChartSub] = useState('São Paulo · Todos os setores · Último trimestre');
  const [lineData, setLineData] = useState(lineEvolutionData['Todos os setores'].data);

  const gridLine = darkMode ? 'rgba(255,255,255,.07)' : '#f0f4f8';
  const tooltipBg = darkMode ? '#1a2535' : '#fff';
  const titleColor = darkMode ? '#e2eaf4' : '#1A3A5C';

  const applyFilters = () => {
    setChartSub(`${estado} · ${cnae} · ${periodo}`);
    const base = lineEvolutionData[cnae] || lineEvolutionData['Todos os setores'];
    const noisy = base.data.map(d => ({
      ...d,
      value: Math.round(d.value + (Math.random() - 0.5) * d.value * 0.12),
    }));
    setLineData(noisy);
  };

  const selectSx = {
    bgcolor: darkMode ? 'rgba(255,255,255,.05)' : '#F4F7FB',
    '& .MuiOutlinedInput-notchedOutline': { borderColor: '#D5E3F0' },
    minWidth: 140,
    fontSize: '.83rem',
  };

  return (
    <Box>
      {/* Header */}
      <Box sx={{ mb: 2.5 }}>
        <Typography sx={{ fontWeight: 800, fontSize: '1.3rem', color: titleColor }}>
          Explorar Mercado
        </Typography>
        <Typography variant="body2" sx={{ color: 'text.secondary', mt: 0.5 }}>
          Filtre e visualize dados de CNPJs por CNAE, porte, estado e período
        </Typography>
      </Box>

      {/* Filters bar */}
      <Card sx={{ mb: 2.5, boxShadow: '0 2px 8px rgba(0,0,0,.04)' }}>
        <CardContent>
          <Box sx={{ display: 'flex', gap: 2, flexWrap: 'wrap', alignItems: 'flex-end' }}>
            <FormControl size="small">
              <InputLabel>Estado</InputLabel>
              <Select value={estado} label="Estado" onChange={(e: SelectChangeEvent) => setEstado(e.target.value)} sx={selectSx}>
                {ESTADOS.map(v => <MenuItem key={v} value={v} sx={{ fontSize: '.83rem' }}>{v}</MenuItem>)}
              </Select>
            </FormControl>

            <FormControl size="small">
              <InputLabel>Setor (CNAE)</InputLabel>
              <Select value={cnae} label="Setor (CNAE)" onChange={(e: SelectChangeEvent) => setCnae(e.target.value)} sx={selectSx}>
                {Object.keys(lineEvolutionData).map(v => <MenuItem key={v} value={v} sx={{ fontSize: '.83rem' }}>{v}</MenuItem>)}
              </Select>
            </FormControl>

            <FormControl size="small">
              <InputLabel>Porte</InputLabel>
              <Select value={porte} label="Porte" onChange={(e: SelectChangeEvent) => setPorte(e.target.value)} sx={selectSx}>
                {PORTES.map(v => <MenuItem key={v} value={v} sx={{ fontSize: '.83rem' }}>{v}</MenuItem>)}
              </Select>
            </FormControl>

            <FormControl size="small">
              <InputLabel>Período</InputLabel>
              <Select value={periodo} label="Período" onChange={(e: SelectChangeEvent) => setPeriodo(e.target.value)} sx={selectSx}>
                {PERIODOS.map(v => <MenuItem key={v} value={v} sx={{ fontSize: '.83rem' }}>{v}</MenuItem>)}
              </Select>
            </FormControl>

            <Button
              variant="contained"
              onClick={applyFilters}
              sx={{
                bgcolor: '#2E6DA4', '&:hover': { bgcolor: '#1A3A5C' },
                height: 40, px: 2.5, fontWeight: 700, borderRadius: '8px',
                textTransform: 'none',
              }}
            >
              🔍 Aplicar Filtros
            </Button>
          </Box>
        </CardContent>
      </Card>

      {/* Charts */}
      <Box sx={{ display: 'grid', gridTemplateColumns: '2fr 1fr', gap: 2 }}>
        <Card sx={{ boxShadow: '0 2px 8px rgba(0,0,0,.05)' }}>
          <CardContent>
            <Typography sx={{ fontWeight: 700, color: titleColor, fontSize: '.9rem', mb: '4px' }}>
              Evolução de Aberturas por Mês
            </Typography>
            <Typography sx={{ fontSize: '.75rem', color: 'text.secondary', mb: 1.5 }}>
              {chartSub}
            </Typography>
            <ResponsiveContainer width="100%" height={220}>
              <LineChart data={lineData} margin={{ top: 5, right: 10, left: -10, bottom: 5 }}>
                <CartesianGrid strokeDasharray="3 3" stroke={gridLine} />
                <XAxis dataKey="mes" tick={{ fontSize: 10 }} />
                <YAxis tick={{ fontSize: 10 }} />
                <Tooltip
                  contentStyle={{ background: tooltipBg, borderRadius: 8, fontSize: '.8rem', border: '1px solid #D5E3F0' }}
                />
                <Line
                  type="monotone"
                  dataKey="value"
                  stroke="#2E6DA4"
                  strokeWidth={2.5}
                  dot={{ r: 5, fill: '#2E6DA4', strokeWidth: 0 }}
                  activeDot={{ r: 7 }}
                />
              </LineChart>
            </ResponsiveContainer>
          </CardContent>
        </Card>

        <Card sx={{ boxShadow: '0 2px 8px rgba(0,0,0,.05)' }}>
          <CardContent>
            <Typography sx={{ fontWeight: 700, color: titleColor, fontSize: '.9rem', mb: '4px' }}>
              Top 5 Cidades
            </Typography>
            <Typography sx={{ fontSize: '.75rem', color: 'text.secondary', mb: 1.5 }}>
              Volume de abertura no período selecionado
            </Typography>
            <ResponsiveContainer width="100%" height={220}>
              <BarChart
                data={cidadesBarData}
                layout="vertical"
                margin={{ top: 5, right: 10, left: 10, bottom: 5 }}
              >
                <CartesianGrid strokeDasharray="3 3" stroke={gridLine} />
                <XAxis type="number" tick={{ fontSize: 10 }} />
                <YAxis type="category" dataKey="name" tick={{ fontSize: 10 }} width={78} />
                <Tooltip
                  contentStyle={{ background: tooltipBg, borderRadius: 8, fontSize: '.8rem', border: '1px solid #D5E3F0' }}
                />
                <Bar dataKey="value" radius={[0, 6, 6, 0]}>
                  {cidadesBarData.map((_, idx) => (
                    <Cell key={idx} fill={COLORS[idx % COLORS.length]} />
                  ))}
                </Bar>
              </BarChart>
            </ResponsiveContainer>
          </CardContent>
        </Card>
      </Box>
    </Box>
  );
};

export default ExplorarMercado;
