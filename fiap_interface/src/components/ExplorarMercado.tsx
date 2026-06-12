import React, { useState, useEffect } from 'react';
import {
  Box, Card, CardContent, Typography,
  Select, MenuItem, FormControl, InputLabel, Button, SelectChangeEvent, Skeleton,
} from '@mui/material';
import {
  LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip,
  BarChart, Bar, Cell, ResponsiveContainer,
} from 'recharts';
import {
  fetchFiltros, fetchEvolution, fetchCidades,
  FiltroOptions, EvolutionPoint, BarDatum,
} from '../services/marketApi';

const TODOS = 'Todos';
const TODOS_SETORES = 'Todos os setores';

interface Props {
  darkMode?: boolean;
}

const ExplorarMercado: React.FC<Props> = ({ darkMode = false }) => {
  // Opções dos filtros (vêm da API / base de dados). "Todos" é sempre o padrão.
  const [options, setOptions] = useState<FiltroOptions>({
    estados: [TODOS], portes: [TODOS], periodos: [TODOS], setores: [TODOS_SETORES],
  });

  const [estado, setEstado] = useState(TODOS);
  const [cnae, setCnae] = useState(TODOS_SETORES);
  const [porte, setPorte] = useState(TODOS);
  const [periodo, setPeriodo] = useState(TODOS);
  const [chartSub, setChartSub] = useState(`${TODOS} · ${TODOS_SETORES} · ${TODOS}`);
  const [lineData, setLineData] = useState<EvolutionPoint[]>([]);
  const [cidades, setCidades] = useState<BarDatum[]>([]);
  const [loading, setLoading] = useState(true);

  const gridLine = darkMode ? 'rgba(255,255,255,.07)' : '#f0f4f8';
  const tooltipBg = darkMode ? '#1a2535' : '#fff';
  const titleColor = darkMode ? '#e2eaf4' : '#1A3A5C';

  // Busca os dados (já filtrados pelo backend) e atualiza a tela.
  const load = async (filtros: { uf: string; porte: string; cnae: string; periodo: string }) => {
    setLoading(true);
    try {
      const [evo, cid] = await Promise.all([fetchEvolution(filtros), fetchCidades(filtros)]);
      setLineData(evo.data);
      setCidades(cid);
      setChartSub(`${filtros.uf} · ${filtros.cnae} · ${filtros.periodo}`);
    } catch (err) {
      console.error('Erro ao carregar dados de mercado:', err);
    } finally {
      setLoading(false);
    }
  };

  // Na montagem: carrega as opções de filtro da API e os dados com o padrão "Todos".
  useEffect(() => {
    (async () => {
      try {
        const opts = await fetchFiltros();
        setOptions(opts);
      } catch (err) {
        console.error('Erro ao carregar filtros:', err);
      }
      await load({ uf: TODOS, porte: TODOS, cnae: TODOS_SETORES, periodo: TODOS });
    })();
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  const applyFilters = () => load({ uf: estado, porte, cnae, periodo });

  const selectSx = {
    bgcolor: darkMode ? 'rgba(255,255,255,.05)' : '#F4F7FB',
    '& .MuiOutlinedInput-notchedOutline': { borderColor: '#D5E3F0' },
    minWidth: 140,
    fontSize: '.83rem',
  };

  return (
    <Box>
      {/* Header */}
      <Box sx={{ mb: 2.5, display: 'flex', justifyContent: 'space-between', alignItems: 'flex-start', gap: 2, flexWrap: 'wrap' }}>
        <Box>
          <Typography sx={{ fontWeight: 800, fontSize: '1.3rem', color: titleColor }}>
            Explorar Mercado
          </Typography>
          <Typography variant="body2" sx={{ color: 'text.secondary', mt: 0.5 }}>
            Filtre e visualize dados de CNPJs por CNAE, porte, estado e período
          </Typography>
        </Box>
        <Button
          onClick={applyFilters}
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

      {/* Filters bar */}
      <Card sx={{ mb: 2.5, boxShadow: '0 2px 8px rgba(0,0,0,.04)' }}>
        <CardContent>
          <Box sx={{ display: 'flex', gap: 2, flexWrap: 'wrap', alignItems: 'flex-end' }}>
            <FormControl size="small">
              <InputLabel>Estado</InputLabel>
              <Select value={estado} label="Estado" onChange={(e: SelectChangeEvent) => setEstado(e.target.value)} sx={selectSx}>
                {options.estados.map(v => <MenuItem key={v} value={v} sx={{ fontSize: '.83rem' }}>{v}</MenuItem>)}
              </Select>
            </FormControl>

            <FormControl size="small">
              <InputLabel>Setor (CNAE)</InputLabel>
              <Select value={cnae} label="Setor (CNAE)" onChange={(e: SelectChangeEvent) => setCnae(e.target.value)} sx={selectSx}>
                {options.setores.map(v => <MenuItem key={v} value={v} sx={{ fontSize: '.83rem' }}>{v}</MenuItem>)}
              </Select>
            </FormControl>

            <FormControl size="small">
              <InputLabel>Porte</InputLabel>
              <Select value={porte} label="Porte" onChange={(e: SelectChangeEvent) => setPorte(e.target.value)} sx={selectSx}>
                {options.portes.map(v => <MenuItem key={v} value={v} sx={{ fontSize: '.83rem' }}>{v}</MenuItem>)}
              </Select>
            </FormControl>

            <FormControl size="small">
              <InputLabel>Período</InputLabel>
              <Select value={periodo} label="Período" onChange={(e: SelectChangeEvent) => setPeriodo(e.target.value)} sx={selectSx}>
                {options.periodos.map(v => <MenuItem key={v} value={v} sx={{ fontSize: '.83rem' }}>{v}</MenuItem>)}
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
      {loading ? (
        <Box sx={{ display: 'grid', gridTemplateColumns: { xs: '1fr', md: '2fr 1fr' }, gap: 2 }}>
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
      ) : (
      <Box sx={{ display: 'grid', gridTemplateColumns: { xs: '1fr', md: '2fr 1fr' }, gap: 2 }}>
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
                  contentStyle={{ background: tooltipBg, borderRadius: 8, fontSize: '.8rem', border: '1px solid #D5E3F0', color: titleColor }}
                  labelStyle={{ color: titleColor }}
                  itemStyle={{ color: titleColor }}
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
                data={cidades}
                layout="vertical"
                margin={{ top: 5, right: 10, left: 10, bottom: 5 }}
              >
                <CartesianGrid strokeDasharray="3 3" stroke={gridLine} />
                <XAxis type="number" tick={{ fontSize: 10 }} />
                <YAxis type="category" dataKey="name" tick={{ fontSize: 10 }} width={78} />
                <Tooltip
                  contentStyle={{ background: tooltipBg, borderRadius: 8, fontSize: '.8rem', border: '1px solid #D5E3F0', color: titleColor }}
                  labelStyle={{ color: titleColor }}
                  itemStyle={{ color: titleColor }}
                />
                <Bar dataKey="value" radius={[0, 6, 6, 0]}>
                  {cidades.map((entry, idx) => (
                    <Cell key={idx} fill={entry.color} />
                  ))}
                </Bar>
              </BarChart>
            </ResponsiveContainer>
          </CardContent>
        </Card>
      </Box>
      )}
    </Box>
  );
};

export default ExplorarMercado;
