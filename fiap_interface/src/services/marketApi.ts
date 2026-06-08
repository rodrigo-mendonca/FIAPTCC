// Cliente das APIs de dados de mercado (CNPJs / aberturas de empresas).
// Substitui o antigo mockData.ts — os dados agora vêm da fiap_api.

const API_URL = process.env.REACT_APP_API_URL;

// ===================== Tipos =====================

export interface Metric {
  label: string;
  value: string;
  sub: string;
  icon: string;
  topColor: string;
}

export interface BarDatum {
  name: string;
  value: number;
  color: string;
  unit?: string;
  tooltipLabel?: string;
}

export interface Insight {
  icon: string;
  title: string;
  text: string;
  badge: string;
  badgeBg: string;
  badgeColor: string;
  borderColor: string;
}

export interface EvolutionPoint {
  mes: string;
  value: number;
}

// Série única da evolução, já filtrada pelo backend.
export interface EvolutionSeries {
  data: EvolutionPoint[];
}

// Opções dos filtros da tela Explorar Mercado (todas vindas da base).
export interface FiltroOptions {
  estados: string[];
  portes: string[];
  periodos: string[];
  setores: string[];
}

// Filtros aplicados às consultas de evolução/cidades.
export interface MarketFiltros {
  uf: string;
  porte: string;
  cnae: string;
  periodo: string;
}

// ===================== Helper =====================

async function getJson<T>(path: string): Promise<T> {
  const response = await fetch(`${API_URL}${path}`);
  if (!response.ok) {
    throw new Error(`Falha ao buscar ${path}: ${response.status}`);
  }
  return response.json() as Promise<T>;
}

function queryString(filtros: MarketFiltros): string {
  const params = new URLSearchParams({
    uf: filtros.uf,
    porte: filtros.porte,
    cnae: filtros.cnae,
    periodo: filtros.periodo,
  });
  return `?${params.toString()}`;
}

// ===================== Endpoints =====================

export const fetchMetrics = () => getJson<Metric[]>('/api/market/metrics');
export const fetchSetores = () => getJson<BarDatum[]>('/api/market/setores');
export const fetchPorte = () => getJson<BarDatum[]>('/api/market/porte');
export const fetchInsights = () => getJson<Insight[]>('/api/market/insights');
export const fetchExportChart = () => getJson<BarDatum[]>('/api/market/export-chart');

export const fetchFiltros = () => getJson<FiltroOptions>('/api/market/filtros');
export const fetchEvolution = (filtros: MarketFiltros) =>
  getJson<EvolutionSeries>(`/api/market/evolution${queryString(filtros)}`);
export const fetchCidades = (filtros: MarketFiltros) =>
  getJson<BarDatum[]>(`/api/market/cidades${queryString(filtros)}`);
