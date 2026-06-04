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

export type EvolutionData = Record<string, { data: EvolutionPoint[] }>;

// ===================== Helper =====================

async function getJson<T>(path: string): Promise<T> {
  const response = await fetch(`${API_URL}${path}`);
  if (!response.ok) {
    throw new Error(`Falha ao buscar ${path}: ${response.status}`);
  }
  return response.json() as Promise<T>;
}

// ===================== Endpoints =====================

export const fetchMetrics = () => getJson<Metric[]>('/api/market/metrics');
export const fetchSetores = () => getJson<BarDatum[]>('/api/market/setores');
export const fetchPorte = () => getJson<BarDatum[]>('/api/market/porte');
export const fetchInsights = () => getJson<Insight[]>('/api/market/insights');
export const fetchEvolution = () => getJson<EvolutionData>('/api/market/evolution');
export const fetchCidades = () => getJson<BarDatum[]>('/api/market/cidades');
export const fetchExportChart = () => getJson<BarDatum[]>('/api/market/export-chart');
