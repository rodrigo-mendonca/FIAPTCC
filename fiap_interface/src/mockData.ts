export const metricsData = [
  { label: "CNPJs Ativos no Brasil", value: "21,4M", sub: "↑ +3,2% vs. ano anterior", icon: "🏢", variant: "blue" as const },
  { label: "Aberturas (último trimestre)", value: "412K", sub: "↑ +8,7% vs. mesmo período", icon: "📈", variant: "green" as const },
  { label: "CNAEs em Alta (SP)", value: "47", sub: "Saúde, Tech e Educação lideram", icon: "🎯", variant: "orange" as const },
  { label: "Sua última consulta", value: "2s", sub: "Tempo médio de resposta", icon: "⚡", variant: "teal" as const },
];

export const setoresBarData = [
  { name: "Saúde", value: 42.3 },
  { name: "Tecnologia", value: 38.1 },
  { name: "Educação", value: 31.7 },
  { name: "Alimentação", value: 28.4 },
  { name: "Financeiro", value: 24.9 },
  { name: "Construção", value: 22.1 },
  { name: "Transporte", value: 18.6 },
  { name: "Varejo", value: 14.2 },
];

export const porteDonutData = [
  { name: "MEI", value: 52 },
  { name: "Microempresa", value: 28 },
  { name: "Peq. Porte", value: 16 },
  { name: "Grande", value: 4 },
];

export const insightsData = [
  {
    icon: "🏥",
    title: "Saúde & Bem-estar em alta",
    text: "CNAE 86.30 cresceu 23% em aberturas este trimestre em SP – maior alta do ano.",
    badge: "↑ +23% Q4",
    badgeType: "up" as const,
    variant: "green" as const,
  },
  {
    icon: "💡",
    title: "Tecnologia mantém ritmo",
    text: "Desenvolvimento de software (CNAE 62.01) soma 18.400 novas empresas em 2025.",
    badge: "Oportunidade B2B",
    badgeType: "new" as const,
    variant: "blue" as const,
  },
  {
    icon: "⚠️",
    title: "Varejo físico desacelera",
    text: "CNAEs de varejo registraram queda de 4% em novas aberturas vs. Q3.",
    badge: "↓ -4% Q4",
    badgeType: "down" as const,
    variant: "orange" as const,
  },
];

export const lineEvolutionData: Record<string, { data: { mes: string; value: number }[] }> = {
  "Todos os setores": {
    data: [
      { mes: "Out", value: 12400 },
      { mes: "Nov", value: 14800 },
      { mes: "Dez", value: 16200 },
    ],
  },
  "Saúde e Bem-estar": {
    data: [
      { mes: "Out", value: 3200 },
      { mes: "Nov", value: 3900 },
      { mes: "Dez", value: 4500 },
    ],
  },
  "Tecnologia": {
    data: [
      { mes: "Out", value: 2800 },
      { mes: "Nov", value: 3100 },
      { mes: "Dez", value: 3400 },
    ],
  },
  "Educação": {
    data: [
      { mes: "Out", value: 1900 },
      { mes: "Nov", value: 2200 },
      { mes: "Dez", value: 2600 },
    ],
  },
  "Varejo": {
    data: [
      { mes: "Out", value: 1400 },
      { mes: "Nov", value: 1200 },
      { mes: "Dez", value: 1100 },
    ],
  },
};

export const cidadesBarData = [
  { name: "São Paulo", value: 9840 },
  { name: "Guarulhos", value: 3210 },
  { name: "Campinas", value: 2780 },
  { name: "Santo André", value: 1940 },
  { name: "Osasco", value: 1620 },
];

export const exportChartData = [
  { name: "Saúde", value: 42.3 },
  { name: "Tech", value: 38.1 },
  { name: "Educação", value: 31.7 },
  { name: "Alimentação", value: 28.4 },
  { name: "Financeiro", value: 24.9 },
];
