"""
SQL Factory for managing database connections and executing queries.
This factory uses SQLAlchemy to connect to PostgreSQL database and execute queries.
"""

from sqlalchemy import create_engine, text
from sqlalchemy.orm import sessionmaker
import json
import os

class SQLFactory:
    def __init__(self):
        """
        Initialize the SQL Factory with database path from DATABASE_URL environment variable
        """
        # Use DATABASE_URL environment variable for PostgreSQL connection
        self.db_path = os.getenv('DATABASE_URL')
        
        # Check if DATABASE_URL is set and not empty
        if not self.db_path:
            raise ValueError("DATABASE_URL environment variable is not set. Please configure the database connection URL.")
        
        # Create engine and session factory
        self.engine = create_engine(self.db_path)
        self.SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=self.engine)
    
    def execute_query(self, query, params=None):
        """
        Execute a SQL query and return results as JSON array

        Args:
            query (str): SQL query to execute
            params (dict, optional): Named bind parameters (:name) for the query

        Returns:
            list: Array of dictionaries representing the query results
        """
        try:
            # Create a new session
            db_session = self.SessionLocal()

            # Execute the query
            result = db_session.execute(text(query), params or {})
            
            # Get column names
            columns = result.keys()
            
            # Convert to array of JSON objects
            rows = []
            for row in result:
                row_dict = dict(zip(columns, row))
                rows.append(row_dict)
            
            return rows
            
        except Exception as e:
            raise Exception(f"Error executing query: {str(e)}")
        finally:
            db_session.close()
    
    def get_engine(self):
        """
        Get the SQLAlchemy engine

        Returns:
            Engine: The SQLAlchemy engine instance
        """
        return self.engine

    # ==================================================================
    # Dados de mercado (CNPJs / aberturas de empresas)
    # ------------------------------------------------------------------
    # Consumidos pelas telas Dashboard, Explorar Mercado e Exportar.
    # Os números são derivados da base dimensional (star schema):
    #   fat_empresas_mensal  – fato mensal (nova_empresa, baixada, ...)
    #   dim_empresas_mensal  – snapshot mensal (porte, situacao, cnae...)
    #   dim_cnaes / dim_cnaes_segmento / dim_municipios / dim_competencias
    #
    # Códigos observados na base:
    #   situacao: 2 = ATIVA, 8 = BAIXADA  (padrão Receita Federal)
    #   porte:    1 = MEI/Micro (mei=true), 2 = Pequeno Porte, 3 = Grande
    #
    # As cores e ícones permanecem definidos aqui: são metadados de
    # apresentação, não dados de negócio. O formato de retorno é idêntico
    # ao que o frontend já consome (fiap_interface/src/services/marketApi.ts).
    #
    # São instance methods pois dependem da conexão com o banco.
    # ==================================================================

    # Paleta usada para colorir séries/barras de forma determinística.
    _PALETTE = [
        "#2E6DA4", "#4A9FD4", "#27AE60", "#E67E22", "#8E44AD",
        "#1ABC9C", "#c8e73c", "#F39C12", "#a42e2e", "#16A085",
    ]

    _MESES_PT = ["Jan", "Fev", "Mar", "Abr", "Mai", "Jun",
                 "Jul", "Ago", "Set", "Out", "Nov", "Dez"]

    @classmethod
    def _mes_label(cls, competencia):
        """Converte 'YYYYMM' (ex.: '202605') em 'Mai/26'."""
        competencia = str(competencia)
        mes = int(competencia[4:6])
        return f"{cls._MESES_PT[mes - 1]}/{competencia[2:4]}"

    def _color(self, i):
        return self._PALETTE[i % len(self._PALETTE)]

    @staticmethod
    def _delta_sub(cur, base, periodo):
        """Monta o texto de variação percentual de um cartão de métrica."""
        if not base:
            return "sem base comparativa"
        pct = (cur - base) / base * 100
        seta = "↑" if pct >= 0 else "↓"
        return f"{seta} {pct:+.1f}% {periodo}"

    def _aberturas_por_segmento(self):
        """Total de novas empresas (aberturas) agrupado por segmento de CNAE."""
        return self.execute_query("""
            SELECT s.denominacao AS nome, COUNT(*) AS value
            FROM fat_empresas_mensal f
            JOIN dim_empresas_mensal em
              ON em.id_dim_empresa = f.id_dim_empresa
             AND em.id_dim_competencia = f.id_dim_competencia
            JOIN dim_cnaes c ON c.id = em.id_dim_cnae
            JOIN dim_cnaes_segmento s ON s.id = c.id_dim_cnae_segmento
            WHERE f.nova_empresa
            GROUP BY s.denominacao
            ORDER BY value DESC
        """)

    def get_market_metrics(self):
        """Cartões de métricas da visão geral do mercado (Dashboard)."""
        row = self.execute_query("""
            WITH ult  AS (SELECT MAX(id) AS id FROM dim_competencias),
                 prev AS (SELECT MAX(id) AS id FROM dim_competencias
                          WHERE id < (SELECT id FROM ult)),
                 ano  AS (SELECT (SELECT id FROM ult) - 12 AS id)
            SELECT
              (SELECT COUNT(*) FROM dim_empresas_mensal
                 WHERE id_dim_competencia = (SELECT id FROM ult) AND situacao = 2) AS ativas_ult,
              (SELECT COUNT(*) FROM dim_empresas_mensal
                 WHERE id_dim_competencia = (SELECT id FROM ano) AND situacao = 2) AS ativas_ano,
              (SELECT COUNT(*) FROM fat_empresas_mensal
                 WHERE id_dim_competencia = (SELECT id FROM ult) AND nova_empresa) AS novas_ult,
              (SELECT COUNT(*) FROM fat_empresas_mensal
                 WHERE id_dim_competencia = (SELECT id FROM prev) AND nova_empresa) AS novas_prev,
              (SELECT COUNT(DISTINCT id_dim_cnae) FROM dim_empresas_mensal
                 WHERE id_dim_competencia = (SELECT id FROM ult)) AS cnaes_ult,
              (SELECT COUNT(*) FROM fat_empresas_mensal
                 WHERE id_dim_competencia = (SELECT id FROM ult) AND baixada) AS baixas_ult
        """)[0]

        def fmt(n):
            return f"{int(n):,}".replace(",", ".")

        return [
            {"label": "CNPJs Ativos", "value": fmt(row["ativas_ult"]),
             "sub": self._delta_sub(row["ativas_ult"], row["ativas_ano"], "vs. ano anterior"),
             "icon": "🏢", "topColor": "#2E6DA4"},
            {"label": "Aberturas (último mês)", "value": fmt(row["novas_ult"]),
             "sub": self._delta_sub(row["novas_ult"], row["novas_prev"], "vs. mês anterior"),
             "icon": "📈", "topColor": "#27AE60"},
            {"label": "CNAEs distintos", "value": fmt(row["cnaes_ult"]),
             "sub": "Ativos no último mês", "icon": "🎯", "topColor": "#E67E22"},
            {"label": "Baixas (último mês)", "value": fmt(row["baixas_ult"]),
             "sub": "Encerramentos no período", "icon": "📉", "topColor": "#1ABC9C"},
        ]

    def get_market_setores(self):
        """Aberturas por setor – top CNAEs (Dashboard, gráfico de barras)."""
        rows = self._aberturas_por_segmento()
        return [
            {"name": r["nome"], "value": int(r["value"]), "color": self._color(i),
             "unit": "", "tooltipLabel": "Novas empresas"}
            for i, r in enumerate(rows)
        ]

    def get_market_porte(self):
        """Distribuição de empresas por porte na última competência (gráfico de rosca)."""
        rows = self.execute_query("""
            WITH ult AS (SELECT MAX(id_dim_competencia) AS c FROM dim_empresas_mensal)
            SELECT
              CASE WHEN mei THEN 'MEI'
                   WHEN porte = 2 THEN 'Pequeno Porte'
                   WHEN porte = 3 THEN 'Grande Porte'
                   WHEN porte = 1 THEN 'Microempresa'
                   ELSE 'Outros' END AS bucket,
              COUNT(*) AS n
            FROM dim_empresas_mensal
            WHERE id_dim_competencia = (SELECT c FROM ult)
            GROUP BY bucket
            ORDER BY n DESC
        """)
        cores = {
            "MEI": "#4A9FD4", "Microempresa": "#2E6DA4", "Pequeno Porte": "#27AE60",
            "Grande Porte": "#E67E22", "Outros": "#95A5A6",
        }
        total = sum(r["n"] for r in rows) or 1
        return [
            {"name": r["bucket"], "value": round(r["n"] * 100 / total),
             "color": cores.get(r["bucket"], "#95A5A6"), "unit": "%", "tooltipLabel": r["bucket"]}
            for r in rows
        ]

    def get_market_insights(self):
        """Cartões de insights do mercado (Dashboard), derivados das aberturas por setor."""
        rows = self._aberturas_por_segmento()
        if not rows:
            return []

        total = sum(r["value"] for r in rows) or 1
        top = rows[0]
        menor = rows[-1]
        tech = next((r for r in rows if "Informa" in r["nome"]), None)

        insights = [{
            "icon": "🚀",
            "title": f"{top['nome']} lidera as aberturas",
            "text": (f"O segmento '{top['nome']}' concentra {top['value']} novas empresas "
                     f"({top['value'] * 100 / total:.0f}% do total de aberturas)."),
            "badge": f"{top['value'] * 100 / total:.0f}% do total",
            "badgeBg": "#D5F5E3", "badgeColor": "#27AE60", "borderColor": "#27AE60",
        }]

        if tech and tech is not top:
            insights.append({
                "icon": "💡",
                "title": "Tecnologia mantém ritmo",
                "text": (f"'{tech['nome']}' soma {tech['value']} novas empresas no período, "
                         "uma oportunidade relevante para negócios B2B."),
                "badge": "Oportunidade B2B",
                "badgeBg": "#EBF5FB", "badgeColor": "#2E6DA4", "borderColor": "#2E6DA4",
            })

        if menor is not top:
            insights.append({
                "icon": "⚠️",
                "title": f"{menor['nome']} com menor volume",
                "text": (f"'{menor['nome']}' registrou apenas {menor['value']} novas aberturas, "
                         "o menor volume entre os segmentos."),
                "badge": "Menor volume",
                "badgeBg": "#FADBD8", "badgeColor": "#E74C3C", "borderColor": "#E67E22",
            })

        return insights

    # ------------------------------------------------------------------
    # Filtros da tela Explorar Mercado
    # ------------------------------------------------------------------
    # As opções de Estado/Porte/Setor vêm da base; Período é uma janela
    # temporal fixa. Em todos os campos "Todos" (ou "Todos os setores")
    # é a primeira opção e o padrão da tela.

    # Janela (nº de competências mais recentes) por período. "Todos" = None.
    _PERIODO_N = {
        "Último trimestre": 3,
        "Últimos 6 meses": 6,
        "Último ano": 12,
    }

    # Ordem de exibição dos portes no filtro.
    _PORTE_ORDEM = ["MEI", "Microempresa", "Pequeno Porte", "Grande Porte", "Outros"]

    def get_market_filtros(self):
        """Opções dos filtros da tela Explorar Mercado, derivadas da base."""
        estados = self.execute_query("""
            SELECT DISTINCT mun.uf
            FROM dim_empresas_mensal em
            JOIN dim_municipios mun ON mun.id = em.id_dim_municipio
            ORDER BY mun.uf
        """)
        portes = self.execute_query("""
            SELECT DISTINCT
              CASE WHEN mei THEN 'MEI'
                   WHEN porte = 2 THEN 'Pequeno Porte'
                   WHEN porte = 3 THEN 'Grande Porte'
                   WHEN porte = 1 THEN 'Microempresa'
                   ELSE 'Outros' END AS bucket
            FROM dim_empresas_mensal
        """)
        setores = self.execute_query("""
            SELECT DISTINCT seg.denominacao AS nome
            FROM dim_empresas_mensal em
            JOIN dim_cnaes cn ON cn.id = em.id_dim_cnae
            JOIN dim_cnaes_segmento seg ON seg.id = cn.id_dim_cnae_segmento
            ORDER BY seg.denominacao
        """)
        buckets = {r["bucket"] for r in portes}
        return {
            "estados": ["Todos"] + [r["uf"] for r in estados],
            "portes": ["Todos"] + [b for b in self._PORTE_ORDEM if b in buckets],
            "periodos": ["Todos", "Último trimestre", "Últimos 6 meses", "Último ano"],
            "setores": ["Todos os setores"] + [r["nome"] for r in setores],
        }

    def _filtros_clausulas(self, uf, porte, cnae, periodo):
        """Monta cláusulas WHERE + bind params para os filtros de mercado.

        As consultas que usam este helper devem expor os aliases:
          mun (dim_municipios), em (dim_empresas_mensal),
          seg (dim_cnaes_segmento) e c (dim_competencias).
        """
        clausulas, params = [], {}
        if uf and uf != "Todos":
            clausulas.append("mun.uf = :uf")
            params["uf"] = uf
        if porte and porte != "Todos":
            mapa = {
                "MEI": "em.mei = true",
                "Microempresa": "(em.porte = 1 AND em.mei = false)",
                "Pequeno Porte": "em.porte = 2",
                "Grande Porte": "em.porte = 3",
            }
            if porte in mapa:
                clausulas.append(mapa[porte])
        if cnae and cnae != "Todos os setores":
            clausulas.append("seg.denominacao = :cnae")
            params["cnae"] = cnae
        n = self._PERIODO_N.get(periodo)
        if n:
            clausulas.append("c.id > (SELECT MAX(id) FROM dim_competencias) - :pn")
            params["pn"] = n
        return clausulas, params

    def get_market_evolution(self, uf="Todos", porte="Todos",
                             cnae="Todos os setores", periodo="Todos"):
        """Evolução de aberturas por mês (Explorar Mercado), filtrada pela base."""
        # Eixo temporal: todas as competências da janela do período (ascendente),
        # para que meses sem abertura apareçam como zero.
        n = self._PERIODO_N.get(periodo)
        comp_rows = self.execute_query(
            "SELECT competencia FROM dim_competencias ORDER BY id DESC"
            + (" LIMIT :n" if n else ""),
            {"n": n} if n else None,
        )
        competencias = sorted(r["competencia"] for r in comp_rows)

        # O período já está refletido no eixo; aqui filtramos só uf/porte/cnae.
        clausulas, params = self._filtros_clausulas(uf, porte, cnae, periodo=None)
        where = " AND ".join(["f.nova_empresa"] + clausulas)
        rows = self.execute_query(f"""
            SELECT c.competencia, COUNT(*) AS value
            FROM fat_empresas_mensal f
            JOIN dim_competencias c ON c.id = f.id_dim_competencia
            JOIN dim_empresas_mensal em
              ON em.id_dim_empresa = f.id_dim_empresa
             AND em.id_dim_competencia = f.id_dim_competencia
            JOIN dim_municipios mun ON mun.id = em.id_dim_municipio
            JOIN dim_cnaes cn ON cn.id = em.id_dim_cnae
            JOIN dim_cnaes_segmento seg ON seg.id = cn.id_dim_cnae_segmento
            WHERE {where}
            GROUP BY c.competencia
        """, params)

        valor = {r["competencia"]: r["value"] for r in rows}
        return {
            "data": [
                {"mes": self._mes_label(c), "value": int(valor.get(c, 0))}
                for c in competencias
            ]
        }

    def get_market_cidades(self, uf="Todos", porte="Todos",
                           cnae="Todos os setores", periodo="Todos"):
        """Top 5 cidades por volume de aberturas (Explorar Mercado), filtrada pela base."""
        clausulas, params = self._filtros_clausulas(uf, porte, cnae, periodo)
        where = " AND ".join(["f.nova_empresa"] + clausulas)
        rows = self.execute_query(f"""
            SELECT mun.nome, mun.uf, COUNT(*) AS value
            FROM fat_empresas_mensal f
            JOIN dim_competencias c ON c.id = f.id_dim_competencia
            JOIN dim_empresas_mensal em
              ON em.id_dim_empresa = f.id_dim_empresa
             AND em.id_dim_competencia = f.id_dim_competencia
            JOIN dim_municipios mun ON mun.id = em.id_dim_municipio
            JOIN dim_cnaes cn ON cn.id = em.id_dim_cnae
            JOIN dim_cnaes_segmento seg ON seg.id = cn.id_dim_cnae_segmento
            WHERE {where}
            GROUP BY mun.nome, mun.uf
            ORDER BY value DESC
            LIMIT 5
        """, params)
        return [
            {"name": f"{r['nome']}/{r['uf']}", "value": int(r["value"]), "color": self._color(i)}
            for i, r in enumerate(rows)
        ]

    def get_market_export_chart(self):
        """Dados do gráfico de aberturas por setor usado na tela de Exportar (top 5)."""
        rows = self._aberturas_por_segmento()[:5]
        return [
            {"name": r["nome"], "value": int(r["value"]), "color": self._color(i),
             "unit": "", "tooltipLabel": "Aberturas"}
            for i, r in enumerate(rows)
        ]


# Create a singleton instance.
# Protegido para não derrubar a importação do módulo quando DATABASE_URL não
# está configurada — os métodos de dados de mercado (@staticmethod) continuam
# utilizáveis mesmo sem conexão com o banco.
try:
    sql_factory = SQLFactory()
except Exception as e:
    print(f"[WARN] SQLFactory não inicializado (sem DATABASE_URL?): {e}")
    sql_factory = None