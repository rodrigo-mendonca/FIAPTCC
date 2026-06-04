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
    
    def execute_query(self, query):
        """
        Execute a SQL query and return results as JSON array
        
        Args:
            query (str): SQL query to execute
            
        Returns:
            list: Array of dictionaries representing the query results
        """
        try:
            # Create a new session
            db_session = self.SessionLocal()
            
            # Execute the query
            result = db_session.execute(text(query))
            
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
    # São retornados como dados de seed (mock) pois não há, por ora, uma
    # fonte real para essas métricas de mercado (a base loja_db cobre
    # apenas clientes/produtos/vendas). Quando uma tabela existir, basta
    # trocar o corpo do método por self.execute_query("SELECT ..."),
    # mantendo o mesmo formato de retorno — o frontend não muda.
    #
    # São @staticmethod de propósito: não dependem de conexão com o banco,
    # então funcionam mesmo sem DATABASE_URL configurada.
    # ==================================================================

    @staticmethod
    def get_market_metrics():
        """Cartões de métricas da visão geral do mercado (Dashboard)."""
        return [
            {"label": "CNPJs Ativos no Brasil", "value": "21,4M", "sub": "↑ +3,2% vs. ano anterior", "icon": "🏢", "topColor": "#2E6DA4"},
            {"label": "Aberturas (último trimestre)", "value": "412K", "sub": "↑ +8,7% vs. mesmo período", "icon": "📈", "topColor": "#27AE60"},
            {"label": "CNAEs em Alta (SP)", "value": "47", "sub": "Saúde, Tech e Educação lideram", "icon": "🎯", "topColor": "#E67E22"},
            {"label": "Sua última consulta", "value": "2s", "sub": "Tempo médio de resposta", "icon": "⚡", "topColor": "#1ABC9C"},
        ]

    @staticmethod
    def get_market_setores():
        """Aberturas por setor – top CNAEs (Dashboard, gráfico de barras)."""
        return [
            {"name": "Saúde", "value": 42.3, "color": "#a42e2e", "unit": "k", "tooltipLabel": "Novas empresas"},
            {"name": "Tecnologia", "value": 38.1, "color": "#4A9FD4", "unit": "k", "tooltipLabel": "Novas empresas"},
            {"name": "Educação", "value": 31.7, "color": "#27AE60", "unit": "k", "tooltipLabel": "Novas empresas"},
            {"name": "Alimentação", "value": 28.4, "color": "#E67E22", "unit": "k", "tooltipLabel": "Novas empresas"},
            {"name": "Financeiro", "value": 24.9, "color": "#8E44AD", "unit": "k", "tooltipLabel": "Novas empresas"},
            {"name": "Construção", "value": 22.1, "color": "#1ABC9C", "unit": "k", "tooltipLabel": "Novas empresas"},
            {"name": "Transporte", "value": 18.6, "color": "#c8e73c", "unit": "k", "tooltipLabel": "Novas empresas"},
            {"name": "Varejo", "value": 14.2, "color": "#F39C12", "unit": "k", "tooltipLabel": "Novas empresas"},
        ]

    @staticmethod
    def get_market_porte():
        """Distribuição de empresas por porte (Dashboard, gráfico de rosca)."""
        return [
            {"name": "MEI", "value": 52, "color": "#4A9FD4", "unit": "%", "tooltipLabel": "MEI"},
            {"name": "Microempresa", "value": 28, "color": "#2E6DA4", "unit": "%", "tooltipLabel": "Microempresa"},
            {"name": "Peq. Porte", "value": 16, "color": "#27AE60", "unit": "%", "tooltipLabel": "Pequena Empresa"},
            {"name": "Grande", "value": 4, "color": "#E67E22", "unit": "%", "tooltipLabel": "Grande Empresa"},
        ]

    @staticmethod
    def get_market_insights():
        """Cartões de insights do mercado (Dashboard)."""
        return [
            {
                "icon": "🏥",
                "title": "Saúde & Bem-estar em alta",
                "text": "CNAE 86.30 cresceu 23% em aberturas este trimestre em SP – maior alta do ano.",
                "badge": "↑ +23% Q4",
                "badgeBg": "#D5F5E3",
                "badgeColor": "#27AE60",
                "borderColor": "#27AE60",
            },
            {
                "icon": "💡",
                "title": "Tecnologia mantém ritmo",
                "text": "Desenvolvimento de software (CNAE 62.01) soma 18.400 novas empresas em 2026.",
                "badge": "Oportunidade B2B",
                "badgeBg": "#EBF5FB",
                "badgeColor": "#2E6DA4",
                "borderColor": "#2E6DA4",
            },
            {
                "icon": "⚠️",
                "title": "Varejo físico desacelera",
                "text": "CNAEs de varejo registraram queda de 4% em novas aberturas vs. Q3.",
                "badge": "↓ -4% Q4",
                "badgeBg": "#FADBD8",
                "badgeColor": "#E74C3C",
                "borderColor": "#E67E22",
            },
        ]

    @staticmethod
    def get_market_evolution():
        """Evolução de aberturas por mês, por setor (Explorar Mercado)."""
        return {
            "Todos os setores": {
                "data": [
                    {"mes": "Jul/24", "value": 8200}, {"mes": "Ago/24", "value": 8600},
                    {"mes": "Set/24", "value": 9100}, {"mes": "Out/24", "value": 9400},
                    {"mes": "Nov/24", "value": 9900}, {"mes": "Dez/24", "value": 10300},
                    {"mes": "Jan/25", "value": 10700}, {"mes": "Fev/25", "value": 11200},
                    {"mes": "Mar/25", "value": 11600}, {"mes": "Abr/25", "value": 12000},
                    {"mes": "Mai/25", "value": 12500}, {"mes": "Jun/25", "value": 13100},
                    {"mes": "Jul/25", "value": 12800}, {"mes": "Ago/25", "value": 13300},
                    {"mes": "Set/25", "value": 13900}, {"mes": "Out/25", "value": 14200},
                    {"mes": "Nov/25", "value": 14800}, {"mes": "Dez/25", "value": 15100},
                    {"mes": "Jan/26", "value": 15600}, {"mes": "Fev/26", "value": 16000},
                    {"mes": "Mar/26", "value": 16500}, {"mes": "Abr/26", "value": 16900},
                    {"mes": "Mai/26", "value": 17400}, {"mes": "Jun/26", "value": 18000},
                ],
            },
            "Saúde e Bem-estar": {
                "data": [
                    {"mes": "Jul/24", "value": 2200}, {"mes": "Ago/24", "value": 2350},
                    {"mes": "Set/24", "value": 2500}, {"mes": "Out/24", "value": 2650},
                    {"mes": "Nov/24", "value": 2800}, {"mes": "Dez/24", "value": 2950},
                    {"mes": "Jan/25", "value": 3100}, {"mes": "Fev/25", "value": 3250},
                    {"mes": "Mar/25", "value": 3400}, {"mes": "Abr/25", "value": 3550},
                    {"mes": "Mai/25", "value": 3700}, {"mes": "Jun/25", "value": 3850},
                    {"mes": "Jul/25", "value": 3800}, {"mes": "Ago/25", "value": 4000},
                    {"mes": "Set/25", "value": 4200}, {"mes": "Out/25", "value": 4350},
                    {"mes": "Nov/25", "value": 4500}, {"mes": "Dez/25", "value": 4650},
                    {"mes": "Jan/26", "value": 4800}, {"mes": "Fev/26", "value": 4950},
                    {"mes": "Mar/26", "value": 5100}, {"mes": "Abr/26", "value": 5250},
                    {"mes": "Mai/26", "value": 5400}, {"mes": "Jun/26", "value": 5600},
                ],
            },
            "Tecnologia": {
                "data": [
                    {"mes": "Jul/24", "value": 2000}, {"mes": "Ago/24", "value": 2100},
                    {"mes": "Set/24", "value": 2200}, {"mes": "Out/24", "value": 2300},
                    {"mes": "Nov/24", "value": 2400}, {"mes": "Dez/24", "value": 2500},
                    {"mes": "Jan/25", "value": 2600}, {"mes": "Fev/25", "value": 2700},
                    {"mes": "Mar/25", "value": 2800}, {"mes": "Abr/25", "value": 2900},
                    {"mes": "Mai/25", "value": 3000}, {"mes": "Jun/25", "value": 3100},
                    {"mes": "Jul/25", "value": 3050}, {"mes": "Ago/25", "value": 3150},
                    {"mes": "Set/25", "value": 3250}, {"mes": "Out/25", "value": 3350},
                    {"mes": "Nov/25", "value": 3450}, {"mes": "Dez/25", "value": 3550},
                    {"mes": "Jan/26", "value": 3650}, {"mes": "Fev/26", "value": 3750},
                    {"mes": "Mar/26", "value": 3850}, {"mes": "Abr/26", "value": 3950},
                    {"mes": "Mai/26", "value": 4050}, {"mes": "Jun/26", "value": 4200},
                ],
            },
            "Educação": {
                "data": [
                    {"mes": "Jul/24", "value": 1400}, {"mes": "Ago/24", "value": 1460},
                    {"mes": "Set/24", "value": 1520}, {"mes": "Out/24", "value": 1580},
                    {"mes": "Nov/24", "value": 1640}, {"mes": "Dez/24", "value": 1700},
                    {"mes": "Jan/25", "value": 1760}, {"mes": "Fev/25", "value": 1820},
                    {"mes": "Mar/25", "value": 1880}, {"mes": "Abr/25", "value": 1940},
                    {"mes": "Mai/25", "value": 2000}, {"mes": "Jun/25", "value": 2080},
                    {"mes": "Jul/25", "value": 2040}, {"mes": "Ago/25", "value": 2120},
                    {"mes": "Set/25", "value": 2200}, {"mes": "Out/25", "value": 2280},
                    {"mes": "Nov/25", "value": 2360}, {"mes": "Dez/25", "value": 2440},
                    {"mes": "Jan/26", "value": 2520}, {"mes": "Fev/26", "value": 2600},
                    {"mes": "Mar/26", "value": 2680}, {"mes": "Abr/26", "value": 2760},
                    {"mes": "Mai/26", "value": 2840}, {"mes": "Jun/26", "value": 2950},
                ],
            },
            "Varejo": {
                "data": [
                    {"mes": "Jul/24", "value": 1650}, {"mes": "Ago/24", "value": 1620},
                    {"mes": "Set/24", "value": 1600}, {"mes": "Out/24", "value": 1580},
                    {"mes": "Nov/24", "value": 1560}, {"mes": "Dez/24", "value": 1530},
                    {"mes": "Jan/25", "value": 1500}, {"mes": "Fev/25", "value": 1470},
                    {"mes": "Mar/25", "value": 1450}, {"mes": "Abr/25", "value": 1420},
                    {"mes": "Mai/25", "value": 1390}, {"mes": "Jun/25", "value": 1360},
                    {"mes": "Jul/25", "value": 1380}, {"mes": "Ago/25", "value": 1350},
                    {"mes": "Set/25", "value": 1320}, {"mes": "Out/25", "value": 1290},
                    {"mes": "Nov/25", "value": 1260}, {"mes": "Dez/25", "value": 1230},
                    {"mes": "Jan/26", "value": 1200}, {"mes": "Fev/26", "value": 1170},
                    {"mes": "Mar/26", "value": 1140}, {"mes": "Abr/26", "value": 1110},
                    {"mes": "Mai/26", "value": 1080}, {"mes": "Jun/26", "value": 1040},
                ],
            },
        }

    @staticmethod
    def get_market_cidades():
        """Top 5 cidades por volume de abertura (Explorar Mercado)."""
        return [
            {"name": "São Paulo", "value": 9840, "color": "#2E6DA4"},
            {"name": "Guarulhos", "value": 3210, "color": "#4A9FD4"},
            {"name": "Campinas", "value": 2780, "color": "#27AE60"},
            {"name": "Santo André", "value": 1940, "color": "#E67E22"},
            {"name": "Osasco", "value": 1620, "color": "#8E44AD"},
        ]

    @staticmethod
    def get_market_export_chart():
        """Dados do gráfico de aberturas por setor usado na tela de Exportar."""
        return [
            {"name": "Saúde", "value": 42.3, "color": "#2E6DA4", "unit": "k", "tooltipLabel": "Aberturas"},
            {"name": "Tech", "value": 38.1, "color": "#4A9FD4", "unit": "k", "tooltipLabel": "Aberturas"},
            {"name": "Educação", "value": 31.7, "color": "#27AE60", "unit": "k", "tooltipLabel": "Aberturas"},
            {"name": "Alimentação", "value": 28.4, "color": "#E67E22", "unit": "k", "tooltipLabel": "Aberturas"},
            {"name": "Financeiro", "value": 24.9, "color": "#8E44AD", "unit": "k", "tooltipLabel": "Aberturas"},
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