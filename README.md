# 🎓 contacomigo.ai — FIAP MBA · TCC

**Plataforma de Inteligência de Mercado e análise contábil/financeira** que combina:

- **RAG (Retrieval-Augmented Generation)** sobre uma base de conhecimento vetorial (ChromaDB), permitindo "conversar" com a documentação de um sistema comercial (estrutura de tabelas, regras de negócio, serviços e rotinas).
- **Analytics de mercado** sobre os **dados abertos de CNPJ da Receita Federal**, organizados em um *star schema* PostgreSQL e expostos em dashboards e telas de exploração.
- **Agentes com ferramentas (MCP)**: o LLM pode consultar o banco PostgreSQL e pesquisar na internet via servidores **Model Context Protocol**.

O frontend é um SPA em React; o backend é uma API FastAPI que orquestra LLM, embeddings, banco vetorial e banco relacional.

O projeto roda em **dois modos**:

| Modo | LLM | Embeddings | Quando usar |
|------|-----|------------|-------------|
| 🏠 **Local** | LM Studio (GPU própria) | LM Studio | Desenvolvimento sem custo de API, dados privados, sem internet |
| ☁️ **Remoto** | Groq | Jina AI | Produção / servidor sem GPU |

> Como ambos os provedores expõem a **API compatível com OpenAI**, alternar entre eles é só mudar variáveis de ambiente — o código é o mesmo.

---

## 🧰 Tecnologias e pacotes usados

### Backend — `fiap_api` (Python / FastAPI)
Definidos em [`fiap_api/requirements.txt`](fiap_api/requirements.txt):

| Pacote | Uso |
|--------|-----|
| `fastapi`, `uvicorn[standard]`, `starlette` | API HTTP e servidor ASGI |
| `pydantic`, `python-dotenv`, `python-multipart` | Modelos/validação, `.env`, upload de arquivos |
| `httpx` | Streaming HTTP (SSE) para o LLM |
| `langchain`, `langchain-openai`, `langchain-community` | Integração com LLMs e embeddings |
| `langchain-chroma` | Pipeline LangChain ↔ ChromaDB |
| `langgraph` | Agente ReAct (orquestração de ferramentas) |
| `langchain-mcp-adapters` | Ferramentas via **Model Context Protocol** |
| `chromadb` | Cliente do banco vetorial |
| `SQLAlchemy` + `psycopg2-binary` | Acesso ao PostgreSQL (dados de mercado) |
| `PyYAML` | Leitura dos YAML da base de conhecimento |
| `requests` | Chamadas à API de embeddings (`/v1/embeddings`) |
| `nest-asyncio`, `uv` | Suporte a event loop aninhado e execução de MCP servers (`uvx`) |

### Frontend — `fiap_interface` (React / TypeScript)
Definidos em [`fiap_interface/package.json`](fiap_interface/package.json):

| Pacote | Uso |
|--------|-----|
| `react` 19, `react-dom`, `typescript` | Base do SPA |
| `@mui/material` 7, `@mui/icons-material`, `@emotion/*` | Componentes e tema (Material UI) |
| `recharts` | Gráficos do Dashboard e de mercado |
| `react-markdown` + `remark-gfm` | Renderização das respostas do chat em Markdown |
| `react-syntax-highlighter` | Destaque de código nas respostas |
| `vega-embed` | Renderização de gráficos **Vega-Lite** gerados pelo assistente |
| `lucide-react` | Ícones complementares |
| `react-scripts` (CRA) | Build/dev server |

### Infraestrutura
- **ChromaDB** (`fiap_chromadb`) — banco vetorial (distância de cosseno).
- **PostgreSQL 15** — base `cnpj_dados_abertos` (star schema).
- **pgAdmin 4** — administração do PostgreSQL.
- **`acuvity/mcp-server-postgres`** — servidor MCP que expõe o PostgreSQL como ferramenta do agente.
- **Docker Compose** — orquestração de toda a stack.
- **LM Studio** (`fiap_lmstudio`, opcional) — LLM/embeddings locais com aceleração por GPU NVIDIA.

---

## 📐 Arquitetura

```mermaid
flowchart TB
    UI["<b>fiap-interface</b><br/>React 19 · MUI 7 · Recharts<br/><code>porta:3000</code>"]

    subgraph api["⚙️ fiap-api &nbsp;·&nbsp; porta:8001"]
        direction LR
        AGENT["<b>GenAIFactory + Agente</b><br/>FastAPI · LangChain + LangGraph (ReAct)"]
        CHCLI["<b>ChromaDB Factory</b><br/>+ EmbeddingsFactory"]
        SQLF["<b>SQLFactory</b><br/>SQLAlchemy · DataBase"]
    end

    subgraph tools["🛠️ Ferramentas do agente (MCP)"]
        DDG["<b>duckduckgo</b><br/>MCP / stdio"]
        MCP["<b>mcp-postgres</b><br/>MCP server<br/><code>porta:8000</code>"]
    end

    subgraph ia["🤖 Provedores de IA — API OpenAI-compatível"]
        LLM["<b>LLM provider</b><br/>Groq · LM Studio<br/>OpenAI · Azure"]
        EMB["<b>Embeddings</b><br/>LM Studio<br/>OpenAI · Azure"]
    end

    subgraph dados["🗄️ Dados / Infra"]
        CH["<b>chromadb</b><br/>vetorial · cosseno<br/><code>porta:8210</code>"]
        PG["<b>postgres</b><br/>PostgreSQL 15<br/><code>porta:5432</code>"]
        PGA["<b>pgadmin</b><br/>admin Postgres<br/><code>porta:4000</code>"]
    end

    UI <==>|"HTTP / SSE"| api

    AGENT -->|"GenAI · prompt/tokens"| LLM
    AGENT -->|"tool · web"| DDG
    AGENT -->|"tool · SQL"| MCP
    AGENT -->|"tool · RAG"| CH
    CHCLI -.->|"index / query"| CH
    CHCLI -.->|"/v1/embeddings"| EMB
    SQLF -.->|"SQLFactory"| PG
    MCP -->|"SQL"| PG
    PGA -->|"admin"| PG

    classDef bk fill:#f4f0ff,stroke:#5e35b1,color:#311b92;
    classDef tl fill:#fff0f4,stroke:#e91e8c,color:#a01060;
    classDef aip fill:#fff3e0,stroke:#fb8c00,color:#e65100;
    classDef dt fill:#e8f7ec,stroke:#43a047,color:#1b5e20;
    class UI bk;
    class AGENT,CHCLI,SQLF bk;
    class MCP,DDG tl;
    class LLM,EMB aip;
    class CH,PG,PGA dt;
```

> 📐 Versões do diagrama para apresentação/slides:
> - **[`docs/arquitetura.html`](docs/arquitetura.html)** — versão FIAP (tema escuro, animada, roteamento ortogonal sem cruzamentos). Abra no navegador.
> - **[`docs/arquitetura.svg`](docs/arquitetura.svg)** — versão vetorial estática (com legenda).
>
> O **agente LangGraph (ReAct)** orquestra três ferramentas — `mcp-postgres` (SQL), `duckduckgo` (web) e `search_knowledge_base`/RAG sobre o ChromaDB — enquanto as telas de **mercado** consultam o PostgreSQL **diretamente** via `SQLFactory` (linha tracejada). Os **embeddings** são gerados pela `ChromaDB Factory` (`POST /v1/embeddings`) tanto na indexação quanto na busca.

### Componentes

| Diretório / Serviço | Tecnologia | Porta | Função |
|---------------------|-----------|-------|--------|
| [`fiap_api/`](fiap_api/) | FastAPI + LangChain + LangGraph | `8001` | API de orquestração: chat com streaming, gestão do banco vetorial e endpoints de dados de mercado |
| [`fiap_interface/`](fiap_interface/) | React 19 + TypeScript + MUI 7 + Recharts | `3000` | SPA com Dashboard, chat, exploração de mercado, exportação e manutenção |
| [`fiap_chromadb/`](fiap_chromadb/) | ChromaDB | `8210` | Banco de dados vetorial (base de conhecimento, distância de cosseno) |
| `postgres` | PostgreSQL 15 | `5432` | Base `cnpj_dados_abertos` (star schema dos dados da Receita) |
| `pgadmin` | pgAdmin 4 | `4000` | Administração do PostgreSQL |
| `mcp-postgres` | `acuvity/mcp-server-postgres` | `8000` | Servidor MCP que expõe o PostgreSQL como ferramenta para o agente |
| [`fiap_lmstudio/`](fiap_lmstudio/) | LM Studio (opcional) | `1234` / `5900` | Provedor local de LLM e embeddings (alternativa a Groq/Jina) |

> No [`docker-compose.yml`](docker-compose.yml) de produção, `chromadb`, `postgres` e `mcp-postgres` ficam acessíveis **apenas na rede interna** (`fiap-network`), sem portas publicadas no host. Para desenvolvimento, o [`docker-compose-local.yml`](docker-compose-local.yml) publica essas portas.

---

## ⚙️ Como configurar: executando o projeto local

Esta seção descreve o setup comum (banco vetorial, banco relacional e dependências). Em seguida, escolha um dos modos: [LM Studio (local)](#-execução-local-usando-o-lmstudio) ou [Jina + Groq (remoto)](#️-execução-remota-usando-o-jina-e-o-groq).

### 1. Pré-requisitos

- **Docker Desktop** (para ChromaDB, PostgreSQL, pgAdmin e o servidor MCP)
- **Python 3.11+** (para rodar a API fora do container)
- **Node.js 18+** (para rodar a interface)

### 2. Subir a infraestrutura

O [`docker-compose-local.yml`](docker-compose-local.yml) sobe apenas a infraestrutura — ChromaDB, PostgreSQL, pgAdmin e MCP — com as portas publicadas no host:

```powershell
docker-compose -f docker-compose-local.yml up -d
```

### 3. Configurar e iniciar a API

```powershell
cd fiap_api
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
# crie o arquivo .env (ver os modos abaixo)
python main.py
# API em http://localhost:8001  ·  docs em /docs
```

A conexão com o ChromaDB é feita **em background** no startup, com retry e backoff (configurável via `CHROMADB_CONNECT_TIMEOUT` / `CHROMADB_CONNECT_MAX_RETRIES`) — a API sobe imediatamente e passa a enxergar o ChromaDB assim que ele estiver disponível.

### 4. Iniciar a interface

```powershell
cd fiap_interface
npm install
npm start
# Interface em http://localhost:3000
```

> A URL da API é embutida no bundle em *build time* via `REACT_APP_API_URL` (Create React App). Em desenvolvimento, o `proxy` do [`package.json`](fiap_interface/package.json) encaminha as chamadas ao backend.

### Variáveis de ambiente comuns

Independente do modo, estas variáveis configuram a infraestrutura (arquivo `fiap_api/.env`):

```env
# ChromaDB
CHROMADB_HOST=localhost
CHROMADB_PORT=8210
CHROMADB_DEFAULT_RESULTS=100

# PostgreSQL (dados de mercado)
DATABASE_URL=postgresql://postgres:postgres@localhost:5432/cnpj_dados_abertos

# MCP (consulta SQL via agente) — opcional
MCP_POSTGRES_URL=http://localhost:8000/mcp

ENVIRONMENT=local
```

As variáveis de **GenAI** e **Embeddings** mudam conforme o modo escolhido — veja abaixo. O **EnvFactory** ([`factories/env_factory.py`](fiap_api/factories/env_factory.py)) valida todas elas no boot e dá mensagens de erro claras se faltar algo.

---

## 🏠 Execução local: usando o LM Studio

Use este modo para rodar o LLM e os embeddings **na sua própria máquina** (com GPU NVIDIA), sem custo de API e sem enviar dados para fora.

### 1. Instalar e carregar os modelos no LM Studio

1. Baixe o **[LM Studio](https://lmstudio.ai)** (instalador Windows) e instale.
2. Na aba **Search** (lupa), baixe os dois modelos usados pelo projeto:
   - **Chat/LLM**: `gpt-oss-20b`
   - **Embeddings**: `text-embedding-nomic-embed-text-v1.5`
3. Em **Developer → Start Server**, inicie o servidor HTTP na porta **`1234`** (API compatível com OpenAI). Deixe os dois modelos carregados (**Load**).

> 💡 Alternativamente, é possível rodar o LM Studio em container com GPU — veja [`fiap_lmstudio/README.md`](fiap_lmstudio/README.md) (requer WSL2 + NVIDIA Container Toolkit; expõe a API em `1234` e VNC em `5900`).

### 2. Configurar o `.env` da API

```env
# ===== GenAI (LLM local) =====
GENAI_PROVIDER=lmstudio
GENAI_ENDPOINT=http://localhost:1234
GENAI_MODEL=gpt-oss-20b
GENAI_API_KEY=lm-studio
GENAI_API_VERSION=
GENAI_TEMPERATURE=0.7
GENAI_MAX_TOKENS=8192
GENAI_TOP_P=0.95

# ===== Embeddings (local) =====
EMBEDDINGS_PROVIDER=lmstudio
EMBEDDINGS_ENDPOINT=http://localhost:1234
EMBEDDINGS_MODEL=text-embedding-nomic-embed-text-v1.5
EMBEDDINGS_API_KEY=
```

### 3. Subir tudo de uma vez (Windows)

Com o LM Studio já rodando na porta 1234, o script [`debug.bat`](debug.bat) verifica o LM Studio, sobe a infraestrutura (`docker-compose-local.yml`) e inicia API e interface em janelas separadas:

```powershell
.\debug.bat
```

---

## ☁️ Execução remota: usando o Jina e o Groq

Use este modo em **servidores sem GPU** (ou produção): o LLM roda na **Groq** e os embeddings na **Jina AI**. É a configuração padrão do [`docker-compose.yml`](docker-compose.yml).

### 1. Obter as chaves de API

- **Groq** (LLM): crie a chave em https://console.groq.com — formato `gsk_...`
- **Jina AI** (embeddings): crie a chave em https://jina.ai/embeddings — formato `jina_...`

### 2. Exportar as chaves para o Docker

O compose lê `GENAI_API_KEY` e `EMBEDDINGS_API_KEY` do ambiente do host:

```powershell
$env:GENAI_API_KEY = "gsk_sua_chave_groq"
$env:EMBEDDINGS_API_KEY = "jina_sua_chave_jina"
```

### 3. Subir a stack completa

```powershell
# Windows — inicia o Docker Desktop, limpa, builda e sobe tudo
.\run.bat
```

Ou manualmente:

```powershell
docker-compose up -d --build
```

Acesse:
- **Interface**: http://localhost:3000
- **API (docs)**: http://localhost:8001/docs
- **pgAdmin**: http://localhost:4000

### Configuração remota (já presente no `docker-compose.yml`)

```env
# ===== GenAI (Groq, via API compatível com OpenAI) =====
GENAI_PROVIDER=lmstudio
GENAI_ENDPOINT=https://api.groq.com/openai
GENAI_MODEL=qwen/qwen3-32b
GENAI_API_KEY=${GENAI_API_KEY}
GENAI_TEMPERATURE=0.7
GENAI_MAX_TOKENS=8192
GENAI_TOP_P=0.95

# ===== Embeddings (Jina AI) =====
EMBEDDINGS_PROVIDER=jina
EMBEDDINGS_ENDPOINT=https://api.jina.ai
EMBEDDINGS_MODEL=jina-embeddings-v5-text-small
EMBEDDINGS_API_KEY=${EMBEDDINGS_API_KEY}
```

> O `GENAI_PROVIDER` continua `lmstudio` porque a Groq usa o **mesmo protocolo** (API OpenAI-compatível) — basta apontar o `GENAI_ENDPOINT` para a Groq.

---

## 📋 Referência das variáveis de ambiente

### GenAI (LLM) — providers: `lmstudio`, `openai`, `azure`

| Variável | Descrição |
|----------|-----------|
| `GENAI_PROVIDER` | `lmstudio` (inclui Groq/LM Studio) \| `openai` \| `azure` |
| `GENAI_ENDPOINT` | URL base do serviço |
| `GENAI_MODEL` | Nome do modelo |
| `GENAI_API_KEY` | Chave de API (opcional para LM Studio local) |
| `GENAI_API_VERSION` | Obrigatória apenas para Azure |
| `GENAI_TEMPERATURE` / `GENAI_MAX_TOKENS` / `GENAI_TOP_P` | Parâmetros de geração |

### Embeddings — providers: `lmstudio`, `jina`, `openai`, `azure`

| Variável | Descrição |
|----------|-----------|
| `EMBEDDINGS_PROVIDER` | `lmstudio` \| `jina` \| `openai` \| `azure` |
| `EMBEDDINGS_ENDPOINT` | URL base (ex.: `https://api.jina.ai`) |
| `EMBEDDINGS_MODEL` | Modelo de embeddings |
| `EMBEDDINGS_API_KEY` | Chave de API |
| `EMBEDDINGS_TASK` / `EMBEDDINGS_DIMENSIONS` | Opcionais (Jina) |

A indexação e a busca no ChromaDB usam a `RemoteEmbeddingFunction` ([`factories/chromadb_factory.py`](fiap_api/factories/chromadb_factory.py)), que chama `POST /v1/embeddings` e **normaliza os vetores** (norma unitária — torna L2 e cosseno equivalentes).

### Infra

| Variável | Descrição |
|----------|-----------|
| `CHROMADB_HOST` / `CHROMADB_PORT` | Host/porta do servidor ChromaDB |
| `CHROMADB_DEFAULT_RESULTS` | Nº de resultados por busca (padrão 100; compose remoto usa 150) |
| `CHROMADB_CONNECT_TIMEOUT` / `CHROMADB_CONNECT_MAX_RETRIES` | Controle do retry de conexão no startup (`0` = tenta indefinidamente) |
| `DATABASE_URL` | String de conexão PostgreSQL |
| `MCP_POSTGRES_URL` | Endpoint MCP do PostgreSQL |
| `DATA_DIR` | Pasta base para YAML enviados pela interface (senão usa pasta temporária do SO) |

---

## 🖥️ Telas da interface

Definidas em [`fiap_interface/src/App.tsx`](fiap_interface/src/App.tsx):

- **Dashboard** — métricas gerais do mercado (CNPJs ativos, aberturas, CNAEs, baixas), aberturas por setor, distribuição por porte e *insights*.
- **Perguntar aos Dados** — chats especializados sobre a base de conhecimento (RAG). Suporta renderização de Markdown e gráficos **Vega-Lite** gerados pelo assistente.
- **Explorar Mercado** — evolução de aberturas por mês e top cidades, com filtros por UF, porte, setor e período.
- **Exportar** — exportação de gráficos/dados (utilitário em [`src/utils/chartExport.ts`](fiap_interface/src/utils/chartExport.ts)).
- **Manutenção** — gestão de coleções do ChromaDB, upload/teste de documentos ([`VectorDBTest`](fiap_interface/src/components/VectorDBTest.tsx)) e estatísticas.
- **Chat geral flutuante** — assistente em modo "Pesquisa" (usa ferramentas MCP de busca na internet), disponível em todas as telas.

---

## 🤖 Assistentes e agentes

Os *system prompts* estão em [`fiap_api/system_prompts.json`](fiap_api/system_prompts.json). A persona principal é **ContaComigo.IA**, focada em análise financeira/contábil, que responde **somente** com base no contexto recuperado e pode gerar gráficos Vega-Lite sob demanda.

Quando o agente está ativo (LangGraph ReAct), o LLM dispõe de ferramentas:

- **`mcp-postgres`** — consultas SQL ao banco de CNPJs via MCP.
- **`duckduckgo`** — busca na internet (modo Pesquisa).
- **`search_knowledge_base`** — busca semântica no ChromaDB, adicionada dinamicamente quando uma coleção é selecionada.

### Principais rotas da API

| Método | Rota | Descrição |
|--------|------|-----------|
| `POST` | `/api/chat/help/stream` · `/aluno/stream` · `/sql/stream` · `/general/stream` | Chat com streaming (SSE) por assistente |
| `POST` | `/api/vectordb/upload` · `/upload-batch` | Upload de YAML com detecção automática de tipo e indexação |
| `GET`  | `/vectordb/collections` · `/vectordb/stats` | Coleções e estatísticas |
| `POST` | `/vectordb/create-collection` · `/vectordb/query` · `/vectordb/add-item` | Criar coleção, buscar e inserir |
| `POST` | `/vectordb/collection/{name}/reindex` | Recria a coleção com métrica de cosseno e reindexa os YAML do disco |
| `GET`  | `/api/market/metrics` · `/setores` · `/porte` · `/insights` · `/filtros` · `/evolution` · `/cidades` · `/export-chart` | Dados de mercado (CNPJ) |
| `GET`  | `/health` · `/health/chromadb` · `/health/lmstudio` | Health checks |

---

## 🧠 Base de conhecimento (ChromaDB)

A base é alimentada por arquivos **YAML** organizados em quatro categorias (exemplos em [`tests/chromadb/data/`](tests/chromadb/data/)):

- **`base_dados`** — estrutura de tabelas e colunas (documentos `table` / `field`).
- **`regras_negocio`** — regras de negócio (`business_rule`).
- **`servicos`** — rotinas de sistema/serviços (`service` / `rotina_sistema`).
- **`rotinas_usuario`** — procedimentos de usuário (`rotina_usuario`).

No upload, o tipo é **detectado automaticamente** pelo conteúdo/nome ([`detect_file_type`](fiap_api/main.py)); se houver dúvida, um validador via LLM ([`FileValidator`](fiap_api/factories/validation_factory.py)) auxilia a classificação. Os documentos são otimizados para reduzir tokens ([`DocumentOptimizer`](fiap_api/factories/document_optimizer.py)) e indexados com **distância de cosseno**.

---

## 📊 Dados de mercado (PostgreSQL)

Os números das telas de mercado derivam de um **star schema** dos dados abertos de CNPJ da Receita Federal, montado em [`tests/receita/`](tests/receita/):

- **Fato**: `fat_empresas_mensal` (aberturas/baixas mensais).
- **Dimensões**: `dim_empresas_mensal`, `dim_cnaes`, `dim_cnaes_segmento`, `dim_municipios`, `dim_competencias`.

As consultas ficam encapsuladas na [`SQLFactory`](fiap_api/factories/sql_factory.py) (SQLAlchemy), que também aplica os filtros de UF/porte/setor/período das telas de exploração. Códigos observados na base: `situacao` 2=ATIVA / 8=BAIXADA; `porte` 1=MEI/Micro, 2=Pequeno, 3=Grande.

---

## 🧪 Testes

```powershell
# Testes do ChromaDB (base de conhecimento)
python tests\chromadb\test_chromadb_database.py
python tests\chromadb\test_user_routines.py

# Testes da SQLFactory / estrutura do banco
cd tests\sql
pip install -r requirements.txt
python -m pytest -v        # ou: python run_tests.py

# Teste de variáveis de ambiente
python tests\test_env_factory.py
```

Detalhes dos testes de banco em [`tests/sql/README.md`](tests/sql/README.md).

---

## 🗂️ Estrutura do repositório

```
FIAPTCC/
├─ fiap_api/                 # API FastAPI (orquestração)
│  ├─ main.py                # rotas: chat (SSE), vectordb, market
│  ├─ system_prompts.json    # personas dos assistentes
│  └─ factories/             # GenAI, Embeddings, ChromaDB, SQL, Env, Validation, DocumentOptimizer
├─ fiap_interface/           # Frontend React (SPA)
│  └─ src/
│     ├─ App.tsx             # layout + navegação
│     ├─ components/         # Dashboard, ChatTabs, ExplorarMercado, Manutencao, ...
│     ├─ contexts/           # CollectionContext, NotificationContext
│     └─ services/marketApi.ts
├─ fiap_chromadb/            # Servidor ChromaDB (Docker / local)
├─ fiap_lmstudio/            # Setup opcional do LM Studio (GPU + Docker)
├─ tests/
│  ├─ chromadb/data/         # YAML da base de conhecimento (4 categorias)
│  ├─ receita/               # schema + dados do star schema CNPJ
│  └─ sql/                   # testes da SQLFactory
├─ docker-compose.yml        # stack completa (remoto: Groq + Jina, rede interna)
├─ docker-compose-local.yml  # só a infraestrutura (portas publicadas)
├─ run.bat                   # sobe a stack completa via Docker (modo remoto)
└─ debug.bat                 # sobe infra + API + UI locais (modo LM Studio)
```

---

## 🛠️ Troubleshooting

- **A API sobe mas não acha o ChromaDB** — normal no boot: a conexão é assíncrona com retry. Verifique se o container `chromadb` está de pé (`docker ps`) e cheque `GET /health/chromadb`.
- **`LMStudio not found on port 1234`** (no `debug.bat`) — inicie o servidor HTTP do LM Studio (Developer → Start Server) **antes** de rodar o script.
- **Erro de variável de ambiente obrigatória** — o `EnvFactory` aborta o boot indicando exatamente qual `GENAI_*`/`EMBEDDINGS_*` falta no `.env`.
- **Busca vetorial retorna vazio** — confirme que indexou documentos na coleção e que os embeddings estão sendo gerados (sem chave válida da Jina, ou com o LM Studio sem o modelo de embeddings carregado, a indexação falha de forma visível). Use `/vectordb/collection/{name}/reindex` após trocar a métrica/modelo.
- **GPU do LM Studio em Docker** — veja a seção de pré-requisitos (WSL2 + NVIDIA Container Toolkit) em [`fiap_lmstudio/README.md`](fiap_lmstudio/README.md).

---

> Projeto desenvolvido como **TCC do MBA FIAP**. Fonte dos dados de mercado: **Receita Federal do Brasil (RFB)** — dados abertos de CNPJ.
