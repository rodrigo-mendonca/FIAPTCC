# 🎓 FIAP MBA - TCC

Sistema para demonstração e testes de busca semântica usando **ChromaDB** e **LMStudio**, com API Python e frontend React.

**O que este repositório contém (resumo):**
- ChromaDB para armazenamento de embeddings (porta 8200)
- LMStudio para geração de embeddings e chat (ports 1234 / 8080)
- API Python (FastAPI) para orquestração (porta 8000)
- Frontend React para testes e demonstração (porta 3000)

**Observação importante sobre o LMStudio:**
- O LMStudio pode ser executado de duas formas: diretamente pelo aplicativo Windows ou como um container Docker. A seção abaixo mostra os passos para ambas as opções e como carregar os dois modelos usados pelo projeto.

## LMStudio — execução e modelos

### 1) Instalando e executando o LMStudio no Windows

1. **Baixar o instalador:**
   - Acesse [LM Studio — Official Website](https://lmstudio.ai)
   - Faça download do instalador para **Windows** (.exe)
   - Alternativamente, baixe direto do [GitHub Releases](https://github.com/lmstudio-ai/lmstudio-docs/releases)

2. **Instalar:**
   - Execute o arquivo `.exe` e siga o assistente de instalação padrão do Windows
   - Escolha o diretório de instalação (padrão: `C:\Users\<seu-usuario>\AppData\Local\LM Studio`)
   - Aguarde a conclusão da instalação

3. **Executar:**
   - Abra o **LM Studio** a partir do menu Iniciar ou do atalho da área de trabalho
   - Interface gráfica abre na porta `8080` (http://localhost:8080)

4. **Configurar servidor HTTP:**
   - Na interface gráfica do LM Studio, vá para **Developer > Start Server**
   - Configure a porta `1234` para a API HTTP (embeddings e chat)
   - O servidor ficará disponível em `http://localhost:1234`

### 2) Executando o LMStudio via Docker

- O repositório inclui scripts e um `docker-compose` para orquestrar todos os serviços. Para iniciar via Docker (Windows PowerShell):

```powershell
# Inicia todos os serviços (inclui LMStudio quando configurado no compose)
.\start-docker.bat
# Ou diretamente:
# docker-compose up -d
```

- Exemplo genérico de execução do LMStudio em container (substitua a imagem pelo upstream/versão que desejar):

```powershell
docker run -d --name lmstudio \
    -p 8080:8080 -p 1234:1234 \
    -v $env:USERPROFILE\.lmstudio\models:/root/.cache/lm-studio/models \
    ghcr.io/nomic-ai/lm-studio:latest
```

### 3) Como baixar e carregar modelos

**Modelos necessários para este projeto:**
- Modelo de embeddings: `text-embedding-nomic-embed-text-v1.5` (gera embeddings dos documentos no ChromaDB)
- Modelo de chat/LLM: `gpt-oss-20b` (fornece respostas no chat)

#### Método 1: Via Interface Gráfica do LM Studio (Recomendado para Windows)

1. Abra o **LM Studio** (já deve estar em execução)
2. Clique na aba **Search** (lupa) no menu lateral
3. Na barra de busca, digite o nome do primeiro modelo: `nomic-embed-text-v1.5`
4. Localize o resultado **text-embedding-nomic-embed-text-v1.5** (Nomic Embeddings)
5. Clique no botão **Download** (seta para baixo)
6. Aguarde o download completar (tamanho ~274 MB)
7. Repita os passos 3-6 para o modelo de chat: busque por `gpt-oss-20b`
8. Clique em **Load** para carregar o modelo na memória (aparecerá em **Open**/**Loaded Models**)

#### Método 2: Via Terminal/PowerShell

Se os modelos já estiverem baixados, carregue-os manualmente:

```powershell
# A partir do PowerShell (com LM Studio em execução):

# Carregar modelo de embeddings
curl -X POST "http://localhost:1234/v1/models/load" `
  -H "Content-Type: application/json" `
  -d '{"model": "text-embedding-nomic-embed-text-v1.5"}'

# Carregar modelo de chat
curl -X POST "http://localhost:1234/v1/models/load" `
  -H "Content-Type: application/json" `
  -d '{"model": "gpt-oss-20b"}'

# Listar modelos carregados
curl "http://localhost:1234/v1/models"
```

#### Diretório de Armazenamento

Os modelos são armazenados em:
- **Windows (instalação nativa):** `%USERPROFILE%\.lmstudio\models` (ex: `C:\Users\seu-usuario\.lmstudio\models`)
- **Docker:** Mapeado para `%USERPROFILE%\.lmstudio\models` no host

### 4) Portas principais

- `8080`: UI web do LMStudio (chat)
- `1234`: API interna/HTTP do LMStudio (endpoints de embeddings e chat)

## Configuração de Providers (GenAI e Embeddings)

O projeto suporta múltiplos provedores de IA generativa e embeddings: **LMStudio**, **OpenAI** e **Azure OpenAI**. A configuração é feita através do arquivo `.env` único, alterando o `GENAI_PROVIDER` e as variáveis correspondentes.

### Estrutura de Configuração

```
.env                 # Configuração completa (CHROMADB, CORS, ENVIRONMENT, GenAI, Embeddings)
```

### Variáveis Unificadas

Cada provider usa o mesmo conjunto de variáveis:

- `GENAI_PROVIDER` — define o provedor (lmstudio, openai, azure)
- `GENAI_ENDPOINT` — URL/endpoint do serviço (ex: `http://192.168.50.30:1234` para LMStudio)
- `GENAI_MODEL` — nome do modelo (ex: `gpt-3.5-turbo`)
- `GENAI_API_KEY` — chave de API (vazio para LMStudio)
- `GENAI_API_VERSION` — versão da API (vazio para LMStudio e OpenAI, obrigatório para Azure)

Igual para embeddings: `EMBEDDINGS_PROVIDER`, `EMBEDDINGS_ENDPOINT`, `EMBEDDINGS_MODEL`, `EMBEDDINGS_API_KEY`, `EMBEDDINGS_API_VERSION`

### Como Trocar de Provider

#### 1. Localmente (sem Docker)

Edite o arquivo `.env` e altere o `GENAI_PROVIDER` e as variáveis correspondentes:

```bash
# Use LMStudio (padrão)
# GENAI_PROVIDER=lmstudio
# GENAI_ENDPOINT=http://192.168.50.30:1234/v1
# ...

# Use OpenAI
# GENAI_PROVIDER=openai
# GENAI_API_KEY=sk-...
# ...

# Use Azure
# GENAI_PROVIDER=azure
# GENAI_ENDPOINT=https://seu-recurso.openai.azure.com/
# GENAI_API_VERSION=2024-02-15-preview
# ...
```

Depois execute:

```bash
python fiap_api/main.py
```

#### 2. Com Docker Compose

O `docker-compose.yml` já carrega o `.env` único. Apenas edite o arquivo `.env` como acima e inicie:

```powershell
docker-compose up -d
```

#### 3. Via Command Line (Docker)

```bash
# Edite o .env conforme acima
docker-compose --env-file .env up -d
```

### Exemplos de Configuração

Edite o arquivo `.env` com as configurações desejadas. Aqui estão exemplos para cada provider:

**LMStudio** (no `.env`)
```env
GENAI_PROVIDER=lmstudio
GENAI_ENDPOINT=http://192.168.50.30:1234/v1
GENAI_MODEL=gpt-3.5-turbo

EMBEDDINGS_PROVIDER=lmstudio
EMBEDDINGS_ENDPOINT=http://192.168.50.30:1234/v1
EMBEDDINGS_MODEL=text-embedding-nomic-embed-text-v1.5
```

**OpenAI** (no `.env`)
```env
GENAI_PROVIDER=openai
GENAI_MODEL=gpt-3.5-turbo
GENAI_API_KEY=sk-...

EMBEDDINGS_PROVIDER=openai
EMBEDDINGS_MODEL=text-embedding-3-small
EMBEDDINGS_API_KEY=sk-...
```

**Azure** (no `.env`)
```env
GENAI_PROVIDER=azure
GENAI_ENDPOINT=https://seu-recurso.openai.azure.com/
GENAI_MODEL=gpt-35-turbo
GENAI_API_KEY=seu-api-key
GENAI_API_VERSION=2024-02-15-preview

EMBEDDINGS_PROVIDER=azure
EMBEDDINGS_ENDPOINT=https://seu-recurso.openai.azure.com/
EMBEDDINGS_MODEL=text-embedding
EMBEDDINGS_API_KEY=seu-api-key
EMBEDDINGS_API_VERSION=2024-02-15-preview
```

## Pré-requisitos

## Início rápido (resumido)

- Usando scripts do repositório (Windows):
    - `start-all.bat` — inicia todos os componentes localmente (quando apropriado)
    - `start-docker.bat` — inicia a stack via Docker

- Ou manualmente:

```powershell
# Com Docker Compose
docker-compose up -d

# Ou, se preferir usar apenas os scripts de exemplo
start-all.bat
```

## Componentes resumidos

- `fiap_api/` — API Python (FastAPI) — `http://localhost:8000`
- `fiap_interface/` — Frontend React — `http://localhost:3000`
- `fiap_chromadb/` — integração e cliente ChromaDB — `http://localhost:8200`

## Modelos usados pelo projeto

- `text-embedding-nomic-embed-text-v1.5` — modelo de embeddings (ChromaDB)
- `gpt-oss-20b` — modelo de conversa/LLM usado no LMStudio

Se quiser que eu inclua comandos exatos para baixar as releases do LM Studio (link direto para a AppImage) ou um exemplo `docker-compose` específico para o `lmstudio` com image tag, eu posso adicionar — diga se prefere link direto para GitHub Releases ou manter apenas a referência local ao arquivo em `fiap_lmstudio/AppImage/`.

## Testes rápidos

- Testar ChromaDB (exemplo):
```powershell
python tests\chromadb\test_chromadb_database.py
```

---
