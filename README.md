# 🎓 FIAP MBA - TCC

Sistema para demonstração e testes de busca semântica usando **ChromaDB** e **LMStudio**, com API Python e frontend React.

**O que este repositório contém (resumo):**
- ChromaDB para armazenamento de embeddings (porta 8200)
- LMStudio para geração de embeddings e chat (ports 1234 / 8080)
- API Python (FastAPI) para orquestração (porta 8000)
- Frontend React para testes e demonstração (porta 3000)

**Observação importante sobre o LMStudio:**
- O LMStudio pode ser executado de duas formas: diretamente pelo aplicativo (AppImage) ou como um container Docker. A seção abaixo mostra os passos para ambas as opções e como carregar os dois modelos usados pelo projeto.

## LMStudio — execução e modelos

1) Executando o LMStudio localmente (AppImage — Linux/WSL)

- Se você já tem a AppImage incluída no repositório, copie-a ou use a que está em `fiap_lmstudio/AppImage/LM-Studio-0.3.23-3-x64.AppImage`.
- Torne executável e rode:

```bash
chmod +x LM-Studio-0.3.23-3-x64.AppImage
./LM-Studio-0.3.23-3-x64.AppImage
```

- Recomendação: execute em Linux nativo ou WSL2 se estiver no Windows. O AppImage abre a interface gráfica (porta 8080 para o chat). Para disponibilizar a API interna (embeddings/chat HTTP) verifique o serviço `lms server` (porta 1234 no container/configuração usada aqui).

2) Executando o LMStudio via Docker

- O repositório inclui scripts e um `docker-compose` para orquestrar todos os serviços. Para iniciar via Docker (Windows PowerShell):

```powershell
# Inicia todos os serviços (inclui LMStudio quando configurado no compose)
.
start-docker.bat
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

3) Onde colocar / como carregar modelos

- Diretório de modelos do LMStudio (mapeado no container): `~/.lmstudio/models` (no Windows, use `%USERPROFILE%\.lmstudio\models` para mapear). Copie os arquivos dos modelos para esse diretório antes de iniciar o LMStudio ou use a UI/CLI do LMStudio para instalar.

- Dois modelos necessários para este projeto:
    - Modelo de embeddings: `text-embedding-nomic-embed-text-v1.5` (usado para gerar embeddings dos documentos indexados no ChromaDB)
    - Modelo de chat/LLM: `gpt-oss-20b` (ex.: usado pelo container/`start_services.sh` para prover respostas no chat)

- Para carregar manualmente via CLI do LMStudio (dentro do container ou no host que fornece o `lms`):

```bash
# Carregar modelo de embeddings
lms load text-embedding-nomic-embed-text-v1.5 --yes

# Carregar modelo de chat (exemplo forçando uso de GPU e contexto grande)
lms load gpt-oss-20b --gpu 0.9 --context-length 32768 --yes

# Ver modelos carregados
lms ps

# Iniciar servidor HTTP do LMStudio (se necessário)
lms server start --port 1234 --cors &
```

4) Portas principais

- `8080`: UI web do LMStudio (chat)
- `1234`: API interna/HTTP do LMStudio (endpoints de embeddings e controle via `lms`)

## Pré-requisitos

- Baixe o **LM-Studio-0.3.23-3-x64.AppImage** em [GitHub Releases](https://github.com/lmstudio-ai/lmstudio-docs/releases) ou use a versão fornecida
- Coloque o arquivo na pasta `fiap_lmstudio/AppImage/` do repositório
- O arquivo já está configurado no `.gitignore` (não será versionado)

## Configuração de Providers (GenAI e Embeddings)

O projeto suporta múltiplos provedores de IA generativa e embeddings: **LMStudio**, **OpenAI** e **Azure OpenAI**. A configuração é feita através de arquivos `.env` separados para facilitar a troca entre provedores.

### Estrutura de Configuração

```
.env                 # Configuração geral (CHROMADB, CORS, ENVIRONMENT)
.env.lmstudio       # Configuração do LMStudio
.env.openai         # Configuração do OpenAI
.env.azure          # Configuração do Azure OpenAI
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

Edite o `.env` e descomente o arquivo desejado:

```bash
# Use LMStudio (padrão)
source .env && source .env.lmstudio && python fiap_api/main.py

# Use OpenAI
source .env && source .env.openai && python fiap_api/main.py

# Use Azure
source .env && source .env.azure && python fiap_api/main.py
```

#### 2. Com Docker Compose

Altere o `docker-compose.yml` na seção `fiap-api`:

```yaml
# Para LMStudio (padrão)
env_file:
  - .env
  - .env.lmstudio

# Para OpenAI
env_file:
  - .env
  - .env.openai

# Para Azure
env_file:
  - .env
  - .env.azure
```

Depois inicie:

```powershell
docker-compose up -d
```

#### 3. Via Command Line (Docker)

```bash
# LMStudio
docker-compose --env-file .env --env-file .env.lmstudio up -d

# OpenAI
docker-compose --env-file .env --env-file .env.openai up -d

# Azure
docker-compose --env-file .env --env-file .env.azure up -d
```

### Exemplos de Configuração

**LMStudio** (`.env.lmstudio`)
```env
GENAI_PROVIDER=lmstudio
GENAI_ENDPOINT=http://192.168.50.30:1234
GENAI_MODEL=gpt-3.5-turbo

EMBEDDINGS_PROVIDER=lmstudio
EMBEDDINGS_ENDPOINT=http://192.168.50.30:1234
EMBEDDINGS_MODEL=text-embedding-nomic-embed-text-v1.5
```

**OpenAI** (`.env.openai`)
```env
GENAI_PROVIDER=openai
GENAI_MODEL=gpt-3.5-turbo
GENAI_API_KEY=sk-...

EMBEDDINGS_PROVIDER=openai
EMBEDDINGS_MODEL=text-embedding-3-small
EMBEDDINGS_API_KEY=sk-...
```

**Azure** (`.env.azure`)
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
