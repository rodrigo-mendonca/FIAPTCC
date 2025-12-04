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
