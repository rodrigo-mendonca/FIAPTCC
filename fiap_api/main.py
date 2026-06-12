from fastapi import FastAPI, HTTPException, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
import os
import sys
import asyncio
from contextlib import asynccontextmanager
from dotenv import load_dotenv
import json

# Console do Windows usa cp1252; os logs imprimem caracteres unicode (✓, ✗, ⚠️).
# Sem isto, um print pode lançar UnicodeEncodeError e abortar a inicialização.
for _stream in (sys.stdout, sys.stderr):
    try:
        _stream.reconfigure(encoding="utf-8")
    except Exception:
        pass
from typing import AsyncGenerator, List, Optional, Dict, Any
import requests
from factories import GenAIFactory, EmbeddingsFactory, ChromaDBClient, EnvFactory, FileValidator, SQLFactory, sql_factory
from factories.genai_factory import ChatResponseGenerator

# Carrega variáveis de ambiente

load_dotenv()

# Corrige uso de CHROMADB_DEFAULT_RESULTS vindo do .env
CHROMADB_DEFAULT_RESULTS = int(os.getenv("CHROMADB_DEFAULT_RESULTS", 100))

# Carrega system prompts do arquivo JSON
def load_system_prompts():
    try:
        with open('system_prompts.json', 'r', encoding='utf-8') as f:
            return json.load(f)
    except FileNotFoundError:
        return {
            "help": "Responda SOMENTE baseado no contexto fornecido abaixo.",
            "sql": "Você é um especialista em SQL.",
            "aluno": "Você é um assistente de aprendizado."
        }
    except Exception as e:
        return {
            "help": "Responda SOMENTE baseado no contexto fornecido abaixo.",
            "sql": "Você é um especialista em SQL.",
            "aluno": "Você é um assistente de aprendizado."
        }

SYSTEM_PROMPTS = load_system_prompts()

# Configuração - Variáveis obrigatórias serão validadas via EnvFactory
ENVIRONMENT = os.getenv("ENVIRONMENT", "local")

# Inicializar GenAI
try:
    genai = GenAIFactory.create()
    print(f"[OK] GenAI iniciado: provider={os.getenv('GENAI_PROVIDER', 'lmstudio')}")
except Exception as e:
    print(f"[ERROR] Erro ao inicializar GenAI: {e}")
    genai = None

# Inicializar Embeddings
try:
    embeddings = EmbeddingsFactory.create()
    print(f"[OK] Embeddings iniciado")
except Exception as e:
    print(f"[ERROR] Erro ao inicializar Embeddings: {e}")
    embeddings = None

# Inicializar ChromaDB
# IMPORTANTE: o construtor NÃO faz rede; a conexão é feita em background no
# startup (lifespan) para não bloquear o boot da API. chromadb.HttpClient()
# valida a conexão de forma síncrona e, se o servidor ainda não estiver pronto,
# travaria o import inteiro do módulo (e, portanto, a subida do uvicorn).
try:
    chromadb_client = ChromaDBClient()
except Exception as e:
    print(f"[ERROR] Erro ao instanciar ChromaDBClient: {e}")
    chromadb_client = None

# Definido quando a conexão é bem-sucedida (usado pelos geradores de resposta)
chroma_client = None

# Parâmetros de conexão configuráveis via .env
CHROMADB_CONNECT_TIMEOUT = float(os.getenv("CHROMADB_CONNECT_TIMEOUT", "5"))
CHROMADB_CONNECT_MAX_RETRIES = int(os.getenv("CHROMADB_CONNECT_MAX_RETRIES", "0"))  # 0 = tenta indefinidamente


async def _connect_chromadb_with_retry():
    """
    Conecta ao ChromaDB em background, com timeout por tentativa e backoff.

    Roda fora do caminho de import/boot: a API responde imediatamente e passa a
    enxergar o ChromaDB assim que a conexão for estabelecida. Cada tentativa é
    isolada em uma thread com timeout, de modo que um servidor lento/indisponível
    nunca trava o event loop.
    """
    global chroma_client
    if not chromadb_client:
        print("[STARTUP] ChromaDBClient indisponível; API seguirá sem ChromaDB")
        return

    attempt = 0
    delay = 2.0
    while True:
        attempt += 1
        try:
            connected = await asyncio.wait_for(
                asyncio.to_thread(chromadb_client.connect),
                timeout=CHROMADB_CONNECT_TIMEOUT,
            )
        except asyncio.TimeoutError:
            connected = False
            print(f"[STARTUP] Conexão com ChromaDB excedeu {CHROMADB_CONNECT_TIMEOUT}s (tentativa {attempt})")
        except Exception as e:
            connected = False
            print(f"[STARTUP] Erro ao conectar ao ChromaDB (tentativa {attempt}): {e}")

        if connected:
            chroma_client = chromadb_client
            print(f"[STARTUP] ✓ ChromaDB conectado (tentativa {attempt})")
            return

        if CHROMADB_CONNECT_MAX_RETRIES and attempt >= CHROMADB_CONNECT_MAX_RETRIES:
            print(f"[STARTUP] Desistindo após {attempt} tentativas; API seguirá sem ChromaDB")
            return

        await asyncio.sleep(delay)
        delay = min(delay * 2, 30.0)  # backoff exponencial até 30s


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Dispara a conexão em background sem bloquear a subida do servidor
    connect_task = asyncio.create_task(_connect_chromadb_with_retry())
    try:
        yield
    finally:
        connect_task.cancel()


app = FastAPI(title="LMStudio Chat API", version="1.0.0", lifespan=lifespan)

# Configuração CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Em produção, especifique as origens permitidas
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Modelo para mensagem
class Message(BaseModel):
    role: str  # "user" ou "assistant"
    content: str
    timestamp: Optional[str] = None

# Modelo para request
class ChatRequest(BaseModel):
    message: str
    session_id: Optional[str] = None
    context: Optional[List[Message]] = None

# Modelo para limpar chat
class ClearChatRequest(BaseModel):
    session_id: str

# Modelos para ChromaDB
class DatabaseUploadRequest(BaseModel):
    database_name: str
    overwrite: Optional[bool] = False

class CreateCollectionRequest(BaseModel):
    name: str

class DatabaseSearchRequest(BaseModel):
    query: str
    database_name: Optional[str] = None
    limit: Optional[int] = 5

class DatabaseSearchResponse(BaseModel):
    results: List[Dict[str, Any]]
    total_found: int
    query: str

# Novos modelos para teste ChromaDB
class VectorDBQueryRequest(BaseModel):
    question: str
    context: Optional[str] = "all"

class VectorDBQueryResponse(BaseModel):
    question: str
    results: List[Dict[str, Any]]
    total_results: int
    processing_time: float

class DatabaseUpdateRequest(BaseModel):
    database_structure: Dict[str, Any]

# Modelo para upload unificado de arquivo
class UnifiedFileUploadResponse(BaseModel):
    message: str
    file_type: str  # 'regras_negocio', 'base_dados', 'servicos', 'rotinas_usuario'
    file_name: str
    status: str  # 'success', 'error'

# Novos modelos para chat especializado
class ChatMessage(BaseModel):
    role: str  # "user" ou "assistant"
    content: str
    timestamp: Optional[str] = None

class SpecializedChatRequest(BaseModel):
    message: str
    session_id: Optional[str] = None
    context: Optional[List[ChatMessage]] = None

class SpecializedChatResponse(BaseModel):
    response: str
    session_id: str
    context_used: Optional[str] = None

# Sistema simples de gerenciamento de sessões em memória
# Em produção, use Redis ou banco de dados
chat_sessions: Dict[str, List[Message]] = {}

async def generate_specialized_response_stream(
    message: str, 
    system_prompt: str,
    context: Optional[List[ChatMessage]] = None,
    use_chromadb: bool = True,
    collection_name: Optional[str] = None,
    similarity_threshold: float = 0.2
) -> AsyncGenerator[str, None]:
    """
    Gera resposta especializada com streaming usando GenAI com contexto do ChromaDB
    
    Wrapper que chama a função do genai_factory
    """
    async for chunk in ChatResponseGenerator.generate_streaming_response(
        message=message,
        system_prompt=system_prompt,
        context=[{
            "role": msg.role,
            "content": msg.content
        } for msg in (context or [])] if context else None,
        use_chromadb=use_chromadb,
        chromadb_client=chromadb_client,
        similarity_threshold=similarity_threshold,
        collection_name=collection_name
    ):
        yield chunk

def detect_file_type(content: str, filename: str) -> Optional[str]:
    """
    Detecta automaticamente o tipo de arquivo baseado em seu conteúdo e nome
    
    Procura por chaves YAML específicas de cada tipo:
    - tabela: base_dados
    - rules: regras_negocio
    - rotinas: servicos
    - rotinas_usuario: rotinas_usuario
    
    Returns: 'regras_negocio', 'base_dados', 'servicos', 'rotinas_usuario', ou None
    """
    content_lower = content.lower()
    filename_lower = filename.lower()
    
    # Verificar por estruturas YAML específicas (mais confiável)
    
    # Rotinas de Usuário - procura por "rotinas_usuario:"
    if 'rotinas_usuario:' in content_lower:
        return 'rotinas_usuario'
    
    # Serviços do Sistema - procura por "rotinas:" sem "rotinas_usuario"
    if 'rotinas:' in content_lower and 'rotinas_usuario:' not in content_lower:
        # Verificar também por tipo_servico
        if 'tipo_servico:' in content_lower or 'tipo: backup' in content_lower or 'tipo: sincronizacao' in content_lower:
            return 'servicos'
    
    # Base de Dados - procura por "tabela:" ou "colunas:"
    if 'tabela:' in content_lower or ('chave_primaria:' in content_lower and 'colunas:' in content_lower):
        return 'base_dados'
    
    # Regras de Negócio - procura por "rules:" ou "table_name:" com "rules"
    if 'rules:' in content_lower or ('table_name:' in content_lower and 'rule_id:' in content_lower):
        return 'regras_negocio'
    
    # Fallback: procurar por palavras-chave indicadoras de tipo
    
    # Rotinas de usuário - palavras-chave
    if any(word in content_lower for word in ['procedimento', 'passo a passo', 'passos:', 'papeis_necessarios', 'tempo_estimado', 'frequencia_execucao', 'usuarios_alvo']):
        return 'rotinas_usuario'
    
    # Serviços - palavras-chave
    if any(word in content_lower for word in ['backup', 'sincronizacao', 'limpeza', 'manutencao', 'frequencia: diaria', 'horario:', 'schedule', 'automation', 'automacao', 'rotina_sistema']):
        if 'rotinas_usuario:' not in content_lower:
            return 'servicos'
    
    # Regras de negócio - palavras-chave
    if any(word in content_lower for word in ['validação', 'política', 'limite', 'desconto', 'aprovação', 'constraint', 'validation', 'business_rule', 'regra_validacao', 'tabela_desconto', 'tabela_limite']):
        return 'regras_negocio'
    
    # Base de Dados - palavras-chave mais específicas
    if any(word in content_lower for word in ['coluna', 'relacionamento', 'índice', 'chave_primaria', 'tipo: integer', 'tipo: varchar', 'colunas_importantes', 'modelo_csharp', 'database:', 'schema']):
        return 'base_dados'
    
    # Tentar detectar pelo nome do arquivo
    if 'negocio' in filename_lower or 'rule' in filename_lower or '_rules' in filename_lower or 'desconto' in filename_lower or 'limite' in filename_lower:
        return 'regras_negocio'
    if 'database' in filename_lower or 'estrutura' in filename_lower or 'table' in filename_lower or 'clientes' in filename_lower or 'produtos' in filename_lower or 'vendas' in filename_lower or 'tmov' in filename_lower or 'tcad' in filename_lower:
        return 'base_dados'
    if 'servico' in filename_lower or 'service' in filename_lower or 'backup' in filename_lower or 'automacao' in filename_lower or 'sync' in filename_lower or 'sincronizacao' in filename_lower:
        return 'servicos'
    if 'rotina' in filename_lower or 'procedimento' in filename_lower or 'workflow' in filename_lower or 'user_routine' in filename_lower:
        return 'rotinas_usuario'
    
    # Base de dados - palavras-chave
    if any(word in content_lower for word in ['coluna', 'relacionamento', 'índice', 'chave_primaria', 'tipo: integer', 'tipo: varchar']):
        return 'base_dados'
    
    # Tentar detectar pelo nome do arquivo
    if 'negocio' in filename_lower or 'rule' in filename_lower or '_rules' in filename_lower:
        return 'regras_negocio'
    if 'database' in filename_lower or 'estrutura' in filename_lower or 'table' in filename_lower or 'clientes' in filename_lower or 'produtos' in filename_lower or 'vendas' in filename_lower:
        return 'base_dados'
    if 'servico' in filename_lower or 'service' in filename_lower or 'backup' in filename_lower or 'automacao' in filename_lower or 'sync' in filename_lower:
        return 'servicos'
    if 'rotina' in filename_lower or 'usuario' in filename_lower or 'procedure' in filename_lower or 'processo' in filename_lower:
        return 'rotinas_usuario'
    
    return None


async def save_yaml_file(content: str, file_type: str, filename: str) -> bool:
    """Salva arquivo YAML na pasta correta"""
    try:
        import yaml
        # Mapear tipo para pasta
        folder_map = {
            'regras_negocio': 'regras_negocio',
            'base_dados': 'base_dados',
            'servicos': 'servicos',
            'rotinas_usuario': 'rotinas_usuario'
        }
        if file_type not in folder_map:
            return False
        folder_name = folder_map[file_type]

        # Usar DATA_DIR se existir, senão caminho relativo
        import tempfile
        data_dir = os.getenv('DATA_DIR')
        if data_dir:
            base_path = os.path.join(data_dir, folder_name)
        else:
            # Usa a pasta temporária do sistema operacional
            temp_dir = tempfile.gettempdir()
            base_path = os.path.join(temp_dir, 'fiap_uploads', folder_name)
        os.makedirs(base_path, exist_ok=True)

        # Nome do arquivo (remover extensão antiga e adicionar .yaml)
        file_base_name = os.path.splitext(filename)[0]
        filepath = os.path.join(base_path, f"{file_base_name}.yaml")

        # Tentar parsear como YAML ou JSON
        try:
            data = yaml.safe_load(content)
        except Exception:
            try:
                data = json.loads(content)
            except Exception:
                # Se não for YAML nem JSON válido, salvar como está
                with open(filepath, 'w', encoding='utf-8') as f:
                    f.write(content)
                return True

        # Salvar como YAML
        with open(filepath, 'w', encoding='utf-8') as f:
            yaml.dump(data, f, default_flow_style=False, allow_unicode=True, sort_keys=False)
        return True
    except Exception as e:
        print(f"[ERROR] save_yaml_file: {e} | Caminho: {filepath if 'filepath' in locals() else 'N/A'}")
        return False

# ==================== Nova Função: Streaming com MCP Tools ====================

async def generate_response_with_tools_stream(
    message: str,
    system_prompt: str,
    context: Optional[List[Dict]] = None,
    collection_name: Optional[str] = None,
) -> AsyncGenerator[str, None]:
    """
    Versão com suporte a MCP Tools
    Usa LangGraph Agent + streaming de tokens
    """
    if not genai or not hasattr(genai, 'is_lmstudio_with_mcp'):
        # Fallback para método antigo se não estiver usando MCP
        async for chunk in ChatResponseGenerator.generate_streaming_response(
            message=message,
            system_prompt=system_prompt,
            context=context,
            use_chromadb=bool(collection_name),
            chromadb_client=chroma_client,
            collection_name=collection_name
        ):
            yield chunk
        return

    # ====================== Caminho com MCP Tools ======================
    try:
        # Construir mensagens
        messages = []
        messages.append({"role": "system", "content": system_prompt})

        if context:
            messages.extend(context[-10:])  # último 10 mensagens

        messages.append({"role": "user", "content": message})

        # Usar o wrapper que criamos anteriormente
        async for chunk in stream_agent_response(
            genai_wrapper=genai,
            messages=messages,
            chromadb_client=chromadb_client,
            collection_name=collection_name
        ):
            yield chunk

    except Exception as e:
        error_msg = f"Erro ao processar com ferramentas: {str(e)}"
        yield f"data: {json.dumps({'content': error_msg})}\n\n"


async def stream_agent_response(
    genai_wrapper,
    messages: list,
    chromadb_client=None,
    collection_name: str | None = None
):
    from langchain_core.messages import HumanMessage, SystemMessage, AIMessage
    from langgraph.prebuilt import create_react_agent
    from langchain_core.tools import tool
    import json
    import traceback

    lc_messages = []

    for msg in messages:
        role = msg.get("role")
        content = msg.get("content", "")

        if role == "system":
            lc_messages.append(SystemMessage(content=content))
        elif role == "assistant":
            lc_messages.append(AIMessage(content=content))
        else:
            lc_messages.append(HumanMessage(content=content))

    # ==========================================
    # Mantém tools originais vindas do wrapper
    # ==========================================
    tools = list(getattr(genai_wrapper, "tools", []) or [])

    # ==========================================
    # Adiciona Tool ChromaDB se collection existir
    # ==========================================
    if chromadb_client and collection_name:
        print("[INFO] Adicionando ferramenta de busca no ChromaDB para a coleção:", collection_name)
        @tool
        def search_knowledge_base(query: str) -> str:
            """
            Use esta ferramenta SEMPRE que a pergunta envolver:
            - Documentação
            - Regras de negócio
            - Processos internos
            - Estrutura técnica da aplicação
            - Para ajudar buscar de informações na base de dados
            - Informações sobre a base de dados
            """
            try:
                print("[INFO] Consultando ChromaDB para a coleção:", collection_name)
                print("[INFO] Consultando ChromaDB com a query:", query)
                chromadb_client.set_collection(collection_name)
                results = chromadb_client.query(
                    query,
                    n_results=CHROMADB_DEFAULT_RESULTS
                )
                print("results", results)
                if not results:
                    return "Nenhum resultado encontrado."

                formatted = []

                for idx, result in enumerate(results, 1):
                    formatted.append(
                        f"[{idx}] "
                        f"Conteúdo: {result.get('content', '')}\n"
                        f"Metadata: {result.get('metadata', {})}\n"
                        f"Similaridade: {result.get('similarity', 0):.2f}"
                    )

                return "\n\n".join(formatted)

            except Exception as e:
                return f"Erro ao consultar ChromaDB: {str(e)}"

        tools.append(search_knowledge_base)

    # ==========================================
    # MESMA CONFIG do generate_response_with_tools_stream
    # ==========================================
    agent = create_react_agent(
        model=genai_wrapper.llm,
        tools=tools
    )

    try:
        async for event in agent.astream_events(
            {"messages": lc_messages},
            version="v2"
        ):
            event_type = event["event"]

            if event_type == "on_chat_model_stream":
                chunk = event["data"]["chunk"].content

                if chunk:
                    yield f"data: {json.dumps({'content': chunk})}\n\n"

            elif event_type == "on_tool_start":
                payload = {
                    "content": "\n[Executando consulta]\n"
                }

                yield f"data: {json.dumps(payload)}\n\n"

            elif event_type == "on_tool_end":
                payload = {
                    "content": "\n[Consulta concluída]\n\n"
                }

                yield f"data: {json.dumps(payload)}\n\n"

    except Exception as e:
        traceback.print_exc()

        yield f"data: {json.dumps({'content': f'Erro interno do agente: {str(e)}'})}\n\n"


@app.post("/api/vectordb/upload")
async def upload_file_unified(file: UploadFile = File(...), collection_name: str = Form(...)) -> UnifiedFileUploadResponse:
    """
    Endpoint unificado para upload de arquivo
    """
    try:
        # Validar que uma coleção foi fornecida
        if not collection_name or not collection_name.strip():
                raise HTTPException(status_code=400, detail="collection_name é obrigatório")
        
        target_collection = collection_name
        # Validar que temos um arquivo
        if not file or file.size == 0:
                raise HTTPException(status_code=400, detail="Arquivo vazio ou inválido")
        
        # Verificar se ChromaDB está disponível
        if not chromadb_client or not chromadb_client.client:
            raise HTTPException(status_code=503, detail="ChromaDB não está disponível. Tente reconectar.")
        
        # Ler conteúdo do arquivo
        content = await file.read()
        content_str = content.decode('utf-8')
        
        # Detectar tipo de arquivo
        detected_type = detect_file_type(content_str, file.filename or "unknown")
        if not detected_type:
            raise HTTPException(
                status_code=400, 
                detail=f"Não foi possível detectar o tipo de arquivo. Verifique o conteúdo do arquivo.\nArquivo: {file.filename}"
            )
        
        # Salvar arquivo na pasta correta
        save_success = await save_yaml_file(content_str, detected_type, file.filename or "documento.yaml")
        
        if not save_success:
            raise HTTPException(status_code=400, detail="Erro ao salvar arquivo")
        
        # Reindexar ChromaDB
        try:
            # Primeiro, garantir que a coleção está selecionada
            if not chromadb_client.set_collection(target_collection):
                if not chromadb_client.create_collection(target_collection):
                    return UnifiedFileUploadResponse(
                        message=f"Arquivo salvo mas não foi possível criar a coleção '{target_collection}'",
                        file_type=detected_type,
                        file_name=file.filename or "documento",
                        status="error"
                    )
            return UnifiedFileUploadResponse(
                message=f"Arquivo '{file.filename}' carregado com sucesso como {detected_type}",
                file_type=detected_type,
                file_name=file.filename or "documento",
                status="success"
            )
        except Exception as e:
            return UnifiedFileUploadResponse(
                message=f"Arquivo '{file.filename}' salvo mas houve erro ao reindexar: {str(e)}",
                file_type=detected_type,
                file_name=file.filename or "documento",
                status="error"
            )
        
    except HTTPException:
        raise
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Erro ao processar arquivo: {str(e)}")


async def validate_file_with_llm(content: str, filename: str, detected_type: Optional[str]) -> Dict[str, Any]:
    """
    Valida arquivo usando LLM se o tipo não foi detectado ou se há dúvida
    Retorna análise da LLM sobre o arquivo
    
    Usa FileValidator do genai_factory
    """
    return await FileValidator.validate_with_llm(
        genai_client=genai,
        content=content,
        filename=filename,
        detected_type=detected_type
    )


async def _process_uploaded_file(
    filename: str,
    content_bytes: bytes,
    size: Optional[int],
    target_collection: str
) -> Dict[str, Any]:
    """
    Processa um único arquivo já lido em memória: detecta o tipo, salva em disco
    e indexa no ChromaDB. Retorna um dict com o resultado (sucesso ou erro).

    Extraído do loop do upload em lote para permitir o streaming de progresso.
    """
    try:
        if not content_bytes or len(content_bytes) == 0:
            return {
                "filename": filename,
                "status": "error",
                "message": "Arquivo vazio",
                "type": None
            }

        content_str = content_bytes.decode('utf-8')

        # Detectar tipo
        detected_type = detect_file_type(content_str, filename or "unknown")

        # Se não detectou, usar LLM para validar
        if not detected_type:
            validation = await validate_file_with_llm(content_str, filename or "unknown", detected_type)
            detected_type = validation.get("detected_type")
            llm_info = validation.get("llm_analysis", {})

            if not detected_type:
                return {
                    "filename": filename,
                    "status": "error",
                    "message": "Arquivo não pôde ser classificado. Verifique o conteúdo.",
                    "type": None,
                    "llm_suggestion": llm_info
                }

        # Salvar arquivo
        save_success = await save_yaml_file(content_str, detected_type, filename or "documento.yaml")

        if not save_success:
            return {
                "filename": filename,
                "status": "error",
                "message": "Erro ao salvar arquivo no servidor",
                "type": detected_type
            }

        # Indexar arquivo imediatamente após salvar
        try:
            file_base_name = os.path.splitext(filename or "documento")[0]

            if detected_type == 'base_dados':
                try:
                    import yaml
                    data = yaml.safe_load(content_str)
                    if data and 'tabela' in data:
                        # Obter o nome da tabela (pode ser string ou dict)
                        table_name_value = data['tabela']
                        if isinstance(table_name_value, dict):
                            table_name = table_name_value.get("nome", file_base_name)
                            table_metadata = table_name_value
                        else:
                            table_name = str(table_name_value)
                            table_metadata = data  # Usar data completo como metadata

                        # 1. Criar documento principal da tabela
                        table_desc = table_metadata.get("descricao", "") if isinstance(table_metadata, dict) else ""
                        table_text = f"Tabela {table_name}: {table_desc}"
                        doc_id = f"table_{file_base_name}"

                        chromadb_client.add_document(
                            text=table_text,
                            metadata={
                                "type": "table",
                                "table_name": table_name,
                                "source_file": filename,
                                "source": "user_upload"
                            },
                            id=doc_id,
                            collection_name=target_collection
                        )

                        # 2. Indexar colunas importantes (estrutura nova dos docs_cg)
                        colunas_importantes = data.get("colunas_importantes", [])
                        if isinstance(colunas_importantes, list):
                            for col in colunas_importantes:
                                if isinstance(col, dict) and col.get("nome"):
                                    col_name = col.get("nome")
                                    col_text = f"Coluna {col_name}: {col.get('descricao', '')} | Tipo: {col.get('tipo', 'unknown')} | Nulo: {col.get('nulo', 'N/A')}"
                                    col_doc_id = f"field_{file_base_name}_{col_name}"

                                    chromadb_client.add_document(
                                        text=col_text,
                                        metadata={
                                            "type": "field",
                                            "table_name": table_name,
                                            "field_name": col_name,
                                            "data_type": col.get("tipo", "unknown"),
                                            "nullable": col.get("nulo", False),
                                            "source": "user_upload"
                                        },
                                        id=col_doc_id,
                                        collection_name=target_collection
                                    )

                        # 3. Fallback: indexar fields para estrutura antiga (se houver)
                        if isinstance(table_metadata, dict):
                            fields = table_metadata.get("fields", {})
                            if isinstance(fields, dict):
                                for field_name, field_data in fields.items():
                                    if isinstance(field_data, dict) and field_data.get("pesquisavel", True):
                                        field_desc = field_data.get("descricao", "")
                                        field_type = field_data.get("tipo", "unknown")
                                        field_text = f"Campo {field_name} da tabela {table_name} ({field_type}): {field_desc}"
                                        field_doc_id = f"field_{file_base_name}_{field_name}"

                                        chromadb_client.add_document(
                                            text=field_text,
                                            metadata={
                                                "type": "field",
                                                "table_name": table_name,
                                                "field_name": field_name,
                                                "data_type": field_data.get("tipo", "unknown"),
                                                "source": "user_upload"
                                            },
                                            id=field_doc_id,
                                            collection_name=target_collection
                                        )
                except Exception as index_error:
                    import traceback
                    print(f"[ERROR] Erro ao indexar base_dados: {index_error}")
                    traceback.print_exc()
            else:
                # Para outros tipos, fazer indexação genérica
                doc_id = f"{detected_type}_{file_base_name}"
                chromadb_client.add_document(
                    text=content_str[:1000],
                    metadata={
                        "type": detected_type,
                        "source_file": filename,
                        "source": "user_upload"
                    },
                    id=doc_id,
                    collection_name=target_collection
                )

        except Exception as index_error:
            import traceback
            print(f"[ERROR] Erro ao indexar documento genérico: {index_error}")
            traceback.print_exc()

        return {
            "filename": filename,
            "status": "success",
            "message": f"Arquivo '{filename}' importado com sucesso como {detected_type}",
            "type": detected_type,
            "size": size
        }

    except Exception as e:
        import traceback
        traceback.print_exc()
        print(f"[ERROR] Erro ao processar arquivo '{filename}': {str(e)}")
        return {
            "filename": filename,
            "status": "error",
            "message": str(e),
            "type": None
        }


@app.post("/api/vectordb/upload-batch")
async def upload_files_batch(
    files: List[UploadFile] = File(...),
    collection_name: str = Form(...)
):
    """
    Endpoint para upload em lote de múltiplos arquivos, com progresso via
    streaming (Server-Sent Events).

    Em vez de processar tudo e só então responder (o que causava timeout no
    navegador em lotes grandes), a resposta é enviada incrementalmente: um
    evento por arquivo conforme ele é processado, mantendo a conexão ativa.

    Eventos emitidos (cada linha `data: <json>`):
    - {"event": "start", "total": N, "collection": "..."}
    - {"event": "file_start", "index": i, "total": N, "filename": "..."}
    - {"event": "file_done", "index": i, "total": N, ...resultado do arquivo}
    - {"event": "complete", "success_count", "error_count", "results", ...}
    """
    # Validações básicas (antes de iniciar o stream, para retornar HTTP de erro)
    if not collection_name or not collection_name.strip():
        raise HTTPException(status_code=400, detail="collection_name é obrigatório")

    if not files or len(files) == 0:
        raise HTTPException(status_code=400, detail="Forneça pelo menos um arquivo")

    if not chromadb_client or not chromadb_client.client:
        raise HTTPException(status_code=503, detail="ChromaDB não está disponível")

    # Garantir que a coleção existe
    target_collection = collection_name
    if not chromadb_client.set_collection(target_collection):
        if not chromadb_client.create_collection(target_collection):
            raise HTTPException(status_code=500, detail=f"Não foi possível criar coleção '{target_collection}'")

    # Ler todo o conteúdo dos arquivos ANTES de iniciar o stream — o corpo da
    # requisição precisa ser consumido enquanto o request ainda está disponível.
    buffered_files = []
    for file in files:
        try:
            raw = await file.read()
        except Exception:
            raw = b""
        buffered_files.append({
            "filename": file.filename,
            "size": file.size if file.size is not None else len(raw),
            "content": raw,
        })

    async def event_stream() -> AsyncGenerator[str, None]:
        results = []
        total = len(buffered_files)

        yield f"data: {json.dumps({'event': 'start', 'total': total, 'collection': target_collection})}\n\n"

        for idx, item in enumerate(buffered_files, 1):
            filename = item["filename"] or "documento.yaml"

            yield f"data: {json.dumps({'event': 'file_start', 'index': idx, 'total': total, 'filename': filename})}\n\n"
            await asyncio.sleep(0)  # libera o loop para enviar o chunk imediatamente

            result = await _process_uploaded_file(
                filename=filename,
                content_bytes=item["content"],
                size=item["size"],
                target_collection=target_collection
            )
            results.append(result)

            yield f"data: {json.dumps({'event': 'file_done', 'index': idx, 'total': total, **result})}\n\n"
            await asyncio.sleep(0)

        success_count = len([r for r in results if r['status'] == 'success'])
        error_count = len([r for r in results if r['status'] == 'error'])

        complete_payload = {
            'event': 'complete',
            'status': 'success' if success_count > 0 else 'error',
            'total_files': total,
            'success_count': success_count,
            'error_count': error_count,
            'results': results,
            'collection': target_collection,
        }
        yield f"data: {json.dumps(complete_payload)}\n\n"

    return StreamingResponse(
        event_stream(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",  # desabilita buffering em proxies (nginx)
        }
    )


@app.post("/api/vectordb/reconnect")
async def reconnect_chromadb():
    """
    Força reconexão ao ChromaDB
    """
    global chroma_client
    try:
        if not chromadb_client:
            return {"status": "error", "message": "ChromaDB client não inicializado"}
        # connect() é bloqueante; isola em thread para não travar o event loop
        connected = await asyncio.to_thread(chromadb_client.connect)
        if connected:
            chroma_client = chromadb_client
            return {
                "status": "ok",
                "message": "ChromaDB reconectado com sucesso"
            }
        else:
            return {
                "status": "error",
                "message": "Falha ao reconectar ao ChromaDB"
            }
    except Exception as e:
        return {
            "status": "error",
            "message": str(e)
        }

@app.get("/api/vectordb/health")
async def health_check():
    """
    Verifica o status da conexão com ChromaDB
    Se desconectado, tenta reconectar automaticamente
    """
    try:
        if not chromadb_client:
            return {
                "status": "error",
                "message": "ChromaDB client não inicializado",
                "connected": False
            }
        
        # Verificar se está conectado
        if chromadb_client.client:
            try:
                heartbeat = chromadb_client.client.heartbeat()
                return {
                    "status": "ok",
                    "message": "ChromaDB conectado",
                    "connected": True,
                    "heartbeat": heartbeat
                }
            except Exception as e:
                pass
        
        # Tentar reconectar
        if chromadb_client.connect():
            return {
                "status": "ok",
                "message": "ChromaDB reconectado com sucesso",
                "connected": True,
                "reconnected": True
            }
        else:
            return {
                "status": "error",
                "message": "Falha ao reconectar ao ChromaDB",
                "connected": False
            }
    except Exception as e:
        return {
            "status": "error",
            "message": str(e),
            "connected": False
        }

@app.get("/health/chromadb")
async def health_chromadb():
    """Verifica saúde do ChromaDB (compatibilidade com frontend)"""
    try:
        if not chromadb_client or not chromadb_client.client:
            return {"status": "offline", "connected": False}
        
        try:
            chromadb_client.client.heartbeat()
            return {"status": "online", "connected": True}
        except:
            # Tentar reconectar
            if chromadb_client.connect():
                return {"status": "online", "connected": True}
            return {"status": "offline", "connected": False}
    except:
        return {"status": "offline", "connected": False}

@app.get("/health/lmstudio")
async def health_lmstudio():
    """Verifica saúde do GenAI (compatibilidade com frontend)"""
    try:
        genai_params = EnvFactory.get_genai_params()
        # Tentar acessar o endpoint do GenAI
        response = requests.get(f"{genai_params.endpoint}/v1/models", timeout=5)
        if response.status_code == 200:
            return {"status": "online", "connected": True, "provider": genai_params.provider}
        return {"status": "offline", "connected": False, "provider": genai_params.provider}
    except Exception as e:
        return {"status": "offline", "connected": False, "error": str(e)}

@app.get("/api/debug/chromadb-status")
async def debug_chromadb_status():
    """Endpoint de debug para verificar status do ChromaDB"""
    try:
        debug_info = {
            "chromadb_client_initialized": chromadb_client is not None,
            "chromadb_client_has_connection": chromadb_client is not None and chromadb_client.client is not None if chromadb_client else False,
            "chromadb_host": chromadb_client.host if chromadb_client else None,
            "chromadb_port": chromadb_client.port if chromadb_client else None,
        }
        
        if chromadb_client and chromadb_client.client:
            try:
                stats = chromadb_client.get_collection_stats()
                debug_info["stats"] = stats
                debug_info["collections_count"] = len(stats.get("collections", []))
                debug_info["success"] = True
            except Exception as e:
                debug_info["error"] = str(e)
                debug_info["success"] = False
                import traceback
                traceback.print_exc()
        else:
            debug_info["error"] = "ChromaDB client not initialized or not connected"
            debug_info["success"] = False
        
        return debug_info
    except Exception as e:
        return {
            "error": str(e),
            "success": False
        }

@app.get("/health")
async def health_general():
    """Verifica saúde geral da API"""
    return {"status": "ok", "message": "API está funcionando"}


# ============ ROTAS DE CHAT (Assistentes Especializados) ============

@app.post("/api/chat/help/stream")
async def chat_help_stream_endpoint(request: SpecializedChatRequest, collection_name: str = ""):
    system_prompt = SYSTEM_PROMPTS.get("help", "Responda baseado no contexto fornecido.")
    
    async def generate():
        async for chunk in generate_response_with_tools_stream(
            message=request.message,
            system_prompt=system_prompt,
            context=[{"role": m.role, "content": m.content} for m in (request.context or [])],
            collection_name=collection_name or None
        ):
            yield chunk

    return StreamingResponse(generate(), media_type="text/event-stream")


@app.post("/api/chat/aluno/stream")
async def chat_aluno_stream_endpoint(request: SpecializedChatRequest, collection_name: str = ""):
    system_prompt = SYSTEM_PROMPTS.get("aluno", "Você é um assistente de aprendizado.")
    
    async def generate():
        async for chunk in generate_response_with_tools_stream(
            message=request.message,
            system_prompt=system_prompt,
            context=[{"role": m.role, "content": m.content} for m in (request.context or [])],
            collection_name=collection_name or None
        ):
            yield chunk

    return StreamingResponse(generate(), media_type="text/event-stream")


@app.post("/api/chat/sql/stream")
async def chat_sql_stream_endpoint(request: SpecializedChatRequest, collection_name: str = ""):
    system_prompt = SYSTEM_PROMPTS.get("sql", "Você é um especialista em SQL.")
    
    async def generate():
        async for chunk in generate_response_with_tools_stream(
            message=request.message,
            system_prompt=system_prompt,
            context=[{"role": m.role, "content": m.content} for m in (request.context or [])],
            collection_name=collection_name or None
        ):
            yield chunk

    return StreamingResponse(generate(), media_type="text/event-stream")


@app.post("/api/chat/general/stream")
async def chat_general_stream_endpoint(request: SpecializedChatRequest, collection_name: str = ""):
    system_prompt = SYSTEM_PROMPTS.get("general", "Você é um assistente útil.")
    
    async def generate():
        async for chunk in generate_response_with_tools_stream(
            message=request.message,
            system_prompt=system_prompt,
            context=[{"role": m.role, "content": m.content} for m in (request.context or [])],
            collection_name=None  # chat geral não usa ChromaDB
        ):
            yield chunk

    return StreamingResponse(generate(), media_type="text/event-stream")


# ===========================
# FUNÇÕES AUXILIARES E MODELOS
# ===========================

# ============ ROTAS DE VECTORDB (compatibilidade com interface) ============

@app.get("/vectordb/stats")
async def get_vectordb_stats(collection_name: str = ""):
    """Retorna estatísticas da coleção ou de todas as coleções se collection_name estiver vazio"""
    try:
        if not chromadb_client:
            raise HTTPException(status_code=503, detail="ChromaDB não disponível")
        
        # Verifica se client está conectado, se não tenta reconectar
        if not chromadb_client.client:
            chromadb_client.connect()
        
        # Se collection_name está vazio, retorna estatísticas gerais (todas as coleções)
        if not collection_name or collection_name.strip() == "":
            stats = chromadb_client.get_collection_stats()
            if not isinstance(stats, dict):
                stats = {"error": "Resposta inválida do ChromaDB"}
            return stats
        
        # Caso contrário, tenta obter stats de uma coleção específica
        if not chromadb_client.set_collection(collection_name):
            # Lista coleções disponíveis para debug
            try:
                available_stats = chromadb_client.get_collection_stats()
                available_collections = available_stats.get("collections", [])
                collection_names =  [c.get("name", "unknown") for c in available_collections] if isinstance(available_collections, list) else []
                error_msg = f"Coleção '{collection_name}' não existe. Coleções disponíveis: {collection_names}"
            except:
                error_msg = f"Coleção '{collection_name}' não existe."
            
            raise HTTPException(status_code=404, detail=error_msg)
        
        stats = chromadb_client.get_collection_stats()
        
        # Garantir que é um dicionário válido
        if not isinstance(stats, dict):
            stats = {"error": "Resposta inválida do ChromaDB"}
        
        return stats
    except HTTPException:
        raise
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/vectordb/collections")
async def list_collections():
    """Retorna lista de todas as coleções disponíveis"""
    try:
        if not chromadb_client:
            raise HTTPException(status_code=503, detail="ChromaDB não disponível")
        
        stats = chromadb_client.get_collection_stats()
        collections = stats.get("collections", [])
        
        return {
            "status": "success",
            "total": len(collections),
            "collections": collections
        }
    except HTTPException:
        raise
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/debug/chromadb/{collection_name}")
async def debug_chromadb(collection_name: str):
    """Endpoint de debug para testar a conexão e disponibilidade de uma coleção"""
    try:
        debug_info = {
            "collection_name": collection_name,
            "chromadb_client_exists": chromadb_client is not None,
            "tests": {}
        }
        
        if not chromadb_client:
            debug_info["error"] = "ChromaDB client não inicializado"
            return debug_info
        
        # Teste 1: Verificar se client existe
        debug_info["tests"]["client_exists"] = hasattr(chromadb_client, 'client') and chromadb_client.client is not None
        
        # Teste 2: Tentar definir a coleção
        try:
            collection_set = chromadb_client.set_collection(collection_name)
            debug_info["tests"]["set_collection_success"] = collection_set
            
            if not collection_set:
                # Listar coleções disponíveis
                try:
                    all_stats = chromadb_client.get_collection_stats()
                    debug_info["available_collections"] = all_stats
                except Exception as list_e:
                    debug_info["available_collections_error"] = str(list_e)
        except Exception as sc_e:
            debug_info["tests"]["set_collection_error"] = str(sc_e)
            debug_info["tests"]["set_collection_success"] = False
        
        # Teste 3: Verificar se collection está definida
        debug_info["tests"]["has_query_method"] = hasattr(chromadb_client, 'query')
        debug_info["tests"]["has_collection_attr"] = hasattr(chromadb_client, 'collection')
        
        # Teste 4: Tentar fazer uma query simples
        if hasattr(chromadb_client, 'query') and debug_info["tests"]["set_collection_success"]:
            try:
                results = chromadb_client.query("test", n_results=1)
                debug_info["tests"]["query_success"] = True
                debug_info["tests"]["query_result_count"] = len(results) if results else 0
            except Exception as q_e:
                debug_info["tests"]["query_error"] = str(q_e)
                debug_info["tests"]["query_success"] = False
        
        return debug_info
        
    except Exception as e:
        import traceback
        traceback.print_exc()
        return {"error": str(e), "collection_name": collection_name}


@app.post("/vectordb/create-collection")
async def create_collection(request: CreateCollectionRequest):
    """Cria uma nova coleção"""
    try:
        if not chromadb_client:
            raise HTTPException(status_code=503, detail="ChromaDB não disponível")
        
        collection_name = request.name
        if chromadb_client.create_collection(collection_name):
            return {"message": f"Coleção '{collection_name}' criada com sucesso", "status": "success"}
        else:
            raise HTTPException(status_code=500, detail=f"Falha ao criar coleção '{collection_name}'")
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/vectordb/collection/{collection_name}")
async def get_collection(collection_name: str):
    """Obtém informações da coleção"""
    try:
        if not chromadb_client:
            raise HTTPException(status_code=503, detail="ChromaDB não disponível")
        
        if not chromadb_client.set_collection(collection_name):
            raise HTTPException(status_code=404, detail=f"Coleção '{collection_name}' não existe")
        
        stats = chromadb_client.get_collection_stats()
        return stats
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/vectordb/collection/{collection_name}/delete")
async def delete_collection_post(collection_name: str):
    """Deleta uma coleção (via POST como workaround)"""
    try:
        if not chromadb_client:
            raise HTTPException(status_code=503, detail="ChromaDB não disponível")
        
        if chromadb_client.delete_collection(collection_name):
            return {"message": f"Coleção '{collection_name}' deletada com sucesso", "status": "success"}
        else:
            raise HTTPException(status_code=500, detail=f"Falha ao deletar coleção '{collection_name}'")
    except HTTPException:
        raise
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))


@app.delete("/vectordb/collection/{collection_name}")
async def delete_collection_endpoint(collection_name: str):
    """Deleta uma coleção"""
    try:
        if not chromadb_client:
            raise HTTPException(status_code=503, detail="ChromaDB não disponível")
        
        if chromadb_client.delete_collection(collection_name):
            return {"message": f"Coleção '{collection_name}' deletada com sucesso", "status": "success"}
        else:
            raise HTTPException(status_code=500, detail=f"Falha ao deletar coleção '{collection_name}'")
    except HTTPException:
        raise
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/vectordb/collection/{collection_name}/reindex")
async def reindex_collection(collection_name: str):
    """
    Recria a coleção e reindexa todos os YAML do disco.

    Necessário após mudar a métrica de distância: o `hnsw:space` é fixado na
    criação da coleção e não muda em coleções já existentes. Este endpoint
    apaga a coleção (limpando a métrica L2 antiga e vetores ruins) e a
    reconstrói com cosseno + a mesma embedding_function da busca.
    """
    try:
        if not chromadb_client or not chromadb_client.client:
            raise HTTPException(status_code=503, detail="ChromaDB não disponível")

        # Resolve a pasta base dos YAML (mesma lógica de save_yaml_file)
        import tempfile
        data_dir = os.getenv('DATA_DIR')
        base_path = data_dir if data_dir else os.path.join(tempfile.gettempdir(), 'fiap_uploads')

        # 1. Apaga a coleção existente (remove métrica L2 e vetores antigos)
        try:
            chromadb_client.delete_collection(collection_name)
        except Exception as del_err:
            print(f"[REINDEX] Coleção '{collection_name}' não existia ou falhou ao deletar: {del_err}")

        # 2. Recria com cosseno (embedding_function correta) e reingesta os YAML
        #    do disco na MESMA coleção nomeada usada pela busca
        summary = chromadb_client.reindex_collection_from_folder(base_path, collection_name)

        return {
            "status": "success",
            "message": f"Coleção '{collection_name}' reindexada com métrica de cosseno",
            "base_path": base_path,
            "summary": summary,
        }
    except HTTPException:
        raise
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/vectordb/query")
async def query_vectordb(request: dict):
    """Faz uma busca na coleção"""
    try:
        if not chromadb_client:
            raise HTTPException(status_code=503, detail="ChromaDB não disponível")
        
        question = request.get("question", "")
        collection_name = request.get("collection_name")
        
        # Validar que collection_name foi fornecido
        if not collection_name:
            return {"status": "error", "message": "collection_name é obrigatório"}
        
        n_results = request.get("n_results", CHROMADB_DEFAULT_RESULTS)
        
        print(f"[DEBUG] Querying ChromaDB collection '{collection_name}' with question: {question} and n_results: {n_results}")

        # Definir a coleção
        chromadb_client.set_collection(collection_name)
        
        # Executar a query
        print(f"[DEBUG] Numero resultados {n_results}")
        results = chromadb_client.query(question, n_results=n_results)
        
        # Formatar resultados de forma clara
        formatted_results = []
        table_names = set()
        
        if results:
            for result in results:
                table_name = result.get('metadata', {}).get('table_name', '')
                if table_name:
                    table_names.add(table_name)
                
                formatted_results.append({
                    'id': result.get('id', ''),
                    'type': result.get('type', 'unknown'),
                    'table': table_name,
                    'content': result.get('content', '')[:300],  # Limitar a 300 chars
                    'relevance': round(result.get('similarity', 0) * 100, 1),
                    'metadata': result.get('metadata', {})
                })
        
        return {
            "question": question,
            "tables_found": list(sorted(table_names)),
            "results": formatted_results,
            "total_results": len(formatted_results)
        }
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/vectordb/add-item")
async def add_item_vectordb(request: dict):
    """Adiciona um item à coleção"""
    try:
        if not chromadb_client:
            raise HTTPException(status_code=503, detail="ChromaDB não disponível")
        
        collection_name = request.get("collection_name")
        if not collection_name:
            return {"status": "error", "message": "collection_name é obrigatório"}
        
        content = request.get("content", "")
        metadata = request.get("metadata", {})
        
        chromadb_client.set_collection(collection_name)
        chromadb_client.add_document(content, metadata)
        
        return {"message": "Item adicionado com sucesso"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ============ ROTAS DE DADOS DE MERCADO (Dashboard / Explorar / Exportar) ============
# Dados de CNPJs e aberturas de empresas consumidos pelas telas do frontend.
# Os dados são derivados da base dimensional via SQLFactory (factories/sql_factory.py).

def _require_market_db():
    """Garante que há conexão com o banco para servir os dados de mercado."""
    if sql_factory is None:
        raise HTTPException(
            status_code=503,
            detail="Banco de dados indisponível (DATABASE_URL não configurada).",
        )
    return sql_factory


@app.get("/api/market/metrics")
async def get_market_metrics():
    """Cartões de métricas da visão geral do mercado (Dashboard)."""
    return _require_market_db().get_market_metrics()


@app.get("/api/market/setores")
async def get_market_setores():
    """Aberturas por setor – top CNAEs (Dashboard, gráfico de barras)."""
    return _require_market_db().get_market_setores()


@app.get("/api/market/porte")
async def get_market_porte():
    """Distribuição de empresas por porte (Dashboard, gráfico de rosca)."""
    return _require_market_db().get_market_porte()


@app.get("/api/market/insights")
async def get_market_insights():
    """Cartões de insights do mercado (Dashboard)."""
    return _require_market_db().get_market_insights()


@app.get("/api/market/filtros")
async def get_market_filtros():
    """Opções dos filtros (estados, portes, períodos, setores) – Explorar Mercado."""
    return _require_market_db().get_market_filtros()


@app.get("/api/market/evolution")
async def get_market_evolution(
    uf: str = "Todos",
    porte: str = "Todos",
    cnae: str = "Todos os setores",
    periodo: str = "Todos",
):
    """Evolução de aberturas por mês (Explorar Mercado), filtrada por uf/porte/cnae/período."""
    return _require_market_db().get_market_evolution(uf, porte, cnae, periodo)


@app.get("/api/market/cidades")
async def get_market_cidades(
    uf: str = "Todos",
    porte: str = "Todos",
    cnae: str = "Todos os setores",
    periodo: str = "Todos",
):
    """Top 5 cidades por volume de abertura (Explorar Mercado), filtrada pelos mesmos filtros."""
    return _require_market_db().get_market_cidades(uf, porte, cnae, periodo)


@app.get("/api/market/export-chart")
async def get_market_export_chart():
    """Dados do gráfico de aberturas por setor usado na tela de Exportar."""
    return _require_market_db().get_market_export_chart()


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8001)

