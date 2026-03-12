from fastapi import FastAPI, HTTPException, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
import os
from dotenv import load_dotenv
import json
from typing import AsyncGenerator, List, Optional, Dict, Any
import requests
from factories import GenAIFactory, EmbeddingsFactory, ChromaDBClient, EnvFactory, FileValidator
from factories.genai_factory import ChatResponseGenerator

# Carrega variáveis de ambiente
load_dotenv()

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
try:
    chromadb_client = ChromaDBClient()
    
    # Tenta conectar ao ChromaDB
    if chromadb_client.connect():
        chroma_client = chromadb_client
        print(f"[OK] ChromaDB iniciado")
    else:
        chroma_client = None
except Exception as e:
    print(f"[ERROR] Erro ao inicializar ChromaDB: {e}")
    chroma_client = None
    chromadb_client = None

app = FastAPI(title="LMStudio Chat API", version="1.0.0")

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
    similarity_threshold: float = 0.3
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
        
        # Caminho da pasta
        base_path = os.path.join(os.path.dirname(__file__), '..', 'tests', 'chromadb', 'data', folder_name)
        os.makedirs(base_path, exist_ok=True)
        
        # Nome do arquivo (remover extensão antiga e adicionar .yaml)
        file_base_name = os.path.splitext(filename)[0]
        filepath = os.path.join(base_path, f"{file_base_name}.yaml")
        
        # Tentar parsear como YAML ou JSON
        try:
            data = yaml.safe_load(content)
        except:
            try:
                data = json.loads(content)
            except:
                # Se não for YAML nem JSON válido, salvar como está
                with open(filepath, 'w', encoding='utf-8') as f:
                    f.write(content)
                return True
        
        # Salvar como YAML
        with open(filepath, 'w', encoding='utf-8') as f:
            yaml.dump(data, f, default_flow_style=False, allow_unicode=True, sort_keys=False)
        
        return True
    except Exception as e:
        return False


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


@app.post("/api/vectordb/upload-batch")
async def upload_files_batch(
    files: List[UploadFile] = File(...), 
    collection_name: str = Form(...),
    include_metadata: str = Form(default="false")
) -> Dict[str, Any]:
    """
    Endpoint para upload em lote de múltiplos arquivos
    
    Características:
    - Suporta N arquivos
    - Auto-detecta categoria (base_dados, regras_negocio, servicos, rotinas_usuario)
    - Valida com LLM se arquivo não tiver categoria clara
    - Opcionalmente importa metadata (estrutura de documentação)
    - Processa sequencialmente mantendo estado
    """
    try:
        # Validações básicas
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
        
        results = []
        include_meta = include_metadata.lower() == "true"
        
        # Processar cada arquivo
        for idx, file in enumerate(files, 1):
            try:
                # Validar arquivo
                if not file or file.size == 0:
                    results.append({
                        "filename": file.filename,
                        "status": "error",
                        "message": "Arquivo vazio",
                        "type": None
                    })
                    continue
                
                # Ler conteúdo
                content = await file.read()
                content_str = content.decode('utf-8')
                
                # Detectar tipo
                detected_type = detect_file_type(content_str, file.filename or "unknown")
                
                # Se não detectou ou confiança baixa, usar LLM para validar
                if not detected_type:
                    validation = await validate_file_with_llm(content_str, file.filename or "unknown", detected_type)
                    detected_type = validation.get("detected_type")
                    llm_info = validation.get("llm_analysis", {})
                    
                    if not detected_type:
                        results.append({
                            "filename": file.filename,
                            "status": "error",
                            "message": "Arquivo não pôde ser classificado. Verifique o conteúdo.",
                            "type": None,
                            "llm_suggestion": llm_info
                        })
                        continue
                
                # Salvar arquivo
                save_success = await save_yaml_file(content_str, detected_type, file.filename or "documento.yaml")
                
                if not save_success:
                    results.append({
                        "filename": file.filename,
                        "status": "error",
                        "message": "Erro ao salvar arquivo no servidor",
                        "type": detected_type
                    })
                    continue
                
                # Indexar arquivo imediatamente após salvar
                try:
                    # Preparar documento para indexação
                    file_base_name = os.path.splitext(file.filename or "documento")[0]
                    
                    # Procesar arquivo baseado no tipo
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
                                
                                success = chromadb_client.add_document(
                                    text=table_text,
                                    metadata={
                                        "type": "table",
                                        "table_name": table_name,
                                        "source_file": file.filename,
                                        "source": "user_upload"
                                    },
                                    id=doc_id,
                                    collection_name=target_collection
                                )
                                
                                doc_count = 1 if success else 0
                                
                                # 2. Indexar colunas importantes (estrutura nova dos docs_cg)
                                colunas_importantes = data.get("colunas_importantes", [])
                                if isinstance(colunas_importantes, list):
                                    for col in colunas_importantes:
                                        if isinstance(col, dict) and col.get("nome"):
                                            col_name = col.get("nome")
                                            col_text = f"Coluna {col_name}: {col.get('descricao', '')} | Tipo: {col.get('tipo', 'unknown')} | Nulo: {col.get('nulo', 'N/A')}"
                                            col_doc_id = f"field_{file_base_name}_{col_name}"
                                            
                                            col_success = chromadb_client.add_document(
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
                                            if col_success:
                                                doc_count += 1
                                
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
                                                
                                                field_success = chromadb_client.add_document(
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
                                                if field_success:
                                                    doc_count += 1
                        except Exception as index_error:
                            import traceback
                            traceback.print_exc()
                    else:
                        # Para outros tipos, fazer indexação genérica
                        doc_id = f"{detected_type}_{file_base_name}"
                        chromadb_client.add_document(
                            text=content_str[:1000],
                            metadata={
                                "type": detected_type,
                                "source_file": file.filename,
                                "source": "user_upload"
                            },
                            id=doc_id,
                            collection_name=target_collection
                        )
                        
                except Exception as index_error:
                    import traceback
                    traceback.print_exc()
                
                # Se include_metadata, adicionar documento de metadata
                metadata_docs = 0
                if include_meta:
                    # Criar documento de metadata
                    try:
                        metadata_doc = {
                            'id': f"metadata_{detected_type}_{os.path.splitext(file.filename or 'doc')[0].lower()}",
                            'text': f"Documentação: {file.filename}\n"
                                   f"Tipo: {detected_type}\n"
                                   f"Estrutura: {content_str[:500]}...",
                            'metadata': {
                                'type': 'documentation_structure',
                                'category': detected_type,
                                'source_file': file.filename,
                                'source': 'file_structure'
                            }
                        }
                        # Aqui seria adicionado ao ChromaDB
                        metadata_docs = 1
                    except Exception as e:
                        pass
                
                results.append({
                    "filename": file.filename,
                    "status": "success",
                    "message": f"Arquivo '{file.filename}' importado com sucesso como {detected_type}",
                    "type": detected_type,
                    "size": file.size,
                    "metadata_docs": metadata_docs
                })
                
            except Exception as e:
                import traceback
                traceback.print_exc()
                results.append({
                    "filename": file.filename,
                    "status": "error",
                    "message": str(e),
                    "type": None
                })
        
        # Resumo
        success_count = len([r for r in results if r['status'] == 'success'])
        error_count = len([r for r in results if r['status'] == 'error'])
        
        return {
            "status": "success" if success_count > 0 else "error",
            "total_files": len(files),
            "success_count": success_count,
            "error_count": error_count,
            "results": results,
            "collection": target_collection,
            "include_metadata": include_meta
        }
        
    except HTTPException:
        raise
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Erro ao processar batch: {str(e)}")


@app.post("/api/vectordb/reconnect")
async def reconnect_chromadb():
    """
    Força reconexão ao ChromaDB
    """
    try:
        if chromadb_client.connect():
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
    """
    Endpoint de chat de dúvidas com streaming de resposta
    Retorna Server-Sent Events (SSE) com conteúdo sendo gerado em tempo real
    
    Query params opcionais:
    - collection_name: coleção ChromaDB para contexto
    """
    try:
        # Usar collection_name do query param ou session_id do request body como fallback
        effective_collection_name = collection_name or request.session_id or ""
        
        # Definir prompt para help
        system_prompt = SYSTEM_PROMPTS.get("help", "Responda baseado no contexto fornecido.")
        
        # Gerar resposta com streaming
        async def generate():
            async for chunk in generate_specialized_response_stream(
                message=request.message,
                system_prompt=system_prompt,
                context=request.context,
                use_chromadb=True,
                collection_name=effective_collection_name if effective_collection_name else None
            ):
                yield chunk
        
        return StreamingResponse(generate(), media_type="text/event-stream")
    except HTTPException:
        raise
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Erro ao processar chat help stream: {str(e)}")


@app.post("/api/chat/aluno/stream")
async def chat_aluno_stream_endpoint(request: SpecializedChatRequest, collection_name: str = ""):
    """
    Endpoint de chat aluno com streaming de resposta
    Retorna Server-Sent Events (SSE) com conteúdo sendo gerado em tempo real
    
    Query params opcionais:
    - collection_name: coleção ChromaDB para contexto
    """
    try:
        # Usar collection_name do query param ou session_id do request body como fallback
        effective_collection_name = collection_name or request.session_id or ""
        
        # Definir prompt para aluno
        system_prompt = SYSTEM_PROMPTS.get("aluno", "Você é um assistente de aprendizado.")
        
        # Gerar resposta com streaming
        async def generate():
            async for chunk in generate_specialized_response_stream(
                message=request.message,
                system_prompt=system_prompt,
                context=request.context,
                use_chromadb=True,
                collection_name=effective_collection_name if effective_collection_name else None
            ):
                yield chunk
        
        return StreamingResponse(generate(), media_type="text/event-stream")
    except HTTPException:
        raise
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Erro ao processar chat aluno stream: {str(e)}")




@app.post("/api/chat/sql/stream")
async def chat_sql_stream_endpoint(request: SpecializedChatRequest, collection_name: str = ""):
    """
    Endpoint de chat SQL com streaming de resposta
    Retorna Server-Sent Events (SSE) com conteúdo sendo gerado em tempo real
    
    Query params opcionais:
    - collection_name: coleção ChromaDB para contexto
    """
    try:
        # Usar collection_name do query param ou session_id do request body como fallback
        effective_collection_name = collection_name or request.session_id or ""
        
        # Definir prompt para SQL
        system_prompt = SYSTEM_PROMPTS.get("sql", "Você é um especialista em SQL.")
        
        # Gerar resposta com streaming
        async def generate():
            async for chunk in generate_specialized_response_stream(
                message=request.message,
                system_prompt=system_prompt,
                context=request.context,
                use_chromadb=True,
                collection_name=effective_collection_name if effective_collection_name else None
            ):
                yield chunk
        
        return StreamingResponse(generate(), media_type="text/event-stream")
    except HTTPException:
        raise
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Erro ao processar chat SQL stream: {str(e)}")


@app.post("/api/chat/general/stream")
async def chat_general_stream_endpoint(request: SpecializedChatRequest, collection_name: str = ""):
    """
    Endpoint de chat geral com streaming de resposta
    Retorna Server-Sent Events (SSE) com conteúdo sendo gerado em tempo real
    
    Query params opcionais:
    - collection_name: coleção ChromaDB para contexto (não usado em chat geral)
    """
    try:
        # Chat geral não usa ChromaDB, então collection_name é ignorado
        
        # Definir prompt para chat geral
        system_prompt = SYSTEM_PROMPTS.get("general", "Você é um assistente amigável e prestativo.")
        
        # Gerar resposta com streaming
        async def generate():
            async for chunk in generate_specialized_response_stream(
                message=request.message,
                system_prompt=system_prompt,
                context=request.context,
                use_chromadb=False,  # Chat geral não usa ChromaDB
                collection_name=None
            ):
                yield chunk
        
        return StreamingResponse(generate(), media_type="text/event-stream")
    except HTTPException:
        raise
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Erro ao processar chat geral stream: {str(e)}")


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
        
        # Definir a coleção
        chromadb_client.set_collection(collection_name)
        
        # Executar a query
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


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)

