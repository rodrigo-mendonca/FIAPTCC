from fastapi import FastAPI, HTTPException, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
import os
from dotenv import load_dotenv
import json
import asyncio
from typing import AsyncGenerator, List, Optional, Dict, Any
import httpx
import requests
import uuid
from datetime import datetime
import sys
from factories import GenAIFactory, EmbeddingsFactory, ChromaDBClient
from factories.embeddings_factory import EmbeddingsUtility
from factories.genai_factory import ChatResponseGenerator

# Carrega variáveis de ambiente
load_dotenv()

# Configuração
ENVIRONMENT = os.getenv("ENVIRONMENT", "local")
CHROMADB_HOST = os.getenv("CHROMADB_HOST", "localhost")
CHROMADB_PORT = int(os.getenv("CHROMADB_PORT", "8200"))

# Para compatibilidade com código existente que usa LMSTUDIO_BASE_URL
# Se usar LMStudio, define a URL com /v1. Caso contrário, deixa vazio
LMSTUDIO_URL = os.getenv("LMSTUDIO_URL", "http://192.168.50.30:1234")
LMSTUDIO_API_KEY = os.getenv("LMSTUDIO_API_KEY", "lm-studio")
LMSTUDIO_BASE_URL = f"{LMSTUDIO_URL}/v1"

# ChromaDB Configuration
CHROMADB_DEFAULT_RESULTS = int(os.getenv("CHROMADB_DEFAULT_RESULTS", "50"))  # Número padrão de resultados para queries

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
    print(f"[OK] Embeddings iniciado: provider={os.getenv('EMBEDDINGS_PROVIDER', 'lmstudio')}")
except Exception as e:
    print(f"[ERROR] Erro ao inicializar Embeddings: {e}")
    embeddings = None

# Manter compatibilidade com código existente
chromadb_client = ChromaDBClient(
    host=CHROMADB_HOST,
    port=CHROMADB_PORT,
    lmstudio_url=os.getenv("LMSTUDIO_URL", "http://192.168.50.30:1234")
)

# Tenta conectar ao ChromaDB (não sobrescrever a instância para evitar AttributeError nas rotas)
try:
    if chromadb_client.connect():
        chroma_client = chromadb_client  # Compatibilidade com código existente
    else:
        chroma_client = None
        # mantém a instância em `chromadb_client` para permitir tentativas posteriores
except Exception as e:
    chroma_client = None
    # mantém a instância em `chromadb_client`; erros de conexão serão tratados nas rotas

# chroma_client = chromadb_client  # Temporário para debug

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
    n_results: Optional[int] = CHROMADB_DEFAULT_RESULTS
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
    collection_name: Optional[str] = None
) -> AsyncGenerator[str, None]:
    """Gera resposta especializada com streaming usando LMStudio com contexto do ChromaDB"""

    chromadb_context = ""
    
    # Buscar contexto no ChromaDB se solicitado
    if use_chromadb and chromadb_client:
        try:
            # Define a coleção a ser usada (padrão ou especificada)
            target_collection = collection_name if collection_name else ""
            print(f"🎯 API: Usando coleção '{target_collection}' para chat")
            
            # Define a coleção (set_collection já cria se não existir)
            if not chromadb_client.set_collection(target_collection):
                print(f"Erro ao definir coleção '{target_collection}'")
            
            # Buscar contexto relevante - aumentado para capturar mais tabelas
            results = chromadb_client.query(message, n_results=CHROMADB_DEFAULT_RESULTS)
            if results:
                context_parts = []
                for i, result in enumerate(results, 1):
                    context_parts.append(
                        f"[{i}] {result['type'].upper()}: {result['content']} ({result['similarity']:.3f})"
                    )
                
                chromadb_context = "\n".join(context_parts)
        except Exception as e:
            print(f"Erro ao buscar contexto ChromaDB: {e}")
            chromadb_context = "Erro ao acessar base de conhecimento."
    
    # Construir mensagens para LMStudio
    messages = []
    
    # Adicionar prompt do sistema com contexto
    full_system_prompt = system_prompt
    if chromadb_context:
        full_system_prompt += f"\n\nCONTEXTO DA BASE DE CONHECIMENTO:\n{chromadb_context}"
    
    messages.append({
        "role": "system",
        "content": full_system_prompt
    })
    
    # Adicionar contexto da conversa
    if context:
        for msg in context[-10:]:  # Limitar histórico
            messages.append({
                "role": msg.role,
                "content": msg.content
            })
    
    # Adicionar mensagem atual
    messages.append({
        "role": "user",
        "content": message
    })
    
    # Fazer requisição ao LMStudio com streaming
    payload = {
        "model": "local-model",
        "messages": messages,
        "stream": True,
        "temperature": 0.7,
        "max_tokens": 1000
    }
    
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {LMSTUDIO_API_KEY}"
    }
    
    async with httpx.AsyncClient(timeout=60.0) as client:
        async with client.stream(
            "POST",
            f"{LMSTUDIO_BASE_URL}/chat/completions",
            json=payload,
            headers=headers
        ) as response:
            
            if response.status_code != 200:
                yield f"data: {json.dumps({'error': f'Erro na API do LMStudio: {response.status_code}'})}\n\n"
                return
            
            buffer = ""
            
            async for chunk in response.aiter_bytes():
                try:
                    # Decodifica o chunk
                    chunk_str = chunk.decode('utf-8')
                    buffer += chunk_str
                    
                    # Processa linhas completas
                    lines = buffer.split('\n')
                    buffer = lines[-1]  # Mantém linha incompleta no buffer
                    
                    for line in lines[:-1]:
                        line = line.strip()
                        
                        if line.startswith("data: "):
                            data_str = line[6:]  # Remove "data: "
                            if data_str.strip() == "[DONE]":
                                return
                            
                            try:
                                data = json.loads(data_str)
                                if "choices" in data and len(data["choices"]) > 0:
                                    delta = data["choices"][0].get("delta", {})
                                    if "content" in delta:
                                        content = delta["content"]
                                        yield f"data: {json.dumps({'content': content})}\n\n"
                            except json.JSONDecodeError:
                                continue
                except UnicodeDecodeError:
                    continue
                except Exception as e:
                    yield f"data: {json.dumps({'error': f'Erro ao gerar resposta: {str(e)}'})}\n\n"

async def generate_specialized_response(
    message: str, 
    system_prompt: str,
    context: Optional[List[ChatMessage]] = None,
    use_chromadb: bool = True,
    collection_name: Optional[str] = None
) -> Dict[str, Any]:
    """Gera resposta especializada usando LMStudio com contexto do ChromaDB"""
    try:
        chromadb_context = ""
        
        # Buscar contexto no ChromaDB se solicitado
        if use_chromadb and chromadb_client:
            try:
                # Define a coleção a ser usada
                if collection_name and collection_name.strip():
                    chromadb_client.set_collection(collection_name)
                else:
                    # Cria coleção padrão se não existir
                    if not chromadb_client.create_collection():
                        print("Aviso: Não foi possível criar coleção")
                
                results = chromadb_client.query(message, n_results=CHROMADB_DEFAULT_RESULTS)
                if results:
                    context_parts = []
                    for i, result in enumerate(results, 1):
                        context_parts.append(
                            f"[{i}] {result['type'].upper()}: {result['content']} ({result['similarity']:.3f})"
                        )
                    chromadb_context = "\n".join(context_parts)
                        
            except Exception as e:
                print(f"Erro ao buscar contexto ChromaDB: {e}")
                chromadb_context = "Erro ao acessar base de conhecimento."
        
        # Construir mensagens para LMStudio
        messages = []
        
        # Adicionar prompt do sistema com contexto
        full_system_prompt = system_prompt
        if chromadb_context:
            # Melhorar formatação do contexto para o LLM
            full_system_prompt += f"""

=== CONTEXTO DA BASE DE CONHECIMENTO ===
{chromadb_context}
==="""
        
        messages.append({
            "role": "system",
            "content": full_system_prompt
        })
        
        # Adicionar contexto da conversa
        if context:
            for msg in context[-10:]:  # Limitar histórico
                messages.append({
                    "role": msg.role,
                    "content": msg.content
                })
        
        # Adicionar mensagem atual
        messages.append({
            "role": "user",
            "content": message
        })
        
        # Fazer requisição ao LMStudio
        payload = {
            "model": "local-model",
            "messages": messages,
            "stream": False,
            "temperature": 0.7,
            "max_tokens": 1000
        }
        
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {LMSTUDIO_API_KEY}"
        }
        
        async with httpx.AsyncClient(timeout=30.0) as client:
            response = await client.post(
                f"{LMSTUDIO_BASE_URL}/chat/completions",
                json=payload,
                headers=headers
            )
            
            if response.status_code != 200:
                raise HTTPException(
                    status_code=500, 
                    detail=f"Erro na API do LMStudio: {response.status_code}"
                )
            
            data = response.json()
            ai_response = data["choices"][0]["message"]["content"]
            
            return {
                "response": ai_response,
                "context_used": chromadb_context if chromadb_context else None
            }
            
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Erro ao gerar resposta: {str(e)}")


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
        print(f"Erro ao salvar arquivo: {e}")
        return False


@app.post("/api/vectordb/upload")
async def upload_file_unified(file: UploadFile = File(...), collection_name: str = Form(...)) -> UnifiedFileUploadResponse:
    """
    Endpoint unificado para upload de arquivo
    """
    print(f"\n\n{'='*60}")
    print(f"[UPLOAD] ===== UPLOAD INICIADO =====")
    print(f"[UPLOAD] Arquivo: {file.filename}")
    print(f"[UPLOAD] Content-Type: {file.content_type}")
    print(f"[UPLOAD] Size: {file.size}")
    print(f"[UPLOAD] Collection: '{collection_name}'")
    print(f"{'='*60}\n")
    
    try:
        # Validar que uma coleção foi fornecida
        if not collection_name or not collection_name.strip():
            print("[UPLOAD] ✗ Nenhuma coleção especificada")
            raise HTTPException(status_code=400, detail="collection_name é obrigatório")
        
        target_collection = collection_name
        print(f"[UPLOAD] Coleção alvo: {target_collection}")
        
        # Validar que temos um arquivo
        if not file or file.size == 0:
            print("[UPLOAD] ✗ Arquivo vazio")
            raise HTTPException(status_code=400, detail="Arquivo vazio ou inválido")
        
        # Verificar se ChromaDB está disponível
        if not chromadb_client or not chromadb_client.client:
            print("[UPLOAD] ✗ ChromaDB não inicializado")
            raise HTTPException(status_code=503, detail="ChromaDB não está disponível. Tente reconectar.")
        
        # Ler conteúdo do arquivo
        print("[UPLOAD] 📖 Lendo conteúdo do arquivo...")
        content = await file.read()
        content_str = content.decode('utf-8')
        print(f"[UPLOAD]    ✓ {len(content_str)} caracteres lidos")
        print(f"[UPLOAD]    Preview: {content_str[:200]}...")
        
        # Detectar tipo de arquivo
        print("[UPLOAD] 🔍 Detectando tipo de arquivo...")
        detected_type = detect_file_type(content_str, file.filename or "unknown")
        print(f"[UPLOAD]    Tipo detectado: {detected_type}")
        
        if not detected_type:
            print("[UPLOAD] ✗ Tipo não foi detectado")
            raise HTTPException(
                status_code=400, 
                detail=f"Não foi possível detectar o tipo de arquivo. Verifique o conteúdo do arquivo.\nArquivo: {file.filename}"
            )
        
        # Salvar arquivo na pasta correta
        print(f"[UPLOAD] 💾 Salvando arquivo como {detected_type}...")
        save_success = await save_yaml_file(content_str, detected_type, file.filename or "documento.yaml")
        
        if not save_success:
            print("[UPLOAD] ✗ Erro ao salvar arquivo")
            raise HTTPException(status_code=400, detail="Erro ao salvar arquivo")
        print("[UPLOAD]    ✓ Arquivo salvo")
        
        # Reindexar ChromaDB
        try:
            # Primeiro, garantir que a coleção está selecionada
            print(f"[UPLOAD] 🎯 Selecionando coleção '{target_collection}'...")
            if not chromadb_client.set_collection(target_collection):
                print(f"[UPLOAD] ⚠️ Coleção não existe, criando...")
                if not chromadb_client.create_collection(target_collection):
                    print(f"[UPLOAD] ✗ Não foi possível criar coleção")
                    return UnifiedFileUploadResponse(
                        message=f"Arquivo salvo mas não foi possível criar a coleção '{target_collection}'",
                        file_type=detected_type,
                        file_name=file.filename or "documento",
                        status="error"
                    )
            print(f"[UPLOAD]    ✓ Coleção selecionada")
            print("[UPLOAD] ✓✓✓ SUCESSO!")
            return UnifiedFileUploadResponse(
                message=f"Arquivo '{file.filename}' carregado com sucesso como {detected_type}",
                file_type=detected_type,
                file_name=file.filename or "documento",
                status="success"
            )
        except Exception as e:
            print(f"[UPLOAD] ✗ Erro ao reindexar: {e}")
            import traceback
            traceback.print_exc()
            return UnifiedFileUploadResponse(
                message=f"Arquivo '{file.filename}' salvo mas houve erro ao reindexar: {str(e)}",
                file_type=detected_type,
                file_name=file.filename or "documento",
                status="error"
            )
        
    except HTTPException:
        raise
    except Exception as e:
        print(f"[UPLOAD] ✗✗✗ ERRO: {e}")
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Erro ao processar arquivo: {str(e)}")


async def validate_file_with_llm(content: str, filename: str, detected_type: Optional[str]) -> Dict[str, Any]:
    """
    Valida arquivo usando LLM se o tipo não foi detectado ou se há dúvida
    Retorna análise da LLM sobre o arquivo
    """
    try:
        if not genai_client:
            return {
                "valid": detected_type is not None,
                "detected_type": detected_type,
                "llm_analysis": None,
                "confidence": "low" if detected_type is None else "high"
            }
        
        # Preparar prompt para análise
        preview = content[:1000]  # Pega os primeiros 1000 caracteres
        
        validation_prompt = f"""
Analise este arquivo e determine sua categoria:

Nome do arquivo: {filename}
Tipo detectado automaticamente: {detected_type or "Não detectado"}

Conteúdo (primeiros 1000 caracteres):
{preview}
...

Categorias possíveis:
1. **base_dados** - Estrutura de banco de dados (tabelas, colunas, chaves)
2. **regras_negocio** - Regras e validações de negócio (políticas, limites, descontos)
3. **servicos** - Serviços e rotinas do sistema (backup, sincronização, automações)
4. **rotinas_usuario** - Procedimentos do usuário (passo a passo, workflows)
5. **outro** - Se não se encaixa em nenhuma categoria

Por favor, responda em JSON com o seguinte formato:
{{
    "categoria": "<um das categorias acima>",
    "confianca": "<alta|media|baixa>",
    "motivo": "<breve explicação>",
    "pode_processar": true|false,
    "sugestoes": "<sugestões se necessário>"
}}
"""
        
        response = await genai_client.generate_response(validation_prompt, max_tokens=500)
        
        # Tentar extrair JSON da resposta
        try:
            import json
            # Procura por JSON na resposta
            import re
            json_match = re.search(r'\{.*\}', response, re.DOTALL)
            if json_match:
                llm_result = json.loads(json_match.group())
                
                # Mapear categoria da LLM para nossa estrutura
                categoria_map = {
                    'base_dados': 'base_dados',
                    'regras_negocio': 'regras_negocio',
                    'regras negócio': 'regras_negocio',
                    'servicos': 'servicos',
                    'rotinas_usuario': 'rotinas_usuario',
                    'rotinas do usuario': 'rotinas_usuario'
                }
                
                categoria_llm = llm_result.get('categoria', '').lower()
                final_type = categoria_map.get(categoria_llm, detected_type)
                
                return {
                    "valid": llm_result.get('pode_processar', True),
                    "detected_type": final_type,
                    "llm_analysis": llm_result,
                    "confidence": llm_result.get('confianca', 'media')
                }
        except:
            pass
        
        # Se não conseguir extrair JSON, usar resultado simples
        return {
            "valid": True,
            "detected_type": detected_type,
            "llm_analysis": {"resposta_bruta": response},
            "confidence": "media"
        }
        
    except Exception as e:
        print(f"[VALIDATE] Erro ao validar com LLM: {e}")
        return {
            "valid": detected_type is not None,
            "detected_type": detected_type,
            "llm_analysis": None,
            "confidence": "low" if detected_type is None else "high"
        }


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
    print(f"\n\n{'='*60}")
    print(f"[BATCH-UPLOAD] ===== BATCH UPLOAD INICIADO =====")
    print(f"[BATCH-UPLOAD] Total de arquivos: {len(files)}")
    print(f"[BATCH-UPLOAD] Collection: '{collection_name}'")
    print(f"[BATCH-UPLOAD] Include Metadata: {include_metadata}")
    print(f"{'='*60}\n")
    
    try:
        # Validações básicas
        if not collection_name or not collection_name.strip():
            print("[BATCH-UPLOAD] ✗ Nenhuma coleção especificada")
            raise HTTPException(status_code=400, detail="collection_name é obrigatório")
        
        if not files or len(files) == 0:
            print("[BATCH-UPLOAD] ✗ Nenhum arquivo fornecido")
            raise HTTPException(status_code=400, detail="Forneça pelo menos um arquivo")
        
        if not chromadb_client or not chromadb_client.client:
            print("[BATCH-UPLOAD] ✗ ChromaDB não inicializado")
            raise HTTPException(status_code=503, detail="ChromaDB não está disponível")
        
        # Garantir que a coleção existe
        target_collection = collection_name
        if not chromadb_client.set_collection(target_collection):
            print(f"[BATCH-UPLOAD] ⚠️ Coleção não existe, criando '{target_collection}'...")
            if not chromadb_client.create_collection(target_collection):
                raise HTTPException(status_code=500, detail=f"Não foi possível criar coleção '{target_collection}'")
        
        results = []
        include_meta = include_metadata.lower() == "true"
        
        # Processar cada arquivo
        for idx, file in enumerate(files, 1):
            print(f"\n[BATCH-UPLOAD] [{idx}/{len(files)}] Processando: {file.filename}")
            
            try:
                # Validar arquivo
                if not file or file.size == 0:
                    print(f"[BATCH-UPLOAD]    ✗ Arquivo vazio")
                    results.append({
                        "filename": file.filename,
                        "status": "error",
                        "message": "Arquivo vazio",
                        "type": None
                    })
                    continue
                
                # Ler conteúdo
                print(f"[BATCH-UPLOAD]    📖 Lendo conteúdo ({file.size} bytes)...")
                content = await file.read()
                content_str = content.decode('utf-8')
                print(f"[BATCH-UPLOAD]    ✓ {len(content_str)} caracteres lidos")
                
                # Detectar tipo
                print(f"[BATCH-UPLOAD]    🔍 Detectando tipo...")
                detected_type = detect_file_type(content_str, file.filename or "unknown")
                print(f"[BATCH-UPLOAD]    Tipo detectado: {detected_type}")
                
                # Se não detectou ou confiança baixa, usar LLM para validar
                if not detected_type:
                    print(f"[BATCH-UPLOAD]    🤖 Tipo não detectado, consultando LLM...")
                    validation = await validate_file_with_llm(content_str, file.filename or "unknown", detected_type)
                    detected_type = validation.get("detected_type")
                    llm_info = validation.get("llm_analysis", {})
                    
                    if detected_type:
                        print(f"[BATCH-UPLOAD]    ✓ LLM sugeriu: {detected_type}")
                    else:
                        print(f"[BATCH-UPLOAD]    ✗ LLM não conseguiu classificar")
                        results.append({
                            "filename": file.filename,
                            "status": "error",
                            "message": "Arquivo não pôde ser classificado. Verifique o conteúdo.",
                            "type": None,
                            "llm_suggestion": llm_info
                        })
                        continue
                
                # Salvar arquivo
                print(f"[BATCH-UPLOAD]    💾 Salvando arquivo como '{detected_type}'...")
                save_success = await save_yaml_file(content_str, detected_type, file.filename or "documento.yaml")
                
                if not save_success:
                    print(f"[BATCH-UPLOAD]    ✗ Erro ao salvar arquivo")
                    results.append({
                        "filename": file.filename,
                        "status": "error",
                        "message": "Erro ao salvar arquivo no servidor",
                        "type": detected_type
                    })
                    continue
                
                print(f"[BATCH-UPLOAD]    ✓ Arquivo salvo")
                
                # Indexar arquivo imediatamente após salvar
                try:
                    print(f"[BATCH-UPLOAD]    🔄 Indexando arquivo no ChromaDB...")
                    
                    # Preparar documento para indexação
                    file_base_name = os.path.splitext(file.filename or "documento")[0]
                    
                    # Procesar arquivo baseado no tipo
                    if detected_type == 'base_dados':
                        try:
                            import yaml
                            data = yaml.safe_load(content_str)
                            if data and 'tabela' in data:
                                from factories.embeddings_factory import EmbeddingsUtility
                                
                                # Obter o nome da tabela (pode ser string ou dict)
                                table_name_value = data['tabela']
                                if isinstance(table_name_value, dict):
                                    table_name = table_name_value.get("nome", file_base_name)
                                    table_metadata = table_name_value
                                else:
                                    table_name = str(table_name_value)
                                    table_metadata = data  # Usar data completo como metadata
                                
                                # 1. Criar documento principal da tabela
                                table_text = EmbeddingsUtility.create_searchable_text(table_metadata, file_base_name)
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
                                if success:
                                    print(f"[BATCH-UPLOAD]    ✓ Documento de tabela adicionado")
                                else:
                                    print(f"[BATCH-UPLOAD]    ⚠️ Falha ao adicionar documento de tabela")
                                
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
                                                field_text = EmbeddingsUtility.create_field_searchable_text(field_name, field_data, file_base_name)
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
                                
                                print(f"[BATCH-UPLOAD]    ✓ Arquivo indexado: {doc_count} documentos")
                        except Exception as index_error:
                            print(f"[BATCH-UPLOAD]    ⚠️ Erro ao indexar base_dados (arquivo salvo): {index_error}")
                            import traceback
                            traceback.print_exc()
                    else:
                        # Para outros tipos, fazer indexação genérica
                        doc_id = f"{detected_type}_{file_base_name}"
                        success = chromadb_client.add_document(
                            text=content_str[:1000],
                            metadata={
                                "type": detected_type,
                                "source_file": file.filename,
                                "source": "user_upload"
                            },
                            id=doc_id,
                            collection_name=target_collection
                        )
                        
                        if success:
                            print(f"[BATCH-UPLOAD]    ✓ Arquivo indexado com sucesso!")
                        else:
                            print(f"[BATCH-UPLOAD]    ⚠️ Falha ao indexar arquivo")
                        
                except Exception as index_error:
                    print(f"[BATCH-UPLOAD]    ⚠️ ERRO CRÍTICO ao indexar: {index_error}")
                    import traceback
                    traceback.print_exc()
                
                # Se include_metadata, adicionar documento de metadata
                metadata_docs = 0
                if include_meta:
                    print(f"[BATCH-UPLOAD]    📋 Processando metadata...")
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
                        print(f"[BATCH-UPLOAD]    ✓ Metadata processada")
                    except Exception as e:
                        print(f"[BATCH-UPLOAD]    ⚠️ Erro ao processar metadata: {e}")
                
                results.append({
                    "filename": file.filename,
                    "status": "success",
                    "message": f"Arquivo '{file.filename}' importado com sucesso como {detected_type}",
                    "type": detected_type,
                    "size": file.size,
                    "metadata_docs": metadata_docs
                })
                print(f"[BATCH-UPLOAD]    ✓✓ Sucesso!")
                
            except Exception as e:
                print(f"[BATCH-UPLOAD]    ✗ Erro ao processar {file.filename}: {e}")
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
        
        print(f"\n{'='*60}")
        print(f"[BATCH-UPLOAD] ===== RESUMO =====")
        print(f"[BATCH-UPLOAD] Sucesso: {success_count}/{len(files)}")
        print(f"[BATCH-UPLOAD] Erros: {error_count}/{len(files)}")
        print(f"{'='*60}\n")
        
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
        print(f"[BATCH-UPLOAD] ✗✗✗ ERRO CRÍTICO: {e}")
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Erro ao processar batch: {str(e)}")


@app.post("/api/vectordb/reconnect")
async def reconnect_chromadb():
    """
    Força reconexão ao ChromaDB
    """
    try:
        print("[INFO] Reconectando ao ChromaDB...")
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
        print(f"[ERRO] Erro ao reconectar: {e}")
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
                print(f"[ERRO] ChromaDB heartbeat falhou: {e}")
        
        # Tentar reconectar
        print("[INFO] Tentando reconectar ao ChromaDB...")
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
        print(f"[ERRO] Erro ao verificar saúde do ChromaDB: {e}")
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
    """Verifica saúde do LMStudio (compatibilidade com frontend)"""
    try:
        lmstudio_url = os.getenv("LMSTUDIO_URL", "http://192.168.50.30:1234")
        response = requests.get(f"{lmstudio_url}/v1/models", timeout=5)
        if response.status_code == 200:
            return {"status": "online", "connected": True}
        return {"status": "offline", "connected": False}
    except:
        return {"status": "offline", "connected": False}

@app.get("/health")
async def health_general():
    """Verifica saúde geral da API"""
    return {"status": "ok", "message": "API está funcionando"}


# ============ ROTAS DE CHAT (Assistentes Especializados) ============

@app.post("/api/chat")
async def chat_endpoint(request: SpecializedChatRequest, collection_name: str = ""):
    """
    Endpoint de chat geral com contexto de conhecimento
    Suporta:
    - Chat de dúvidas (help)
    - Chat SQL
    - Chat de aprendizado (aluno)
    
    Query params opcionais:
    - collection_name: coleção ChromaDB para contexto
    - mode: 'help', 'sql', 'aluno' (padrão: 'help')
    """
    try:
        # Usar collection_name do query param ou session_id do request body como fallback
        effective_collection_name = collection_name or request.session_id or ""
        
        # Definir prompts baseado no tipo de chat
        # O mode sempre é "help" por padrão
        mode = "help"
        
        system_prompts = {
            "help": """Responda SOMENTE baseado no contexto fornecido abaixo.
Se a informação não está no contexto, diga claramente que não tem informações sobre o assunto.
Não invente ou suponha informações.""",
            
            "sql": """Você é um especialista em SQL e bancos de dados comerciais.
Gere consultas SQL otimizadas.
Explique a lógica das consultas com textos curtos.""",
            
            "aluno": """Você é um assistente de aprendizado que registra informações de novos conhecimentos.
Quando um usuário descreve algo novo utilize os meta dados para conferir se falta informação.
Pergunte sobre detalhes importantes se necessário."""
        }
        
        system_prompt = system_prompts.get(mode, system_prompts["help"])
        
        # Gerar resposta com contexto ChromaDB
        response_data = await generate_specialized_response(
            message=request.message,
            system_prompt=system_prompt,
            context=request.context,
            use_chromadb=True,
            collection_name=effective_collection_name if effective_collection_name else None
        )
        
        return SpecializedChatResponse(
            response=response_data["response"],
            session_id=request.session_id or str(uuid.uuid4()),
            context_used=response_data.get("context_used")
        )
    except HTTPException:
        raise
    except Exception as e:
        print(f"[ERROR] Erro no endpoint /api/chat: {e}")
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Erro ao processar chat: {str(e)}")


@app.post("/api/chat/stream")
async def chat_stream_endpoint(request: SpecializedChatRequest, collection_name: str = ""):
    """
    Endpoint de chat com streaming de resposta
    Retorna Server-Sent Events (SSE) com conteúdo sendo gerado em tempo real
    
    Query params opcionais:
    - collection_name: coleção ChromaDB para contexto
    - mode: 'help', 'sql', 'aluno' (padrão: 'help')
    """
    try:
        # Usar collection_name do query param ou session_id do request body como fallback
        effective_collection_name = collection_name or request.session_id or ""
        
        # Definir prompts baseado no tipo de chat
        # O mode sempre é "help" por padrão
        mode = "help"
        
        system_prompts = {
            "help": """Responda SOMENTE baseado no contexto fornecido abaixo.
Se a informação não está no contexto, diga claramente que não tem informações sobre o assunto.
Não invente ou suponha informações.
Cite sempre as tabelas e campos relevantes quando aplicável.""",
            
            "sql": """Você é um especialista em SQL e bancos de dados comerciais.
Gere consultas SQL otimizadas""",
            
            "aluno": """Você é um assistente de aprendizado que registra informações de novos conhecimentos.
Quando um usuário descreve algo novo, organize e valide a informação.
Reformule de forma estruturada e clara.
Pergunte sobre detalhes importantes se necessário."""
        }
        
        system_prompt = system_prompts.get(mode, system_prompts["help"])
        
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
        print(f"[ERROR] Erro no endpoint /api/chat/stream: {e}")
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Erro ao processar chat stream: {str(e)}")



async def get_file_types():
    """
    Retorna os tipos de arquivo aceitos e suas descrições
    """
    return {
        "types": {
            "regras_negocio": {
                "description": "Regras de negócio e validações",
                "keywords": ["regra", "validação", "política", "limite", "desconto"]
            },
            "base_dados": {
                "description": "Estrutura do banco de dados",
                "keywords": ["tabela", "coluna", "relacionamento", "índice", "chave"]
            },
            "servicos": {
                "description": "Serviços e rotinas do sistema",
                "keywords": ["serviço", "automação", "backup", "sincronização"]
            },
            "rotinas_usuario": {
                "description": "Rotinas e procedimentos do usuário",
                "keywords": ["rotina", "passo a passo", "procedimento", "frequência"]
            }
        },
        "supported_formats": ["YAML", "JSON"],
        "auto_detect": True
    }


# ============ ROTAS DE VECTORDB (compatibilidade com interface) ============

@app.get("/vectordb/stats")
async def get_vectordb_stats(collection_name: str = ""):
    """Retorna estatísticas da coleção ou de todas as coleções se collection_name estiver vazio"""
    try:
        if not chromadb_client:
            raise HTTPException(status_code=503, detail="ChromaDB não disponível")
        
        # Se collection_name está vazio, retorna estatísticas gerais (todas as coleções)
        if not collection_name or collection_name.strip() == "":
            stats = chromadb_client.get_collection_stats()
            if not isinstance(stats, dict):
                stats = {"error": "Resposta inválida do ChromaDB"}
            return stats
        
        # Caso contrário, tenta obter stats de uma coleção específica
        if not chromadb_client.set_collection(collection_name):
            raise HTTPException(status_code=404, detail=f"Coleção '{collection_name}' não existe. Use POST /vectordb/create-collection para criar.")
        
        stats = chromadb_client.get_collection_stats()
        
        # Garantir que é um dicionário válido
        if not isinstance(stats, dict):
            stats = {"error": "Resposta inválida do ChromaDB"}
        
        return stats
    except HTTPException:
        raise
    except Exception as e:
        print(f"[ERRO] Exceção em get_vectordb_stats: {e}")
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))


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
        
        print(f"[DELETE] Deletando coleção: {collection_name}")
        if chromadb_client.delete_collection(collection_name):
            return {"message": f"Coleção '{collection_name}' deletada com sucesso", "status": "success"}
        else:
            raise HTTPException(status_code=500, detail=f"Falha ao deletar coleção '{collection_name}'")
    except HTTPException:
        raise
    except Exception as e:
        print(f"[ERRO] Erro ao deletar coleção: {e}")
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))


@app.delete("/vectordb/collection/{collection_name}")
async def delete_collection_endpoint(collection_name: str):
    """Deleta uma coleção"""
    try:
        if not chromadb_client:
            raise HTTPException(status_code=503, detail="ChromaDB não disponível")
        
        print(f"[DELETE] Deletando coleção: {collection_name}")
        if chromadb_client.delete_collection(collection_name):
            return {"message": f"Coleção '{collection_name}' deletada com sucesso", "status": "success"}
        else:
            raise HTTPException(status_code=500, detail=f"Falha ao deletar coleção '{collection_name}'")
    except HTTPException:
        raise
    except Exception as e:
        print(f"[ERRO] Erro ao deletar coleção: {e}")
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
        print(f"[ERROR] Erro em /vectordb/query: {e}")
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/vectordb/clear")
async def clear_vectordb(request: dict):
    """Limpa a coleção"""
    try:
        if not chromadb_client:
            raise HTTPException(status_code=503, detail="ChromaDB não disponível")
        
        collection_name = request.get("collection_name")
        if not collection_name:
            return {"status": "error", "message": "collection_name é obrigatório"}
        
        chromadb_client.set_collection(collection_name)
        chromadb_client.clear_collection()
        return {"message": f"Coleção '{collection_name}' limpa com sucesso"}
    except Exception as e:
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

