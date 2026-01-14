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
LMSTUDIO_BASE_URL = f"{LMSTUDIO_URL}/v1" if os.getenv("GENAI_PROVIDER", "lmstudio") == "lmstudio" else ""

# Inicializar GenAI
try:
    genai = GenAIFactory.create()
    print(f"✅ GenAI iniciado: provider={os.getenv('GENAI_PROVIDER', 'lmstudio')}")
except Exception as e:
    print(f"❌ Erro ao inicializar GenAI: {e}")
    genai = None

# Inicializar Embeddings
try:
    embeddings = EmbeddingsFactory.create()
    print(f"✅ Embeddings iniciado: provider={os.getenv('EMBEDDINGS_PROVIDER', 'lmstudio')}")
except Exception as e:
    print(f"❌ Erro ao inicializar Embeddings: {e}")
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
    n_results: Optional[int] = 3
    context: Optional[str] = "all"

class VectorDBQueryResponse(BaseModel):
    question: str
    results: List[Dict[str, Any]]
    total_results: int
    processing_time: float

class DatabaseUpdateRequest(BaseModel):
    database_structure: Dict[str, Any]

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
            target_collection = collection_name if collection_name else "sistema_comercial"
            print(f"🎯 API: Usando coleção '{target_collection}' para chat")
            
            # Define a coleção (set_collection já cria se não existir)
            if not chromadb_client.set_collection(target_collection):
                print(f"Erro ao definir coleção '{target_collection}'")
            
            # Buscar contexto relevante
            results = chromadb_client.query(message, n_results=5)
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
    use_chromadb: bool = True
) -> Dict[str, Any]:
    """Gera resposta especializada usando LMStudio com contexto do ChromaDB"""
    try:
        chromadb_context = ""
        
        # Buscar contexto no ChromaDB se solicitado
        if use_chromadb and chromadb_client:
            try:
                # Cria coleção se não existir
                if not chromadb_client.create_collection():
                    print("Aviso: Não foi possível criar coleção")
                
                # Buscar contexto relevante
                results = chromadb_client.query(message, n_results=5)
                if results:
                    context_parts = []
                    for i, result in enumerate(results, 1):
                        context_parts.append(
                            f"[{i}] {result['type'].upper()}: {result['content']}"
                        )
                        if result['metadata'].get('table_name'):
                            context_parts.append(f"    Tabela: {result['metadata']['table_name']}")
                        if result['metadata'].get('source'):
                            context_parts.append(f"    Fonte: {result['metadata']['source']}")
                        context_parts.append(f"    Relevância: {result['similarity']:.3f}\n")
                    
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

