from fastapi import FastAPI, HTTPException, UploadFile, File
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
import hashlib
import sys
from db.chromadb_client import ChromaDBClient

# Carrega variáveis de ambiente
load_dotenv()

# Configuração do LMStudio baseada no ambiente
ENVIRONMENT = os.getenv("ENVIRONMENT", "local")
if ENVIRONMENT == "docker":
    LMSTUDIO_BASE_URL = os.getenv("LMSTUDIO_URL", "http://lmchat:1234") + "/v1"
    CHROMADB_HOST = os.getenv("CHROMADB_HOST", "chromadb")
    CHROMADB_PORT = 8200
else:
    LMSTUDIO_BASE_URL = os.getenv("LMSTUDIO_BASE_URL", "http://localhost:8080/v1")
    CHROMADB_HOST = os.getenv("CHROMADB_HOST", "localhost")
    CHROMADB_PORT = 8200

LMSTUDIO_API_KEY = os.getenv("LMSTUDIO_API_KEY", "lm-studio")

# Configuração ChromaDB Client
chromadb_client = ChromaDBClient(
    host=CHROMADB_HOST,
    port=CHROMADB_PORT,
    lmstudio_url=os.getenv("LMSTUDIO_URL", "http://192.168.50.30:1234")  # URL do LMStudio para embeddings
)

# Tenta conectar ao ChromaDB
try:
    if chromadb_client.connect():
        chroma_client = chromadb_client  # Compatibilidade com código existente
    else:
        chroma_client = None
        chromadb_client = None
except Exception as e:
    chroma_client = None
    chromadb_client = None

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

def create_simple_embeddings(texts: List[str]) -> List[List[float]]:
    """Cria embeddings simples para textos em português usando análise de características linguísticas"""
    import re
    import unicodedata
    
    embeddings = []
    
    # Palavras-chave importantes em português para bases de dados
    keywords_pt = [
        'tabela', 'campo', 'chave', 'primaria', 'estrangeira', 'indice', 'relacao',
        'cliente', 'produto', 'pedido', 'venda', 'usuario', 'conta', 'saldo',
        'valor', 'data', 'codigo', 'nome', 'descricao', 'tipo', 'status',
        'ativo', 'inativo', 'criado', 'atualizado', 'deletado', 'numero',
        'endereco', 'telefone', 'email', 'cpf', 'cnpj', 'documento'
    ]
    
    for text in texts:
        # Normalizar texto para análise
        text_normalized = unicodedata.normalize('NFD', text.lower())
        text_normalized = ''.join(c for c in text_normalized if not unicodedata.combining(c))
        
        # Criar vetor de características
        features = []
        
        # 1. Características baseadas em palavras-chave (primeiras 32 dimensões)
        for keyword in keywords_pt:
            if keyword in text_normalized:
                features.append(1.0)
            else:
                features.append(0.0)
        
        # 2. Características de estrutura do texto (próximas 32 dimensões)
        features.extend([
            len(text) / 1000.0,  # Comprimento normalizado
            text.count('|') / 10.0,  # Separadores
            text.count(':') / 10.0,  # Dois pontos
            text.count('_') / 10.0,  # Underscores
            text.count(' ') / 100.0,  # Espaços
            text.upper().count('A') / 100.0,  # Frequência de A
            text.upper().count('E') / 100.0,  # Frequência de E
            text.upper().count('I') / 100.0,  # Frequência de I
            text.upper().count('O') / 100.0,  # Frequência de O
            text.upper().count('U') / 100.0,  # Frequência de U
            text.count('ã') / 10.0,  # Til
            text.count('ç') / 10.0,  # Cedilha
            text.count('á') / 10.0,  # Acentos agudos
            text.count('à') / 10.0,
            text.count('é') / 10.0,
            text.count('ê') / 10.0,
            text.count('í') / 10.0,
            text.count('ó') / 10.0,
            text.count('ô') / 10.0,
            text.count('ú') / 10.0,
            len(re.findall(r'\d+', text)) / 10.0,  # Números
            len(re.findall(r'[A-Z]{2,}', text)) / 10.0,  # Siglas
            text.count('TABELA') / 5.0,  # Padrões específicos
            text.count('CAMPO') / 5.0,
            text.count('DESCRIÇÃO') / 5.0,
            text.count('TIPO') / 5.0,
            text.count('RELACIONAMENTO') / 5.0,
            text.count('CONCEITO') / 5.0,
            text.count('NEGÓCIO') / 5.0,
            text.count('ÁREA') / 5.0,
            1.0 if 'decimal' in text_normalized else 0.0,
            1.0 if 'varchar' in text_normalized else 0.0
        ])
        
        # 3. Hash melhorado para as dimensões restantes
        hash_obj = hashlib.md5((text + "pt-br-semantic").encode('utf-8'))
        hash_hex = hash_obj.hexdigest()
        
        # Converter hash para features adicionais
        hash_features = []
        for i in range(0, min(len(hash_hex), 64), 2):
            hash_features.append(int(hash_hex[i:i+2], 16) / 255.0)
        
        features.extend(hash_features)
        
        # Preencher até 384 dimensões
        while len(features) < 384:
            # Repetir padrão mas com variação
            cycle_features = features[:min(32, 384-len(features))]
            # Adicionar pequena variação baseada na posição
            for j, f in enumerate(cycle_features):
                features.append(min(1.0, f + (len(features) % 10) * 0.01))
        
        embeddings.append(features[:384])
    
    return embeddings

def create_searchable_text(table_data: Dict[str, Any], table_name: str) -> str:
    """Cria texto otimizado para busca semântica em português"""
    text_parts = [f"TABELA: {table_name}"]
    
    # Adicionar descrição
    if "description" in table_data:
        desc = table_data['description']
        text_parts.append(f"DESCRIÇÃO: {desc}")
        # Adicionar sinônimos comuns em português
        if any(word in desc.lower() for word in ['cliente', 'consumidor']):
            text_parts.append("SINÔNIMOS: cliente consumidor usuário")
        if any(word in desc.lower() for word in ['produto', 'item', 'mercadoria']):
            text_parts.append("SINÔNIMOS: produto item mercadoria artigo")
        if any(word in desc.lower() for word in ['pedido', 'order', 'solicitação']):
            text_parts.append("SINÔNIMOS: pedido order solicitação requisição")
        if any(word in desc.lower() for word in ['venda', 'transação', 'compra']):
            text_parts.append("SINÔNIMOS: venda transação compra negociação")
    
    # Adicionar campos com tradução de tipos
    if "fields" in table_data:
        field_list = []
        type_translations = {
            'varchar': 'texto variável',
            'char': 'texto fixo',
            'text': 'texto longo',
            'integer': 'número inteiro',
            'decimal': 'número decimal',
            'float': 'número decimal',
            'date': 'data',
            'datetime': 'data e hora',
            'timestamp': 'carimbo temporal',
            'boolean': 'verdadeiro falso',
            'bit': 'binário'
        }
        
        for field_name, field_data in table_data["fields"].items():
            field_type = field_data.get("type", "unknown")
            field_type_pt = type_translations.get(field_type.lower(), field_type)
            field_desc = field_data.get("description", "")
            
            field_info = f"{field_name} (tipo: {field_type_pt})"
            if field_desc:
                field_info += f" - {field_desc}"
            
            # Adicionar contexto em português para campos importantes
            if any(word in field_name.lower() for word in ['id', 'codigo', 'key']):
                field_info += " (identificador único)"
            elif any(word in field_name.lower() for word in ['nome', 'name']):
                field_info += " (denominação)"
            elif any(word in field_name.lower() for word in ['valor', 'preco', 'saldo']):
                field_info += " (quantia monetária)"
            elif any(word in field_name.lower() for word in ['data', 'date']):
                field_info += " (informação temporal)"
            
            field_list.append(field_info)
        
        text_parts.append(f"CAMPOS: {'; '.join(field_list)}")
    
    # Adicionar relacionamentos com tradução
    if "relationships" in table_data:
        rel_list = []
        for rel in table_data["relationships"]:
            target = rel.get('target_table', '')
            source_field = rel.get('source_field', '')
            rel_desc = f"conecta com tabela {target}"
            if source_field:
                rel_desc += f" através do campo {source_field}"
            rel_list.append(rel_desc)
        if rel_list:
            text_parts.append(f"RELACIONAMENTOS: {'; '.join(rel_list)}")
    
    # Adicionar área de negócio com contexto
    if "business_area" in table_data:
        area = table_data['business_area']
        text_parts.append(f"ÁREA DE NEGÓCIO: {area}")
        
        # Adicionar contexto adicional baseado na área
        area_contexts = {
            'vendas': 'comercial faturamento receita',
            'financeiro': 'contabilidade pagamentos recebimentos',
            'estoque': 'inventário produtos mercadorias',
            'clientes': 'consumidores usuários cadastro',
            'recursos humanos': 'funcionários colaboradores pessoal',
            'marketing': 'propaganda campanhas promoções'
        }
        
        for area_key, context in area_contexts.items():
            if area_key in area.lower():
                text_parts.append(f"CONTEXTO: {context}")
    
    return " | ".join(text_parts)

def create_field_searchable_text(field_name: str, field_data: Dict[str, Any], table_name: str) -> str:
    """Cria texto otimizado para busca de campos específicos em português"""
    # Traduções de tipos para português
    type_translations = {
        'varchar': 'texto variável',
        'char': 'texto fixo caracteres',
        'text': 'texto longo descrição',
        'integer': 'número inteiro',
        'int': 'número inteiro',
        'decimal': 'número decimal valor monetário',
        'float': 'número decimal ponto flutuante',
        'double': 'número decimal dupla precisão',
        'date': 'data dia mês ano',
        'datetime': 'data hora carimbo temporal',
        'timestamp': 'carimbo temporal data hora',
        'boolean': 'verdadeiro falso sim não',
        'bit': 'binário zero um'
    }
    
    field_type = field_data.get('type', 'unknown')
    field_type_pt = type_translations.get(field_type.lower(), field_type)
    
    text_parts = [
        f"CAMPO: {field_name}",
        f"TABELA: {table_name}",
        f"TIPO: {field_type_pt}"
    ]
    
    # Adicionar descrição
    if "description" in field_data:
        text_parts.append(f"DESCRIÇÃO: {field_data['description']}")
    
    # Adicionar características específicas do português
    field_lower = field_name.lower()
    
    # Identificar padrões comuns em português
    if any(pattern in field_lower for pattern in ['id', 'codigo', 'cod', 'key']):
        text_parts.append("FUNÇÃO: identificador único chave primária código")
    elif any(pattern in field_lower for pattern in ['nome', 'name', 'denominacao']):
        text_parts.append("FUNÇÃO: denominação título nome identificação")
    elif any(pattern in field_lower for pattern in ['desc', 'descricao', 'description']):
        text_parts.append("FUNÇÃO: descrição detalhamento explicação")
    elif any(pattern in field_lower for pattern in ['valor', 'preco', 'price', 'saldo', 'total']):
        text_parts.append("FUNÇÃO: valor monetário preço quantia dinheiro")
    elif any(pattern in field_lower for pattern in ['data', 'date', 'criado', 'atualizado']):
        text_parts.append("FUNÇÃO: informação temporal data momento")
    elif any(pattern in field_lower for pattern in ['status', 'situacao', 'estado']):
        text_parts.append("FUNÇÃO: situação estado condição status")
    elif any(pattern in field_lower for pattern in ['email', 'mail', 'correio']):
        text_parts.append("FUNÇÃO: correio eletrônico email contato")
    elif any(pattern in field_lower for pattern in ['telefone', 'phone', 'fone']):
        text_parts.append("FUNÇÃO: telefone contato comunicação")
    elif any(pattern in field_lower for pattern in ['endereco', 'address', 'rua']):
        text_parts.append("FUNÇÃO: endereço localização endereço")
    
    # Verificar se é pesquisável
    if field_data.get("searchable", True):  # Padrão True para português
        text_parts.append("PESQUISÁVEL: sim busca consulta")
    
    # Adicionar sinônimos
    if "synonyms" in field_data:
        text_parts.append(f"SINÔNIMOS: {', '.join(field_data['synonyms'])}")
    
    # Adicionar sinônimos automáticos baseados no nome do campo
    auto_synonyms = []
    if 'cliente' in field_lower:
        auto_synonyms.extend(['consumidor', 'usuário', 'comprador'])
    if 'produto' in field_lower:
        auto_synonyms.extend(['item', 'mercadoria', 'artigo'])
    if 'pedido' in field_lower:
        auto_synonyms.extend(['order', 'solicitação', 'requisição'])
    if 'venda' in field_lower:
        auto_synonyms.extend(['transação', 'compra', 'negociação'])
    
    if auto_synonyms:
        text_parts.append(f"TERMOS RELACIONADOS: {' '.join(auto_synonyms)}")
    
    return " | ".join(text_parts)

async def ingest_database_to_chromadb(database_data: Dict[str, Any], collection_name: str) -> Dict[str, Any]:
    """Faz ingestão completa de dados de database no ChromaDB"""
    if not chroma_client:
        raise HTTPException(status_code=503, detail="ChromaDB não está disponível")
    
    try:
        # Tentar deletar coleção existente
        try:
            chroma_client.delete_collection(collection_name)
        except:
            pass
        
        # Criar nova coleção
        collection = chroma_client.create_collection(
            name=collection_name,
            metadata={"description": f"Database schema: {database_data.get('name', 'Unknown')}"}
        )
        
        documents = []
        metadatas = []
        ids = []
        
        db_info = database_data.get("database", {})
        
        # 1. Ingerir informações das tabelas
        tables = db_info.get("tables", {})
        for table_name, table_data in tables.items():
            # Documento da tabela
            table_text = create_searchable_text(table_data, table_name)
            documents.append(table_text)
            metadatas.append({
                "type": "table",
                "table_name": table_name,
                "business_area": table_data.get("business_area", "general"),
                "importance": "high"
            })
            ids.append(f"table_{table_name}")
            
            # 2. Ingerir campos importantes
            fields = table_data.get("fields", {})
            for field_name, field_data in fields.items():
                if field_data.get("searchable", True) or field_data.get("type") in ["decimal", "integer"] or "saldo" in field_name.lower() or "valor" in field_name.lower():
                    field_text = create_field_searchable_text(field_name, field_data, table_name)
                    documents.append(field_text)
                    metadatas.append({
                        "type": "field",
                        "table_name": table_name,
                        "field_name": field_name,
                        "data_type": field_data.get("type", "unknown"),
                        "business_area": table_data.get("business_area", "general"),
                        "importance": "medium"
                    })
                    ids.append(f"field_{table_name}_{field_name}")
        
        # 3. Ingerir conceitos de negócio
        business_concepts = db_info.get("business_concepts", {})
        for concept_name, concept_data in business_concepts.items():
            concept_text = f"CONCEITO: {concept_name} | DESCRIÇÃO: {concept_data.get('description', '')} | TABELAS RELACIONADAS: {', '.join(concept_data.get('related_tables', []))} | CAMPOS RELACIONADOS: {', '.join(concept_data.get('related_fields', []))} | SINÔNIMOS: {', '.join(concept_data.get('synonyms', []))}"
            documents.append(concept_text)
            metadatas.append({
                "type": "business_concept",
                "concept_name": concept_name,
                "related_tables": ', '.join(concept_data.get("related_tables", [])),
                "related_fields": ', '.join(concept_data.get("related_fields", [])),
                "importance": "high"
            })
            ids.append(f"concept_{concept_name}")
        
        # Gerar embeddings e adicionar ao ChromaDB
        embeddings = create_simple_embeddings(documents)
        
        collection.add(
            documents=documents,
            metadatas=metadatas,
            ids=ids,
            embeddings=embeddings
        )
        
        return {
            "success": True,
            "collection_name": collection_name,
            "documents_added": len(documents),
            "tables_processed": len(tables),
            "concepts_processed": len(business_concepts)
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Erro na ingestão: {str(e)}")

async def search_database_schema(query: str, collection_name: str, limit: int = 5) -> Dict[str, Any]:
    """Busca informações no schema da database"""
    if not chroma_client:
        raise HTTPException(status_code=503, detail="ChromaDB não está disponível")
    
    try:
        collection = chroma_client.get_collection(collection_name)
        
        # Gerar embedding para a query
        query_embeddings = create_simple_embeddings([query])
        
        # Buscar documentos similares
        results = collection.query(
            query_embeddings=query_embeddings,
            n_results=limit
        )
        
        processed_results = []
        if results and 'documents' in results and results['documents']:
            for i, doc in enumerate(results['documents'][0]):
                metadata = results['metadatas'][0][i] if 'metadatas' in results else {}
                distance = results['distances'][0][i] if 'distances' in results else 0
                
                processed_results.append({
                    "document": doc,
                    "metadata": metadata,
                    "relevance_score": round(1 - distance / 100, 3),  # Converter distância em score
                    "type": metadata.get("type", "unknown"),
                    "table_name": metadata.get("table_name", ""),
                    "field_name": metadata.get("field_name", "")
                })
        
        return {
            "results": processed_results,
            "total_found": len(processed_results),
            "query": query,
            "collection": collection_name
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Erro na busca: {str(e)}")



async def generate_response(message: str, context: Optional[List[Message]] = None) -> AsyncGenerator[str, None]:
    """Gera resposta streaming do LMStudio com contexto opcional"""
    try:
        # Primeiro, testa se o LMStudio está acessível
        try:
            async with httpx.AsyncClient(timeout=5.0) as client:
                test_response = await client.get(f"{LMSTUDIO_BASE_URL}/models")
                if test_response.status_code != 200:
                    error_data = {
                        "error": f"LMStudio não está respondendo corretamente. Status: {test_response.status_code}",
                        "type": "error"
                    }
                    yield f"data: {json.dumps(error_data)}\n\n"
                    return
        except httpx.ConnectError:
            # Se não conseguir conectar ao LMStudio, mostra erro
            error_data = {
                "error": "Não foi possível conectar ao LMStudio. Verifique se o serviço está executando.",
                "type": "error"
            }
            yield f"data: {json.dumps(error_data)}\n\n"
            return
        except httpx.TimeoutException:
            # Se timeout, mostra erro
            error_data = {
                "error": "Timeout na conexão com LMStudio. O serviço pode estar sobrecarregado.",
                "type": "error"
            }
            yield f"data: {json.dumps(error_data)}\n\n"
            return
        
        # Constrói a lista de mensagens com contexto
        messages = []
        
        # Adiciona contexto se fornecido
        if context:
            for msg in context:
                messages.append({
                    "role": msg.role,
                    "content": msg.content
                })
        
        # Adiciona a mensagem atual
        messages.append({
            "role": "user",
            "content": message
        })
        
        # Payload para o LMStudio
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
        
        # Faz a requisição streaming para o LMStudio
        async with httpx.AsyncClient(timeout=60.0) as client:
            async with client.stream(
                "POST", 
                f"{LMSTUDIO_BASE_URL}/chat/completions",
                json=payload,
                headers=headers
            ) as response:
                
                if response.status_code != 200:
                    error_text = await response.aread()
                    error_data = {
                        "error": f"Erro na API do LMStudio: {response.status_code} - {error_text.decode()}",
                        "type": "error"
                    }
                    yield f"data: {json.dumps(error_data)}\n\n"
                    return
                
                # Buffer para acumular texto
                buffer = ""
                
                async for line in response.aiter_lines():
                    if line.startswith("data: "):
                        data_str = line[6:]  # Remove "data: "
                        
                        if data_str.strip() == "[DONE]":
                            break
                            
                        try:
                            data = json.loads(data_str)
                            choices = data.get("choices", [])
                            
                            if choices and len(choices) > 0:
                                delta = choices[0].get("delta", {})
                                content = delta.get("content", "")
                                
                                if content:
                                    # Envia cada caractere/palavra imediatamente para efeito streaming
                                    chunk_data = {
                                        "content": content,
                                        "type": "chunk"
                                    }
                                    yield f"data: {json.dumps(chunk_data)}\n\n"
                                    
                                    # Pequena pausa para efeito visual de streaming
                                    await asyncio.sleep(0.01)
                                    
                        except json.JSONDecodeError:
                            # Ignora linhas que não são JSON válido
                            continue
        
        # Sinal de fim
        yield f"data: {json.dumps({'type': 'end'})}\n\n"
        
    except httpx.ConnectError as e:
        error_data = {
            "error": f"Erro de conexão com LMStudio: {str(e)}",
            "type": "error"
        }
        yield f"data: {json.dumps(error_data)}\n\n"
    except Exception as e:
        error_data = {
            "error": f"Erro inesperado: {str(e)}",
            "type": "error"
        }
        yield f"data: {json.dumps(error_data)}\n\n"

@app.get("/")
async def root():
    return {"message": "LMStudio Chat API is running"}

@app.post("/chat")
async def chat_endpoint(request: ChatRequest):
    """Endpoint para chat com streaming e contexto"""
    if not request.message.strip():
        raise HTTPException(status_code=400, detail="Mensagem não pode estar vazia")
    
    # Gera session_id se não fornecido
    session_id = request.session_id or str(uuid.uuid4())
    
    # Recupera contexto da sessão ou usa o fornecido
    context = request.context
    if not context and session_id in chat_sessions:
        context = chat_sessions[session_id]
    
    # Armazena a mensagem do usuário na sessão
    user_message = Message(
        role="user", 
        content=request.message,
        timestamp=datetime.now().isoformat()
    )
    
    if session_id not in chat_sessions:
        chat_sessions[session_id] = []
    
    chat_sessions[session_id].append(user_message)
    
    # Função para capturar a resposta completa e armazenar na sessão
    async def generate_and_store_response():
        full_response = ""
        
        async for chunk in generate_response(request.message, context):
            yield chunk
            
            # Extrai o conteúdo da resposta para armazenar
            if chunk.startswith("data: "):
                try:
                    data_str = chunk[6:].strip()
                    if data_str and data_str != '{"type": "end"}':
                        chunk_data = json.loads(data_str)
                        if chunk_data.get("type") == "chunk" and "content" in chunk_data:
                            full_response += chunk_data["content"]
                except json.JSONDecodeError:
                    continue
        
        # Armazena a resposta completa na sessão
        if full_response.strip():
            assistant_message = Message(
                role="assistant",
                content=full_response.strip(),
                timestamp=datetime.now().isoformat()
            )
            chat_sessions[session_id].append(assistant_message)
        
        # Adiciona session_id na resposta final
        session_info = {
            "type": "session_info",
            "session_id": session_id
        }
        yield f"data: {json.dumps(session_info)}\n\n"
    
    return StreamingResponse(
        generate_and_store_response(),
        media_type="text/plain",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "Content-Type": "text/event-stream",
        }
    )


@app.post("/chat/clear")
async def clear_chat_endpoint(request: ClearChatRequest):
    """Endpoint para limpar o histórico de uma sessão"""
    session_id = request.session_id
    
    if session_id in chat_sessions:
        del chat_sessions[session_id]
        return {"message": f"Histórico da sessão {session_id} foi limpo", "session_id": session_id}
    else:
        return {"message": f"Sessão {session_id} não encontrada", "session_id": session_id}

@app.get("/chat/sessions")
async def list_sessions():
    """Lista todas as sessões ativas"""
    sessions_info = {}
    for session_id, messages in chat_sessions.items():
        sessions_info[session_id] = {
            "message_count": len(messages),
            "last_message": messages[-1].timestamp if messages else None
        }
    return {"sessions": sessions_info}

@app.get("/chat/session/{session_id}")
async def get_session_context(session_id: str):
    """Recupera o contexto completo de uma sessão"""
    if session_id in chat_sessions:
        return {
            "session_id": session_id,
            "messages": chat_sessions[session_id],
            "message_count": len(chat_sessions[session_id])
        }
    else:
        raise HTTPException(status_code=404, detail=f"Sessão {session_id} não encontrada")

@app.delete("/chat/sessions")
async def clear_all_sessions():
    """Limpa todas as sessões"""
    session_count = len(chat_sessions)
    chat_sessions.clear()
    return {"message": f"Todas as {session_count} sessões foram limpas"}

# Endpoints do ChromaDB
@app.post("/database/upload")
async def upload_database_json(
    file: UploadFile = File(...),
    database_name: Optional[str] = None,
    overwrite: bool = False
):
    """Upload e ingestão de arquivo JSON com schema de database"""
    print(f"DEBUG: Upload iniciado - filename: {file.filename}")
    if not chroma_client:
        raise HTTPException(status_code=503, detail="ChromaDB não está disponível")
    
    # Verificar se é arquivo JSON
    if not file.filename.endswith('.json'):
        raise HTTPException(status_code=400, detail="Arquivo deve ser JSON")
    
    try:
        # Ler arquivo JSON
        content = await file.read()
        database_data = json.loads(content.decode('utf-8'))
        
        # Usar nome do arquivo como nome da coleção se não fornecido
        if not database_name:
            database_name = file.filename.replace('.json', '').lower().replace(' ', '_')
        
        # Verificar se coleção já existe
        try:
            existing_collections = [col.name for col in chroma_client.list_collections()]
            if database_name in existing_collections and not overwrite:
                raise HTTPException(
                    status_code=409, 
                    detail=f"Database '{database_name}' já existe. Use overwrite=true para sobrescrever"
                )
        except Exception as e:
            if "já existe" in str(e):
                raise e
        
        # Fazer ingestão
        result = await ingest_database_to_chromadb(database_data, database_name)
        
        return {
            "message": "Database carregada com sucesso",
            "filename": file.filename,
            "database_name": database_name,
            **result
        }
        
    except json.JSONDecodeError:
        raise HTTPException(status_code=400, detail="Arquivo JSON inválido")
    except Exception as e:
        import traceback
        print(f"ERRO DETALHADO NO UPLOAD: {str(e)}")
        print(f"TRACEBACK: {traceback.format_exc()}")
        raise HTTPException(status_code=500, detail=f"Erro no upload: {str(e)}")

@app.post("/database/search", response_model=DatabaseSearchResponse)
async def search_database(request: DatabaseSearchRequest):
    """Busca semântica no schema da database"""
    if not chroma_client:
        raise HTTPException(status_code=503, detail="ChromaDB não está disponível")
    
    # Se não especificou database, buscar na primeira coleção disponível
    collection_name = request.database_name
    if not collection_name:
        collections = chroma_client.list_collections()
        if not collections:
            raise HTTPException(status_code=404, detail="Nenhuma database carregada")
        collection_name = collections[0].name
    
    # Realizar busca
    results = await search_database_schema(request.query, collection_name, request.limit)
    
    return DatabaseSearchResponse(
        results=results["results"],
        total_found=results["total_found"],
        query=results["query"]
    )

@app.get("/database/list")
async def list_databases():
    """Lista todas as databases carregadas no ChromaDB"""
    if not chroma_client:
        raise HTTPException(status_code=503, detail="ChromaDB não está disponível")
    
    try:
        collections = chroma_client.list_collections()
        
        databases = []
        for collection in collections:
            try:
                count = collection.count()
                databases.append({
                    "name": collection.name,
                    "documents_count": count,
                    "metadata": collection.metadata
                })
            except:
                databases.append({
                    "name": collection.name,
                    "documents_count": 0,
                    "metadata": {}
                })
        
        return {
            "databases": databases,
            "total_databases": len(databases)
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Erro ao listar databases: {str(e)}")

@app.delete("/database/{database_name}")
async def delete_database(database_name: str):
    """Remove uma database do ChromaDB"""
    if not chroma_client:
        raise HTTPException(status_code=503, detail="ChromaDB não está disponível")
    
    try:
        chroma_client.delete_collection(database_name)
        return {"message": f"Database '{database_name}' removida com sucesso"}
    except Exception as e:
        raise HTTPException(status_code=404, detail=f"Database '{database_name}' não encontrada ou erro: {str(e)}")

@app.get("/database/{database_name}/info")
async def get_database_info(database_name: str):
    """Obtém informações detalhadas de uma database"""
    if not chroma_client:
        raise HTTPException(status_code=503, detail="ChromaDB não está disponível")
    
    try:
        collection = chroma_client.get_collection(database_name)
        
        # Buscar documentos por tipo
        tables_results = collection.get(where={"type": "table"})
        fields_results = collection.get(where={"type": "field"})
        concepts_results = collection.get(where={"type": "business_concept"})
        
        return {
            "database_name": database_name,
            "total_documents": collection.count(),
            "tables_count": len(tables_results["ids"]) if tables_results["ids"] else 0,
            "fields_count": len(fields_results["ids"]) if fields_results["ids"] else 0,
            "concepts_count": len(concepts_results["ids"]) if concepts_results["ids"] else 0,
            "metadata": collection.metadata
        }
        
    except Exception as e:
        raise HTTPException(status_code=404, detail=f"Database '{database_name}' não encontrada: {str(e)}")

@app.get("/health")
async def health_check():
    """Endpoint para verificar saúde da API"""
    try:
        base_url = os.getenv("LMSTUDIO_BASE_URL", "http://localhost:1234/v1")
        
        # Testa conexão com LMStudio
        async with httpx.AsyncClient(timeout=5.0) as client:
            response = await client.get(f"{base_url}/models")
            
        return {
            "status": "healthy", 
            "lmstudio_url": base_url,
            "lmstudio_status": response.status_code,
            "chromadb_status": "connected" if chromadb_client else "disconnected"
        }
    except Exception as e:
        raise HTTPException(status_code=503, detail=f"Erro na conexão com LMStudio: {str(e)}")

# Endpoints de Chat Especializado com Streaming
@app.get("/api/chat/stream")
@app.post("/api/chat/stream")
async def general_chat_stream(request: SpecializedChatRequest = None, message: str = None, context: str = None, collection_name: Optional[str] = None):
    """Chat geral com streaming e contexto do ChromaDB"""
    
    # Suporte para GET (EventSource) e POST
    if request is None:
        request = SpecializedChatRequest(
            message=message or "",
            context=json.loads(context) if context else None
        )
    
    system_prompt = """Você é um assistente inteligente especializado em sistemas comerciais e bancos de dados.
    
DIRETRIZES:
- Responda de forma clara e educativa
- Use as informações do contexto fornecido sempre que possível
- Se não souber algo, seja honesto sobre isso
- Mantenha respostas focadas e úteis
- Fale em português brasileiro
    
Você tem acesso à base de conhecimento sobre:
- Estrutura de banco de dados comercial
- Regras de negócio
- Serviços do sistema"""
    
    return StreamingResponse(
        generate_specialized_response_stream(
            message=request.message,
            system_prompt=system_prompt,
            context=request.context,
            use_chromadb=True,
            collection_name=collection_name
        ),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "Access-Control-Allow-Origin": "*",
            "Access-Control-Allow-Headers": "*",
        }
    )

@app.get("/api/chat/sql/stream")
@app.post("/api/chat/sql/stream")
async def sql_chat_stream(request: SpecializedChatRequest = None, message: str = None, context: str = None, collection_name: Optional[str] = None):
    """Chat especializado em geração de SQL com streaming"""
    
    # Suporte para GET (EventSource) e POST
    if request is None:
        request = SpecializedChatRequest(
            message=message or "",
            context=json.loads(context) if context else None
        )
    
    system_prompt = """Você é um especialista em SQL que deve responder APENAS com queries SQL válidas baseadas na estrutura do banco de dados fornecida no contexto.

REGRAS IMPORTANTES:
- Responda SOMENTE com código SQL
- Use a estrutura das tabelas fornecida no contexto
- Não inclua explicações, apenas o SQL
- Se não souber, responda: "-- Informações insuficientes para gerar SQL"
- Use sintaxe SQL padrão
- Sempre termine com ponto e vírgula
- Use nomes de tabelas e campos conforme fornecido no contexto
- Considere as relações entre tabelas mostradas no contexto

Estrutura disponível no contexto:"""
    
    return StreamingResponse(
        generate_specialized_response_stream(
            message=request.message,
            system_prompt=system_prompt,
            context=request.context,
            use_chromadb=True,
            collection_name=collection_name
        ),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "Access-Control-Allow-Origin": "*",
            "Access-Control-Allow-Headers": "*",
        }
    )

@app.get("/api/chat/help/stream")
@app.post("/api/chat/help/stream")
async def help_chat_stream(request: SpecializedChatRequest = None, message: str = None, context: str = None, collection_name: Optional[str] = None):
    """Chat especializado em responder dúvidas sobre o sistema com streaming"""
    
    # Suporte para GET (EventSource) e POST
    if request is None:
        request = SpecializedChatRequest(
            message=message or "",
            context=json.loads(context) if context else None
        )
    
    system_prompt = """Você é um assistente especializado em responder dúvidas sobre o sistema comercial baseado nas informações do ChromaDB.

REGRAS IMPORTANTES:
- Responda dúvidas técnicas e de negócio
- Use APENAS as informações fornecidas no contexto ChromaDB
- Seja claro e objetivo
- Se a informação não estiver no contexto, diga: "Esta informação não está disponível no sistema"
- Mantenha foco no sistema comercial
- Pode explicar conceitos e processos
- Fale em português brasileiro
- Formate respostas de forma organizada
- Use bullets ou numeração quando apropriado

Base de conhecimento do ChromaDB sobre:
- Estrutura de tabelas e relacionamentos
- Regras de negócio e validações
- Serviços e processos do sistema"""
    
    return StreamingResponse(
        generate_specialized_response_stream(
            message=request.message,
            system_prompt=system_prompt,
            context=request.context,
            use_chromadb=True,
            collection_name=collection_name
        ),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "Access-Control-Allow-Origin": "*",
            "Access-Control-Allow-Headers": "*",
        }
    )

# Endpoints de Chat Especializado
@app.post("/api/chat", response_model=SpecializedChatResponse)
async def general_chat(request: SpecializedChatRequest):
    """Chat geral com contexto do ChromaDB"""
    
    system_prompt = """Você é um assistente inteligente especializado em sistemas comerciais e bancos de dados.
    
DIRETRIZES:
- Responda de forma clara e educativa
- Use as informações do contexto fornecido sempre que possível
- Se não souber algo, seja honesto sobre isso
- Mantenha respostas focadas e úteis
- Fale em português brasileiro
    
Você tem acesso à base de conhecimento sobre:
- Estrutura de banco de dados comercial
- Regras de negócio
- Serviços do sistema"""
    
    session_id = request.session_id or str(uuid.uuid4())
    
    try:
        result = await generate_specialized_response(
            message=request.message,
            system_prompt=system_prompt,
            context=request.context,
            use_chromadb=True
        )
        
        return SpecializedChatResponse(
            response=result["response"],
            session_id=session_id,
            context_used=result.get("context_used")
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Erro no chat geral: {str(e)}")

@app.post("/api/chat/sql", response_model=SpecializedChatResponse)
async def sql_chat(request: SpecializedChatRequest):
    """Chat especializado em geração de SQL"""
    
    system_prompt = """Você é um especialista em SQL que deve responder APENAS com queries SQL válidas baseadas na estrutura do banco de dados fornecida no contexto.

REGRAS IMPORTANTES:
- Responda SOMENTE com código SQL
- Use a estrutura das tabelas fornecida no contexto
- Não inclua explicações, apenas o SQL
- Se não souber, responda: "-- Informações insuficientes para gerar SQL"
- Use sintaxe SQL padrão
- Sempre termine com ponto e vírgula
- Use nomes de tabelas e campos conforme fornecido no contexto
- Considere as relações entre tabelas mostradas no contexto

Estrutura disponível no contexto:"""
    
    session_id = request.session_id or str(uuid.uuid4())
    
    try:
        result = await generate_specialized_response(
            message=request.message,
            system_prompt=system_prompt,
            context=request.context,
            use_chromadb=True
        )
        
        return SpecializedChatResponse(
            response=result["response"],
            session_id=session_id,
            context_used=result.get("context_used")
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Erro no chat SQL: {str(e)}")

@app.post("/api/chat/help", response_model=SpecializedChatResponse)
async def help_chat(request: SpecializedChatRequest):
    """Chat especializado em responder dúvidas sobre o sistema"""
    
    system_prompt = """Você é um assistente especializado em responder dúvidas sobre o sistema comercial baseado nas informações do ChromaDB.

REGRAS IMPORTANTES:
- Responda dúvidas técnicas e de negócio
- Use APENAS as informações fornecidas no contexto ChromaDB
- Seja claro e objetivo
- Se a informação não estiver no contexto, diga: "Esta informação não está disponível no sistema"
- Mantenha foco no sistema comercial
- Pode explicar conceitos e processos
- Fale em português brasileiro
- Formate respostas de forma organizada
- Use bullets ou numeração quando apropriado

Base de conhecimento do ChromaDB sobre:
- Estrutura de tabelas e relacionamentos
- Regras de negócio e validações
- Serviços e processos do sistema"""
    
    session_id = request.session_id or str(uuid.uuid4())
    
    try:
        result = await generate_specialized_response(
            message=request.message,
            system_prompt=system_prompt,
            context=request.context,
            use_chromadb=True
        )
        
        return SpecializedChatResponse(
            response=result["response"],
            session_id=session_id,
            context_used=result.get("context_used")
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Erro no chat de ajuda: {str(e)}")
@app.post("/vectordb/query", response_model=VectorDBQueryResponse)
async def query_vectordb(request: VectorDBQueryRequest, collection_name: Optional[str] = None):
    """Faz uma pergunta ao ChromaDB e retorna resultados similares"""
    if not chromadb_client:
        raise HTTPException(status_code=503, detail="ChromaDB não está disponível")
    
    try:
        import time
        start_time = time.time()
        
        # Define a coleção a ser usada (padrão ou especificada)
        target_collection = collection_name if collection_name else "sistema_comercial"
        print(f"🎯 API: Usando coleção '{target_collection}' para consulta")
        
        # Define a coleção (set_collection já cria se não existir)
        if not chromadb_client.set_collection(target_collection):
            raise HTTPException(status_code=500, detail=f"Erro ao definir coleção '{target_collection}'")
        
        # Faz a consulta
        results = chromadb_client.query(request.question, n_results=request.n_results, context=request.context)
        
        processing_time = time.time() - start_time
        
        return VectorDBQueryResponse(
            question=request.question,
            results=results or [],
            total_results=len(results) if results else 0,
            processing_time=processing_time
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Erro ao consultar VectorDB: {str(e)}")

@app.get("/vectordb/stats")
async def get_vectordb_stats(collection_name: Optional[str] = None):
    """Retorna estatísticas do VectorDB"""
    print("🚀 API: Endpoint /vectordb/stats chamado!")
    
    if not chromadb_client:
        raise HTTPException(status_code=503, detail="ChromaDB não está disponível")
    
    try:
        print(f"🔍 API: Recebido collection_name='{collection_name}'")
        
        # Define a coleção a ser usada (padrão ou especificada)
        target_collection = collection_name if collection_name else "sistema_comercial"
        print(f"🎯 API: Usando coleção '{target_collection}'")
        
        # Define a coleção (set_collection já cria se não existir)
        if not chromadb_client.set_collection(target_collection):
            raise HTTPException(status_code=500, detail=f"Erro ao definir coleção '{target_collection}'")
        
        stats = chromadb_client.get_collection_stats()
        return stats
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Erro ao obter estatísticas: {str(e)}")@app.post("/vectordb/update-database")
async def update_database_structure(request: DatabaseUpdateRequest):
    """Atualiza a estrutura do banco de dados no VectorDB"""
    if not chromadb_client:
        raise HTTPException(status_code=503, detail="ChromaDB não está disponível")
    
    try:
        # Usa o método do cliente para atualizar a estrutura
        if not chromadb_client.update_database_structure(request.database_structure):
            raise HTTPException(status_code=500, detail="Erro ao atualizar estrutura do banco")
        
        stats = chromadb_client.get_collection_stats()
        
        return {
            "message": "Estrutura do banco atualizada com sucesso",
            "stats": stats
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Erro ao atualizar estrutura: {str(e)}")

@app.post("/vectordb/reload")
async def reload_vectordb():
    """Recarrega todos os dados no VectorDB"""
    if not chromadb_client:
        raise HTTPException(status_code=503, detail="ChromaDB não está disponível")
    
    try:
        # Reconecta
        if not chromadb_client.connect():
            raise HTTPException(status_code=500, detail="Erro ao conectar ao ChromaDB")
        
        # Recria coleção
        if not chromadb_client.create_collection():
            raise HTTPException(status_code=500, detail="Erro ao criar coleção")
            
        # Recarrega dados
        if not chromadb_client.load_and_index_documents():
            raise HTTPException(status_code=500, detail="Erro ao carregar dados")
        
        stats = chromadb_client.get_collection_stats()
        
        return {
            "message": "VectorDB recarregado com sucesso",
            "stats": stats
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Erro ao recarregar VectorDB: {str(e)}")

@app.post("/vectordb/clear")
async def clear_vectordb():
    """
    Limpa toda a base de dados do VectorDB
    """
    try:
        if not chromadb_client.connect():
            raise HTTPException(status_code=500, detail="Erro ao conectar ao ChromaDB")
        
        # Remove a coleção existente se ela existir
        try:
            # Primeiro tenta obter a coleção para ver se existe
            collection = chromadb_client.client.get_collection("sistema_comercial")
            if collection:
                chromadb_client.client.delete_collection("sistema_comercial")
                print("Coleção 'sistema_comercial' deletada com sucesso")
        except Exception as e:
            print(f"Coleção pode não existir ou erro ao deletar: {e}")
        
        # Recria a coleção vazia
        if not chromadb_client.create_collection():
            raise HTTPException(status_code=500, detail="Erro ao recriar coleção")
        
        stats = chromadb_client.get_collection_stats()
        
        return {
            "message": "Base de dados limpa com sucesso",
            "stats": stats
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Erro ao limpar base: {str(e)}")

@app.post("/vectordb/upload")
async def upload_json_file(file: UploadFile = File(...), type: str = ""):
    """
    Carrega um arquivo JSON específico para a base de dados
    """
    try:
        print(f"DEBUG: Upload iniciado - filename: {file.filename}, type: {type}")
        
        if not file.filename.endswith('.json'):
            raise HTTPException(status_code=400, detail="Arquivo deve ser JSON")
        
        # Se type estiver vazio, tentar inferir do nome do arquivo
        if not type:
            if 'business' in file.filename.lower():
                type = 'business_rules'
            elif 'database' in file.filename.lower() or 'struct' in file.filename.lower():
                type = 'database_struct'
            elif 'service' in file.filename.lower():
                type = 'system_services'
            else:
                raise HTTPException(status_code=400, detail="Não foi possível inferir o tipo do arquivo. Especifique o parâmetro 'type'")
        
        if type not in ['business_rules', 'database_struct', 'system_services']:
            raise HTTPException(status_code=400, detail="Tipo deve ser: business_rules, database_struct ou system_services")
        
        # Lê o conteúdo do arquivo
        content = await file.read()
        json_data = json.loads(content.decode('utf-8'))
        print(f"DEBUG: Arquivo JSON carregado com sucesso, tamanho: {len(content)} bytes")
        
        if not chromadb_client.connect():
            raise HTTPException(status_code=500, detail="Erro ao conectar ao ChromaDB")
        
        if not chromadb_client.create_collection():
            raise HTTPException(status_code=500, detail="Erro ao criar coleção")
        
        print(f"DEBUG: Conexão com ChromaDB estabelecida, processando tipo: {type}")
        
        # Processa o arquivo baseado no tipo
        documents = []
        if type == 'business_rules':
            documents = chromadb_client.processor.extract_business_rules_documents(json_data)
        elif type == 'database_struct':
            documents = chromadb_client.processor.extract_database_structure_documents(json_data)
        elif type == 'system_services':
            documents = chromadb_client.processor.extract_services_documents(json_data)
        
        print(f"DEBUG: {len(documents)} documentos extraídos")
        
        if not documents:
            raise HTTPException(status_code=400, detail="Nenhum documento extraído do arquivo")
        
        # Adiciona documentos em lotes
        batch_size = 15
        for i in range(0, len(documents), batch_size):
            batch = documents[i:i+batch_size]
            
            ids = [doc['id'] for doc in batch]
            texts = [doc['text'] for doc in batch]
            metadatas = [doc['metadata'] for doc in batch]
            
            chromadb_client.collection.add(
                documents=texts,
                metadatas=metadatas,
                ids=ids
            )
        
        stats = chromadb_client.get_collection_stats()
        
        return {
            "message": f"Arquivo {type} carregado com sucesso. {len(documents)} documentos adicionados.",
            "stats": stats,
            "documents_added": len(documents)
        }
        
    except json.JSONDecodeError:
        raise HTTPException(status_code=400, detail="Arquivo JSON inválido")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Erro ao carregar arquivo: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=False)
