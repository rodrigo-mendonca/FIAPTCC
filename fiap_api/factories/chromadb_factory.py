#!/usr/bin/env python3
"""
ChromaDB Factory - Gerenciamento de banco de dados vetorial
Responsável por: criação de coleções, adição, deleção, consultas e processamento de documentos

Suporta integração com LangChain via Chroma.from_documents() e retriever
"""

import os
from chromadb import EmbeddingFunction, Documents
import chromadb
import requests
import yaml
from typing import List, Dict, Any, Optional
from datetime import datetime, timezone, timedelta
from pathlib import Path

# Imports adicionais para suporte a LangChain Chroma
try:
    from langchain_chroma import Chroma
    from langchain_core.documents import Document
    LANGCHAIN_CHROMA_AVAILABLE = True
except ImportError:
    LANGCHAIN_CHROMA_AVAILABLE = False
    Chroma = None
    Document = None


def get_local_datetime():
    """Retorna datetime no horário local brasileiro (America/Sao_Paulo)"""
    utc_now = datetime.now(timezone.utc)
    # UTC-3 (horário de Brasília)
    brasilia_offset = timedelta(hours=-3)
    brasilia_tz = timezone(brasilia_offset)
    return utc_now.astimezone(brasilia_tz)


class LMStudioEmbeddingFunction(EmbeddingFunction):
    """
    Função de embedding personalizada que usa o LMStudio
    """
    def __init__(self, endpoint: str, model: str, embedding_dimension: int = 768):
        self.lmstudio_url = endpoint
        self.model = model
        self.embedding_dimension = embedding_dimension
        
    def __call__(self, input: Documents) -> List[List[float]]:
        """
        Gera embeddings usando o LMStudio
        
        Args:
            input: Lista de textos para gerar embeddings
            
        Returns:
            Lista de embeddings (vetores)
        """
        embeddings = []
        
        for text in input:
            try:
                response = requests.post(
                    f"{self.lmstudio_url}/embeddings",
                    headers={"Content-Type": "application/json"},
                    json={
                        "input": text,
                        "model": self.model
                    },
                    timeout=30
                )
                
                if response.status_code == 200:
                    result = response.json()
                    if 'data' in result and len(result['data']) > 0 and 'embedding' in result['data'][0]:
                        embedding = result['data'][0]['embedding']
                        embeddings.append(embedding)
                    else:
                        print(f"⚠️ Resposta de embedding inválida para '{text[:50]}...': {result}")
                        # Fallback: gerar um embedding dummy
                        embeddings.append([0.0] * self.embedding_dimension)
                else:
                    print(f"⚠️ Erro ao gerar embedding para '{text[:50]}...': {response.status_code} - {response.text}")
                    # Fallback: gerar um embedding dummy
                    embeddings.append([0.0] * self.embedding_dimension)
                    
            except Exception as e:
                print(f"⚠️ Erro na requisição de embedding: {e} - Resposta: {response.text if 'response' in locals() else 'N/A'}")
                # Fallback: gerar um embedding dummy
                embeddings.append([0.0] * self.embedding_dimension)
                
        return embeddings


class DatabaseDocumentProcessor:
    """
    Processador para converter dados do banco em documentos para ChromaDB
    """
    
    @staticmethod
    def load_yaml_files_from_folder(folder_path: str) -> List[Dict]:
        """
        Carrega todos os arquivos YAML de uma pasta
        
        Args:
            folder_path: Caminho da pasta contendo arquivos YAML
            
        Returns:
            Lista de dicionários carregados dos arquivos YAML (excluindo _metadata.yaml)
        """
        try:
            if not os.path.isabs(folder_path):
                current_dir = os.path.dirname(os.path.abspath(__file__))
                folder_path = os.path.join(current_dir, folder_path)
            
            data_list = []
            folder = Path(folder_path)
            
            if not folder.exists():
                print(f"⚠️ Pasta não encontrada: {folder_path}")
                return data_list
            
            # Carrega todos os arquivos YAML exceto _metadata.yaml
            for yaml_file in sorted(folder.glob("*.yaml")):
                if yaml_file.name.startswith("_"):
                    continue  # Pula arquivos de metadata
                
                # Carrega arquivo YAML diretamente
                try:
                    with open(yaml_file, 'r', encoding='utf-8') as f:
                        content = yaml.safe_load(f)
                        if content:
                            data_list.append(content)
                except Exception as file_error:
                    print(f"[WARN] Erro ao carregar {yaml_file}: {file_error}")
                    continue
            return data_list
            
        except Exception as e:
            print(f"[OK] Erro ao carregar pasta YAML {folder_path}: {e}")
            return []


class ChromaDBClient:
    """
    Cliente para interação com ChromaDB
    """
    
    def __init__(self, host: str = None, port: int = None, endpoint: str = None, embeddings_model: str = None):
        """
        Inicializa o cliente ChromaDB
        
        Args:
            host: Endereço do servidor ChromaDB (opcional, carregado do env se não fornecido)
            port: Porta do servidor ChromaDB (opcional, carregado do env se não fornecido)
            endpoint: URL para embeddings (opcional, carregado do env se não fornecido)
            embeddings_model: Modelo de embeddings (opcional, carregado do env se não fornecido)
        """
        from .env_factory import EnvFactory
        import os as os_module
        
        # Valores padrão da aplicação
        DEFAULT_CHROMADB_HOST = "chromadb"
        DEFAULT_CHROMADB_PORT = 8000
        DEFAULT_LMSTUDIO_ENDPOINT = "http://lmstudio:1234/v1"
        
        # Carregar parâmetros do ChromaDB
        if host is None:
            host = os_module.getenv("CHROMADB_HOST", DEFAULT_CHROMADB_HOST)
        if port is None:
            port_str = os_module.getenv("CHROMADB_PORT", str(DEFAULT_CHROMADB_PORT))
            try:
                port = int(port_str)
            except (ValueError, TypeError):
                port = DEFAULT_CHROMADB_PORT
        
        # Carregar parâmetros de Embeddings
        if endpoint is None:
            endpoint = os_module.getenv("LMSTUDIO_ENDPOINT", DEFAULT_LMSTUDIO_ENDPOINT)
        
        if embeddings_model is None:
            embeddings_model = os_module.getenv("LMSTUDIO_MODEL", "nomic-embed-text")
        
        try:
            embeddings_params = EnvFactory.get_embeddings_params()
            if endpoint is None or endpoint == DEFAULT_LMSTUDIO_ENDPOINT:
                endpoint = embeddings_params.endpoint
            if embeddings_model is None or embeddings_model == "nomic-embed-text":
                embeddings_model = embeddings_params.model
            print(f"[INIT] Embeddings params loaded from EnvFactory: endpoint={endpoint}, model={embeddings_model}")
        except Exception as e:
            print(f"[WARN] Erro ao carregar parâmetros de embeddings do EnvFactory: {e}")
            print(f"[INIT] Usando valores default: endpoint={endpoint}, model={embeddings_model}")

        self.host = host
        self.port = port
        self.lmstudio_url = endpoint
        self.embeddings_model = embeddings_model
        self.client = None
        self.collection = None
        self.embedding_function = LMStudioEmbeddingFunction(self.lmstudio_url, self.embeddings_model)
        self.processor = DatabaseDocumentProcessor()
        
        print(f"[INIT] ChromaDBClient inicializado: host={self.host}, port={self.port}, embedding={self.embeddings_model}")
    
    def connect(self) -> bool:
        """
        Conecta aos serviços ChromaDB e LMStudio
        
        Returns:
            True se conectou com sucesso
        """
        try:
            print(f"[CONNECT] Tentando conectar ao ChromaDB em {self.host}:{self.port}...")
            self.client = chromadb.HttpClient(host=self.host, port=self.port)
            
            # Testa a conexão fazendo um heartbeat
            try:
                heartbeat = self.client.heartbeat()
                print(f"[CONNECT] ✓ Conexão com ChromaDB estabelecida! Heartbeat: {heartbeat}")
            except Exception as hb_error:
                print(f"[CONNECT] Heartbeat falhou: {hb_error}")
            
            print(f"[CONNECT] ✓ Conectado com sucesso!")
            return True
            
        except Exception as e:
            print(f"[CONNECT] ✗ Erro ao conectar: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    def create_collection(self, collection_name: str) -> bool:
        """
        Cria ou obtém uma coleção no ChromaDB
        
        Args:
            collection_name: Nome da coleção
            
        Returns:
            True se criou/obteve com sucesso
        """
        try:
            print(f"📚 Criando/obtendo coleção '{collection_name}' com embedding LMStudio...")
            
            self.collection = self.client.get_or_create_collection(
                name=collection_name,
                embedding_function=self.embedding_function,
                metadata={"description": "Sistema comercial - estrutura, regras e serviços"}
            )
            
            print(f"[OK] Coleção '{collection_name}' pronta com embedding personalizado!")
            return True
            
        except Exception as e:
            print(f"[OK] Erro ao criar/obter coleção: {e}")
            return False
    
    def query(self, query_text: str, n_results: int = 5, context: str = "all") -> List[Dict]:
        """
        Busca documentos similares na coleção
        
        Args:
            query_text: Texto da consulta
            n_results: Número máximo de resultados
            context: Contexto para filtrar ('all', 'business_rules', 'database_struct', 'system_services', 'user_routines')
            
        Returns:
            Lista de documentos encontrados
        """
        try:
            print(f"[OK] Buscando: '{query_text}' no contexto: {context}")
            
            # Configura filtros baseado no contexto
            where_filter = None
            if context != "all":
                # Mapeia os contextos para os tipos de documentos
                context_mapping = {
                    'business_rules': 'business_rule',
                    'database_struct': ['table', 'column', 'database_info'],
                    'system_services': 'service',
                    'user_routines': 'rotina_usuario'
                }
                
                if context in context_mapping:
                    filter_value = context_mapping[context]
                    if isinstance(filter_value, list):
                        # Para múltiplos tipos (database_struct)
                        where_filter = {"type": {"$in": filter_value}}
                    else:
                        # Para um tipo específico
                        where_filter = {"type": filter_value}
            
            # Se n_results for None ou <= 0, interpretamos como 'sem limite' e usamos o total de documentos
            if n_results is None or (isinstance(n_results, int) and n_results <= 0):
                try:
                    total_docs = self.collection.count() if self.collection else 0
                    # segurança: se coleção vazia ou count falhar, usa um limite alto
                    n_results = total_docs if total_docs and total_docs > 0 else 10000
                except Exception:
                    n_results = 10000

            # Executa a query com ou sem filtro
            query_params = {
                "query_texts": [query_text],
                "n_results": n_results,
                "include": ['documents', 'metadatas', 'distances']
            }
            
            if where_filter:
                query_params["where"] = where_filter
                print(f"📋 Aplicando filtro: {where_filter}")
            
            results = self.collection.query(**query_params)
            
            formatted_results = []
            if results['documents'] and results['documents'][0]:
                for i, doc in enumerate(results['documents'][0]):
                    result = {
                        'id': results['ids'][0][i],
                        'content': doc,
                        'metadata': results['metadatas'][0][i],
                        'similarity': 1 - results['distances'][0][i],
                        'type': results['metadatas'][0][i].get('type', 'unknown')
                    }
                    formatted_results.append(result)
            
            print(f"[OK] Encontrados {len(formatted_results)} resultados")
            return formatted_results
            
        except Exception as e:
            print(f"[OK] Erro na busca: {e}")
            return []
    
    def get_collection_stats(self) -> Dict:
        """
        Obtém estatísticas da coleção atual e lista todas as coleções
        
        Returns:
            Dicionário com estatísticas
        """
        try:
            # Dados básicos
            collection_name = str(self.collection.name) if self.collection else None
            
            # Tenta contar documentos
            total_docs = 0
            try:
                if self.collection:
                    total_docs = int(self.collection.count())
            except Exception as count_err:
                pass
            
            # Tipos e fontes (simples)
            type_counts = {}
            source_counts = {}
            
            if total_docs > 0 and self.collection:
                try:
                    sample = self.collection.get(limit=100, include=['metadatas'])
                    
                    if sample and 'metadatas' in sample:
                        for metadata in sample['metadatas']:
                            if metadata and isinstance(metadata, dict):
                                doc_type = str(metadata.get('type', 'unknown'))
                                doc_source = str(metadata.get('source', 'unknown'))
                                type_counts[doc_type] = type_counts.get(doc_type, 0) + 1
                                source_counts[doc_source] = source_counts.get(doc_source, 0) + 1
                except Exception:
                    pass
            
            # SEMPRE lista todas as coleções, independente de ter uma selecionada
            all_collections = []
            try:
                if self.client:
                    collections = self.client.list_collections()
                    for col in collections:
                        try:
                            all_collections.append({
                                'name': str(col.name),
                                'count': int(col.count()),
                                'id': str(col.name)
                            })
                        except Exception as col_err:
                            print(f"[STATS] Erro ao processar coleção: {col_err}")
                else:
                    print("[STATS] Client is None, não foi possível listar coleções")
            except Exception as list_err:
                print(f"[STATS] Erro ao listar coleções: {list_err}")
            
            # Monta resultado
            result = {
                'total_documentos': total_docs,
                'collection_name': collection_name,
                'embedding_model': self.embeddings_model,
                'last_updated': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                'tipos_documento': type_counts,
                'fontes_dados': source_counts,
                'collections': all_collections
            }
            
            return result
            
        except Exception as e:
            print(f"[STATS] Erro em get_collection_stats: {e}")
            import traceback
            traceback.print_exc()
            # Tenta retornar algo útil mesmo em erro
            try:
                collection_name = str(self.collection.name) if self.collection else None
            except:
                collection_name = None
            
            # Tenta listar coleções mesmo em erro
            all_collections = []
            if self.client:
                try:
                    collections = self.client.list_collections()
                    for col in collections:
                        try:
                            all_collections.append({
                                'name': str(col.name),
                                'count': int(col.count()),
                                'id': str(col.name)
                            })
                        except Exception as col_err:
                            print(f"[STATS-ERR] Erro ao processar coleção: {col_err}")
                except Exception as list_err:
                    print(f"[STATS-ERR] Erro ao listar coleções no except: {list_err}")
            else:
                print("[STATS-ERR] Client is None no except handler")
            
            return {
                'total_documentos': 0,
                'collection_name': collection_name,
                'embedding_model': self.embeddings_model,
                'last_updated': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                'tipos_documento': {},
                'fontes_dados': {},
                'collections': all_collections,
                'error': str(e)
            }
    
    def delete_collection(self, collection_name: str) -> bool:
        """
        Deleta uma coleção
        
        Args:
            collection_name: Nome da coleção a ser deletada
            
        Returns:
            True se deletou com sucesso
        """
        try:
            print(f"[DELETE] �️ Deletando coleção '{collection_name}'...")
            self.client.delete_collection(collection_name)
            
            # Limpar self.collection se era a que foi deletada
            if self.collection and self.collection.name == collection_name:
                self.collection = None
            
            print(f"[DELETE] ✓ Coleção '{collection_name}' deletada com sucesso!")
            return True
        except Exception as e:
            print(f"[DELETE] ✗ Erro ao deletar coleção: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    def set_collection(self, collection_name: str) -> bool:
        """
        Define a coleção atual para uso
        
        Args:
            collection_name: Nome da coleção
            
        Returns:
            True se definiu com sucesso
        """
        try:
            print(f"🔄 Mudando para coleção '{collection_name}'...")

            # Tenta obter a coleção existente
            try:
                self.collection = self.client.get_collection(
                    name=collection_name,
                    embedding_function=self.embedding_function
                )
                print(f"[OK] Coleção '{collection_name}' definida como atual!")
                return True
            except Exception as e:
                print(f"⚠️ Coleção '{collection_name}' não existe. Não criaremos automaticamente nesta chamada. Erro: {e}")
                # Não criar automaticamente aqui para evitar recriação inesperada após deleção
                return False

        except Exception as e:
            print(f"[OK] Erro ao definir coleção: {e}")
            return False

    def add_document(self, text: str, metadata: dict, id: str = None, collection_name: str = None) -> bool:
        """
        Adiciona um único documento na coleção atual ou na coleção especificada.

        Args:
            text: Conteúdo/texto do documento
            metadata: Metadados do documento
            id: Identificador opcional do documento
            collection_name: Se fornecido, muda para essa coleção (criando-a se necessário)

        Returns:
            True se adicionado com sucesso
        """
        try:
            # Se collection_name informado, tenta definir; se não existir, tenta criar
            if collection_name:
                if not self.set_collection(collection_name):
                    # tenta criar e definir
                    created = self.create_collection(collection_name)
                    if not created:
                        print(f"[OK] Falha ao criar coleção '{collection_name}' para adicionar documento")
                        return False
                    # redefine collection
                    if not self.set_collection(collection_name):
                        print(f"[OK] Falha ao definir coleção '{collection_name}' após criação")
                        return False

            # Se coleção não definida, tenta criar padrão
            if not self.collection:
                if not self.create_collection():
                    print("[OK] Nenhuma coleção definida e falha ao criar padrão")
                    return False

            if not id:
                import time
                id = f"manual_{int(time.time()*1000)}"

            self.collection.add(
                documents=[text],
                metadatas=[metadata],
                ids=[id]
            )

            print(f"[OK] Documento '{id}' adicionado na coleção '{self.collection.name}'")
            return True
        except Exception as e:
            print(f"[OK] Erro ao adicionar documento: {e}")
            return False