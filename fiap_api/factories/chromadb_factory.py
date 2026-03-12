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
    
    @staticmethod
    def extract_database_structure_documents(yaml_files_data: List[Dict]) -> List[Dict[str, Any]]:
        """
        Extrai documentos da estrutura do banco de dados de múltiplos arquivos YAML
        
        Args:
            yaml_files_data: Lista de dicionários carregados dos arquivos YAML de base_dados
            
        Returns:
            Lista de documentos formatados para ChromaDB
        """
        documents = []
        
        # Processa cada arquivo YAML
        for file_data in yaml_files_data:
            if not isinstance(file_data, dict) or 'tabela' not in file_data:
                continue
            
            table_name = file_data.get('tabela', '')
            if not isinstance(table_name, str):
                continue
            
            # Documento principal da tabela
            table_doc = {
                'id': f"table_{table_name.lower().replace(' ', '_')}",
                'text': f"Tabela {table_name}: {file_data.get('descricao_curta', '')}. "
                       f"Database: {file_data.get('database', 'não definido')}. "
                       f"Total de registros: {file_data.get('total_registros', '0')}. "
                       f"Última atualização: {file_data.get('ultima_atualizacao', 'não informada')}.",
                'metadata': {
                    'type': 'table',
                    'table_name': table_name,
                    'database': file_data.get('database', ''),
                    'source': 'database_structure'
                }
            }
            documents.append(table_doc)
            
            # Documentos das colunas importantes
            colunas = file_data.get('colunas_importantes', [])
            if isinstance(colunas, list):
                for col_idx, coluna in enumerate(colunas):
                    if not isinstance(coluna, dict):
                        continue
                    
                    col_name = coluna.get('nome', f'col_{col_idx}')
                    col_doc = {
                        'id': f"column_{table_name.lower()}_{col_name}",
                        'text': f"Coluna {col_name} da tabela {table_name}: {coluna.get('descricao', '')}. "
                               f"Tipo: {coluna.get('tipo', 'indefinido')}. "
                               f"Exemplo: {coluna.get('exemplo_significativo', 'não informado')}.",
                        'metadata': {
                            'type': 'column',
                            'table_name': table_name,
                            'column_name': col_name,
                            'column_type': coluna.get('tipo', ''),
                            'source': 'database_structure'
                        }
                    }
                    documents.append(col_doc)
        
        return documents
    
    @staticmethod
    def extract_business_rules_documents(yaml_files_data: List[Dict]) -> List[Dict[str, Any]]:
        """
        Extrai documentos das regras de negócio de múltiplos arquivos YAML
        
        Args:
            yaml_files_data: Lista de dicionários carregados dos arquivos YAML de regras_negocio
            
        Returns:
            Lista de documentos de regras
        """
        documents = []
        
        # Processa cada arquivo YAML
        for file_data in yaml_files_data:
            if not isinstance(file_data, dict) or 'regras_negocio' not in file_data:
                continue
            
            regras = file_data.get('regras_negocio', [])
            if not isinstance(regras, list):
                continue
            
            # Documentos das regras individuais
            for idx, regra in enumerate(regras):
                if not isinstance(regra, dict):
                    continue
                
                nome = regra.get('nome', f'regra_{idx}')
                regra_doc = {
                    'id': f"regra_{nome.lower().replace(' ', '_')}_{idx}",
                    'text': f"Regra de Negócio: {nome}. "
                           f"Explicação: {regra.get('explicacao', '')}. "
                           f"Tipo: {regra.get('tipo', 'indefinido')}. "
                           f"Prioridade: {regra.get('prioridade', 'indefinida')}.",
                    'metadata': {
                        'type': 'business_rule',
                        'nome_regra': nome,
                        'tipo_regra': regra.get('tipo', ''),
                        'prioridade': regra.get('prioridade', ''),
                        'source': 'business_rules'
                    }
                }
                documents.append(regra_doc)
        
        return documents
    
    @staticmethod
    def extract_services_documents(yaml_files_data: List[Dict]) -> List[Dict[str, Any]]:
        """
        Extrai documentos das rotinas de sistema de múltiplos arquivos YAML
        
        Args:
            yaml_files_data: Lista de dicionários carregados dos arquivos YAML de servicos
            
        Returns:
            Lista de documentos de rotinas
        """
        documents = []
        
        # Processa cada arquivo YAML
        for file_data in yaml_files_data:
            if not isinstance(file_data, dict) or 'rotinas' not in file_data:
                continue
                
            rotinas = file_data.get('rotinas', [])
            if not isinstance(rotinas, list):
                continue
            
            # Documentos das rotinas individuais
            for idx, rotina in enumerate(rotinas):
                if not isinstance(rotina, dict):
                    continue
                
                nome_rotina = rotina.get('nome', f'rotina_{idx}')
                rotina_doc = {
                    'id': f"rotina_{nome_rotina.lower().replace(' ', '_')}_{idx}",
                    'text': f"Rotina: {nome_rotina}. "
                           f"Descrição: {rotina.get('descricao', '')}. "
                           f"Tipo: {rotina.get('tipo_servico', 'indefinido')}. "
                           f"Frequência: {rotina.get('frequencia', 'indefinida')}. "
                           f"Prioridade: {rotina.get('prioridade', 'indefinida')}.",
                    'metadata': {
                        'type': 'rotina_sistema',
                        'nome_rotina': nome_rotina,
                        'tipo_servico': rotina.get('tipo_servico', ''),
                        'frequencia': rotina.get('frequencia', ''),
                        'prioridade': rotina.get('prioridade', ''),
                        'source': 'system_services'
                    }
                }
                documents.append(rotina_doc)
        
        return documents

    @staticmethod
    def extract_user_routines_documents(yaml_files_data: List[Dict]) -> List[Dict[str, Any]]:
        """
        Extrai documentos das rotinas de usuário de múltiplos arquivos YAML
        
        Args:
            yaml_files_data: Lista de dicionários carregados dos arquivos YAML de rotinas_usuario
            
        Returns:
            Lista de documentos de rotinas de usuário
        """
        documents = []
        
        # Processa cada arquivo YAML
        for file_data in yaml_files_data:
            if not isinstance(file_data, dict) or 'rotinas_usuario' not in file_data:
                continue
                
            rotinas = file_data.get('rotinas_usuario', [])
            if not isinstance(rotinas, list):
                continue
            
            # Documentos das rotinas individuais
            for idx, rotina in enumerate(rotinas):
                if not isinstance(rotina, dict):
                    continue
                
                # Monta textos auxiliares
                papeis_raw = rotina.get('papeis_necessarios', [])
                papeis_texto = ", ".join(papeis_raw) if isinstance(papeis_raw, list) else str(papeis_raw)
                
                modulos_raw = rotina.get('modulos_envolvidos', [])
                modulos_texto = ", ".join(modulos_raw) if isinstance(modulos_raw, list) else str(modulos_raw)
                
                nome_rotina = rotina.get('nome', f'rotina_usuario_{idx}')
                rotina_doc = {
                    'id': f"rotina_usuario_{nome_rotina.lower().replace(' ', '_')}_{idx}",
                    'text': f"Rotina de Usuário: {nome_rotina}. "
                           f"Descrição: {rotina.get('descricao', '')}. "
                           f"Frequência: {rotina.get('frequencia', 'indefinida')}. "
                           f"Tempo Estimado: {rotina.get('tempo_estimado', 'não informado')}. "
                           f"Papéis: {papeis_texto}. "
                           f"Módulos: {modulos_texto}.",
                    'metadata': {
                        'type': 'rotina_usuario',
                        'nome_rotina': nome_rotina,
                        'frequencia': rotina.get('frequencia', ''),
                        'tempo_estimado': rotina.get('tempo_estimado', ''),
                        'papeis_necessarios': ','.join(papeis_raw) if isinstance(papeis_raw, list) else str(papeis_raw),
                        'modulos_envolvidos': ','.join(modulos_raw) if isinstance(modulos_raw, list) else str(modulos_raw),
                        'source': 'rotinas_usuario'
                    }
                }
                documents.append(rotina_doc)
        
        return documents
    
    def load_and_index_documents(self, base_path: str, chroma_client: Any) -> Dict[str, Any]:
        """
        Carrega todos os documentos do banco de dados local e os indexa no ChromaDB
        
        Args:
            base_path: Caminho base onde estão os arquivos YAML
            chroma_client: Cliente ChromaDB para inserção
            
        Returns:
            Dicionário com resumo de execução
        """
        summary = {
            'total_documents': 0,
            'collections_created': [],
            'errors': []
        }
        
        # Tipos de documento e seus caminhos
        document_types = {
            'base_dados': 'base_dados',
            'regras_negocio': 'regras_negocio',
            'servicos': 'servicos',
            'rotinas_usuario': 'rotinas_usuario'
        }
        
        # Processa cada tipo de documento
        for doc_type, folder_name in document_types.items():
            try:
                folder_path = f"{base_path}/{folder_name}"
                print(f"Processando documentos {doc_type} de {folder_path}")
                
                # Carrega arquivos YAML
                yaml_files = self.load_yaml_files_from_folder(folder_path)
                
                if not yaml_files:
                    print(f"  Nenhum arquivo YAML encontrado em {folder_path}")
                    continue
                
                # Extrai documentos específicos por tipo
                if doc_type == 'base_dados':
                    documents = self.extract_database_structure_documents(yaml_files)
                elif doc_type == 'regras_negocio':
                    documents = self.extract_business_rules_documents(yaml_files)
                elif doc_type == 'servicos':
                    documents = self.extract_services_documents(yaml_files)
                elif doc_type == 'rotinas_usuario':
                    documents = self.extract_user_routines_documents(yaml_files)
                else:
                    continue
                
                # Cria coleção no ChromaDB se não existir
                collection_name = f"{doc_type}_documents"
                collection = chroma_client.get_or_create_collection(collection_name)
                
                # Adiciona documentos em lotes
                batch_size = 15
                for i in range(0, len(documents), batch_size):
                    batch = documents[i:i+batch_size]
                    
                    # Extrai campos para adição
                    ids = [doc['id'] for doc in batch]
                    texts = [doc['text'] for doc in batch]
                    
                    # Adiciona à coleção
                    collection.add(
                        ids=ids,
                        documents=texts,
                        metadatas=[doc.get('metadata', {}) for doc in batch]
                    )
                
                summary['total_documents'] += len(documents)
                summary['collections_created'].append(collection_name)
                print(f"  ✓ {len(documents)} documentos indexados em {collection_name}")
                
            except Exception as e:
                error_msg = f"Erro ao processar {doc_type}: {str(e)}"
                summary['errors'].append(error_msg)
                print(f"  ✗ {error_msg}")
        
        return summary
    
    def ingest_database_to_collection(self, client: Any, collection_name: str, database_path: str) -> int:
        """
        Ingesta arquivos da base de dados para uma coleção no ChromaDB
        
        Args:
            client: Cliente ChromaDB
            collection_name: Nome da coleção alvo
            database_path: Caminho para os dados da base
            
        Returns:
            Número de documentos adicionados
        """
        documents = self.load_yaml_files_from_folder(database_path)
        
        if not documents:
            return 0
        
        # Extrai documentos desta pasta específica
        extracted = self.extract_database_structure_documents(documents)
        
        # Obtém ou cria coleção
        collection = client.get_or_create_collection(collection_name)
        
        # Adiciona documentos
        batch_size = 15
        total_added = 0
        
        for i in range(0, len(extracted), batch_size):
            batch = extracted[i:i+batch_size]
            ids = [doc['id'] for doc in batch]
            texts = [doc['text'] for doc in batch]
            metadatas = [doc.get('metadata', {}) for doc in batch]
            
            collection.add(ids=ids, documents=texts, metadatas=metadatas)
            total_added += len(batch)
        
        return total_added


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
        
        # Carregar parâmetros do ChromaDB
        if host is None:
            host = os_module.getenv("CHROMADB_HOST")
        if port is None:
            port_str = os_module.getenv("CHROMADB_PORT")
            port = int(port_str)
        
        # Carregar parâmetros de Embeddings
        if endpoint is None:
            endpoint = os_module.getenv("LMSTUDIO_ENDPOINT")
        
        if embeddings_model is None:
            embeddings_model = os_module.getenv("LMSTUDIO_MODEL", "nomic-embed-text")
        
        try:
            embeddings_params = EnvFactory.get_embeddings_params()
            if endpoint is None:
                endpoint = embeddings_params.endpoint
            if embeddings_model is None:
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
    
    def search_database_schema(self, query: str, n_results: int = 10) -> List[Dict]:
        """
        Busca na estrutura do banco de dados
        
        Args:
            query: Texto da busca
            n_results: Número de resultados a retornar
            
        Returns:
            Lista de documentos encontrados
        """
        # Seleciona a coleção de base_dados se disponível
        original_collection = self.collection
        try:
            # Tenta usar coleção de base_dados
            if self.client:
                try:
                    db_collection = self.client.get_collection(
                        name="base_dados_documents",
                        embedding_function=self.embedding_function
                    )
                    self.collection = db_collection
                except:
                    # Se não existir, usa a coleção atual
                    pass
            
            # Executa a busca
            results = self.query(query, n_results=n_results, context="database_struct")
            return results
            
        finally:
            # Restaura coleção original
            self.collection = original_collection
    
    def create_langchain_vectorstore(self):
        """
        Cria um Chroma Vectorstore compatível com LangChain
        Nota: Este é um stub - implementação completa dependeria de langchain estar instalado
        
        Returns:
            Vectorstore ou None se não disponível
        """
        try:
            # Tenta importar LangChain se disponível
            from langchain.vectorstores import Chroma
            
            vectorstore = Chroma(
                collection_name=self.collection.name if self.collection else "documents",
                embedding_function=self.embedding_function,
                client=self.client,
                collection_metadata={"hnsw:space": "cosine"}
            )
            
            print("[OK] LangChain Vectorstore criado com sucesso")
            return vectorstore
            
        except ImportError:
            print("[WARN] LangChain não está instalado, avançando sem integração")
            return None
        except Exception as e:
            print(f"[WARN] Erro ao criar LangChain Vectorstore: {e}")
            return None
    
    def get_retriever(self):
        """
        Obtém um retriever do LangChain para a coleção
        Nota: Este é um stub - implementação completa dependeria de langchain estar instalado
        
        Returns:
            Retriever ou None se não disponível
        """
        try:
            from langchain.vectorstores import Chroma
            
            vectorstore = self.create_langchain_vectorstore()
            if not vectorstore:
                return None
            
            retriever = vectorstore.as_retriever(search_kwargs={"k": 5})
            return retriever
            
        except ImportError:
            return None
        except Exception as e:
            print(f"[WARN] Erro ao obter retriever: {e}")
            return None
    
    def query_with_retriever(self, query: str, retriever=None) -> List[Dict]:
        """
        Executa uma consulta usando LangChain retriever se disponível, senão usa query direto
        
        Args:
            query: Texto da consulta
            retriever: Retriever opcional do LangChain
            
        Returns:
            Lista de documentos encontrados
        """
        try:
            if retriever:
                # Usa o retriever do LangChain
                docs = retriever.get_relevant_documents(query)
                results = []
                for i, doc in enumerate(docs):
                    results.append({
                        'id': f"langchain_{i}",
                        'content': doc.page_content,
                        'metadata': doc.metadata if hasattr(doc, 'metadata') else {},
                        'similarity': 0.0,  # LangChain retriever não retorna scores
                        'source': 'langchain'
                    })
                return results
            else:
                # Fallback para query direto
                return self.query(query)
                
        except Exception as e:
            print(f"[WARN] Erro ao usar retriever: {e}")
            # Fallback para query direto em caso de erro
            return self.query(query)