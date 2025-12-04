#!/usr/bin/env python3
"""
ChromaDB Client - Métodos de interação com ChromaDB
Este módulo fornece uma interface simplificada para operações com ChromaDB
"""

import os
from chromadb import EmbeddingFunction, Documents
import chromadb
import requests
import json
from typing import List, Dict, Any
import os
from datetime import datetime, timezone, timedelta

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
    def __init__(self, lmstudio_url: str = os.getenv("LMSTUDIO_URL", "http://192.168.50.30:1234")):
        self.lmstudio_url = lmstudio_url
        
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
                    f"{self.lmstudio_url}/v1/embeddings",
                    headers={"Content-Type": "application/json"},
                    json={
                        "input": text,
                        "model": "text-embedding-nomic-embed-text-v1.5"
                    },
                    timeout=30
                )
                
                if response.status_code == 200:
                    result = response.json()
                    embedding = result['data'][0]['embedding']
                    embeddings.append(embedding)
                else:
                    print(f"⚠️ Erro ao gerar embedding para '{text[:50]}...': {response.status_code}")
                    # Fallback: gerar um embedding dummy
                    embeddings.append([0.0] * 768)  # Tamanho típico do nomic-embed
                    
            except Exception as e:
                print(f"⚠️ Erro na requisição de embedding: {e}")
                # Fallback: gerar um embedding dummy
                embeddings.append([0.0] * 768)
                
        return embeddings

class DatabaseDocumentProcessor:
    """
    Processador para converter dados do banco em documentos para ChromaDB
    """
    
    @staticmethod
    def load_json_file(file_path: str) -> Dict:
        """Carrega um arquivo JSON"""
        try:
            # Garantir que o caminho seja relativo à localização do arquivo
            if not os.path.isabs(file_path):
                current_dir = os.path.dirname(os.path.abspath(__file__))
                file_path = os.path.join(current_dir, file_path)
            
            with open(file_path, 'r', encoding='utf-8') as f:
                return json.load(f)
        except Exception as e:
            print(f"❌ Erro ao carregar arquivo {file_path}: {e}")
            return {}
    
    @staticmethod
    def extract_database_structure_documents(database_data: Dict) -> List[Dict[str, Any]]:
        """
        Extrai documentos da estrutura do banco de dados
        
        Args:
            database_data: Dados carregados do database_structure.json
            
        Returns:
            Lista de documentos formatados para ChromaDB
        """
        documents = []
        
        if 'database' not in database_data or 'tables' not in database_data['database']:
            return documents
        
        tables = database_data['database']['tables']
        
        for table_name, table_info in tables.items():
            # Documento principal da tabela
            table_doc = {
                'id': f"table_{table_name}",
                'text': f"Tabela {table_info['name']}: {table_info.get('description', '')}. "
                       f"Área de negócio: {table_info.get('business_area', 'não definida')}. "
                       f"Chave primária: {table_info.get('primary_key', 'não definida')}. "
                       f"Data de criação: {table_info.get('created_at', 'não informada')}.",
                'metadata': {
                    'type': 'table',
                    'table_name': table_name,
                    'business_area': table_info.get('business_area', ''),
                    'primary_key': table_info.get('primary_key', ''),
                    'source': 'database_structure'
                }
            }
            documents.append(table_doc)
            
            # Documentos das colunas
            if 'columns' in table_info:
                for column_name, column_info in table_info['columns'].items():
                    column_doc = {
                        'id': f"column_{table_name}_{column_name}",
                        'text': f"Coluna {column_info['name']} da tabela {table_name}: {column_info.get('description', '')}. "
                               f"Tipo: {column_info.get('type', 'indefinido')}. "
                               f"Pode ser nulo: {'sim' if column_info.get('nullable', True) else 'não'}. "
                               f"Pesquisável: {'sim' if column_info.get('searchable', False) else 'não'}. "
                               f"Chave primária: {'sim' if column_info.get('primary_key', False) else 'não'}. "
                               f"Auto incremento: {'sim' if column_info.get('auto_increment', False) else 'não'}.",
                        'metadata': {
                            'type': 'column',
                            'table_name': table_name,
                            'column_name': column_name,
                            'data_type': column_info.get('type', ''),
                            'nullable': column_info.get('nullable', True),
                            'searchable': column_info.get('searchable', False),
                            'primary_key': column_info.get('primary_key', False),
                            'source': 'database_structure'
                        }
                    }
                    documents.append(column_doc)
            
            # Documentos dos relacionamentos
            if 'relationships' in table_info:
                for i, relationship in enumerate(table_info['relationships']):
                    rel_doc = {
                        'id': f"relationship_{table_name}_{i}",
                        'text': f"Relacionamento da tabela {table_name}: {relationship.get('description', '')}. "
                               f"Tipo: {relationship.get('type', 'indefinido')}. "
                               f"Tabela relacionada: {relationship.get('related_table', 'não informada')}. "
                               f"Chave estrangeira: {relationship.get('foreign_key', 'não informada')}.",
                        'metadata': {
                            'type': 'relationship',
                            'table_name': table_name,
                            'relationship_type': relationship.get('type', ''),
                            'related_table': relationship.get('related_table', ''),
                            'source': 'database_structure'
                        }
                    }
                    documents.append(rel_doc)
            
            # Documentos dos índices
            if 'indexes' in table_info:
                for index in table_info['indexes']:
                    index_doc = {
                        'id': f"index_{table_name}_{index['name']}",
                        'text': f"Índice {index['name']} da tabela {table_name}: {index.get('description', '')}. "
                               f"Colunas: {', '.join(index.get('columns', []))}. "
                               f"Único: {'sim' if index.get('unique', False) else 'não'}.",
                        'metadata': {
                            'type': 'index',
                            'table_name': table_name,
                            'index_name': index['name'],
                            'unique': index.get('unique', False),
                            'columns': ', '.join(index.get('columns', [])),
                            'source': 'database_structure'
                        }
                    }
                    documents.append(index_doc)
        
        return documents
    
    @staticmethod
    def extract_business_rules_documents(rules_data: Dict) -> List[Dict[str, Any]]:
        """
        Extrai documentos das regras de negócio
        
        Args:
            rules_data: Dados carregados do business_rules.json
            
        Returns:
            Lista de documentos de regras
        """
        documents = []
        
        if 'business_rules' not in rules_data or 'tables' not in rules_data['business_rules']:
            return documents
        
        tables = rules_data['business_rules']['tables']
        
        for table_name, table_rules in tables.items():
            # Documento da categoria de regras da tabela
            category_doc = {
                'id': f"rules_category_{table_name}",
                'text': f"Regras de negócio da tabela {table_name}: {table_rules.get('description', '')}. "
                       f"Total de regras: {len(table_rules.get('rules', []))}.",
                'metadata': {
                    'type': 'rules_category',
                    'table_name': table_name,
                    'total_rules': len(table_rules.get('rules', [])),
                    'source': 'business_rules'
                }
            }
            documents.append(category_doc)
            
            # Documentos das regras individuais
            for rule in table_rules.get('rules', []):
                rule_doc = {
                    'id': f"rule_{rule['rule_id']}",
                    'text': f"Regra {rule['rule_id']} - {rule['name']}: {rule.get('description', '')}. "
                           f"Tipo: {rule.get('type', 'indefinido')}. "
                           f"Prioridade: {rule.get('priority', 'indefinida')}. "
                           f"Campos envolvidos: {', '.join(rule.get('fields_involved', []))}. "
                           f"Mensagem de erro: {rule.get('error_message', 'não informada')}.",
                    'metadata': {
                        'type': 'business_rule',
                        'table_name': table_name,
                        'rule_id': rule['rule_id'],
                        'rule_name': rule['name'],
                        'rule_type': rule.get('type', ''),
                        'priority': rule.get('priority', ''),
                        'fields_involved': ', '.join(rule.get('fields_involved', [])),
                        'source': 'business_rules'
                    }
                }
                documents.append(rule_doc)
        
        # Regras globais
        if 'global_rules' in rules_data['business_rules']:
            for rule in rules_data['business_rules']['global_rules']:
                global_rule_doc = {
                    'id': f"global_rule_{rule['rule_id']}",
                    'text': f"Regra global {rule['rule_id']} - {rule['name']}: {rule.get('description', '')}. "
                           f"Tipo: {rule.get('type', 'indefinido')}. "
                           f"Prioridade: {rule.get('priority', 'indefinida')}. "
                           f"Aplica-se a: {rule.get('applies_to', 'não informado')}.",
                    'metadata': {
                        'type': 'global_rule',
                        'rule_id': rule['rule_id'],
                        'rule_name': rule['name'],
                        'rule_type': rule.get('type', ''),
                        'priority': rule.get('priority', ''),
                        'applies_to': rule.get('applies_to', ''),
                        'source': 'business_rules'
                    }
                }
                documents.append(global_rule_doc)
        
        return documents
    
    @staticmethod
    def extract_services_documents(services_data: Dict) -> List[Dict[str, Any]]:
        """
        Extrai documentos dos serviços do sistema
        
        Args:
            services_data: Dados carregados do system_services.json
            
        Returns:
            Lista de documentos de serviços
        """
        documents = []
        
        if 'system_services' not in services_data or 'services' not in services_data['system_services']:
            return documents
        
        services = services_data['system_services']['services']
        
        for service_name, service_info in services.items():
            # Documento principal do serviço
            schedule_info = service_info.get('schedule', {})
            schedule_text = ""
            if schedule_info.get('frequency') == 'daily':
                schedule_text = f"diariamente às {schedule_info.get('time', 'não informado')}"
            elif schedule_info.get('frequency') == 'weekly':
                schedule_text = f"semanalmente às {schedule_info.get('day', 'não informado')} às {schedule_info.get('time', 'não informado')}"
            elif schedule_info.get('frequency') == 'monthly':
                schedule_text = f"mensalmente no dia {schedule_info.get('day', 'não informado')} às {schedule_info.get('time', 'não informado')}"
            elif schedule_info.get('frequency') == 'hourly':
                schedule_text = f"de hora em hora aos {schedule_info.get('minutes', 'não informado')} minutos"
            elif schedule_info.get('frequency') == 'continuous':
                schedule_text = f"continuamente a cada {schedule_info.get('interval', 'não informado')}"
            
            service_doc = {
                'id': f"service_{service_info['service_id']}",
                'text': f"Serviço {service_info['service_id']} - {service_info['name']}: {service_info.get('description', '')}. "
                       f"Categoria: {service_info.get('category', 'não definida')}. "
                       f"Status: {service_info.get('status', 'indefinido')}. "
                       f"Execução: {schedule_text}. "
                       f"Duração estimada: {service_info.get('estimated_duration', 'não informada')}. "
                       f"Prioridade: {service_info.get('priority', 'indefinida')}.",
                'metadata': {
                    'type': 'system_service',
                    'service_id': service_info['service_id'],
                    'service_name': service_info['name'],
                    'category': service_info.get('category', ''),
                    'status': service_info.get('status', ''),
                    'execution_type': service_info.get('execution_type', ''),
                    'frequency': schedule_info.get('frequency', ''),
                    'priority': service_info.get('priority', ''),
                    'source': 'system_services'
                }
            }
            documents.append(service_doc)
            
            # Documento de lógica de negócio do serviço (se existir)
            if 'business_logic' in service_info:
                logic = service_info['business_logic']
                logic_doc = {
                    'id': f"service_logic_{service_info['service_id']}",
                    'text': f"Lógica de negócio do serviço {service_info['name']}: "
                           f"Condição: {logic.get('condition', 'não informada')}. "
                           f"Ação: {logic.get('action', 'não informada')}. "
                           f"Tabela afetada: {logic.get('affected_table', 'não informada')}.",
                    'metadata': {
                        'type': 'service_logic',
                        'service_id': service_info['service_id'],
                        'service_name': service_info['name'],
                        'affected_table': logic.get('affected_table', ''),
                        'source': 'system_services'
                    }
                }
                documents.append(logic_doc)
        
        return documents

class ChromaDBClient:
    """
    Cliente para interação com ChromaDB
    """
    
    def __init__(self, host: str = os.getenv("CHROMADB_HOST", "localhost"), port: int = int(os.getenv("CHROMADB_PORT", "8200")), lmstudio_url: str = os.getenv("LMSTUDIO_URL", "http://192.168.50.30:1234")):
        """
        Inicializa o cliente ChromaDB
        
        Args:
            host: Endereço do servidor ChromaDB
            port: Porta do servidor ChromaDB
            lmstudio_url: URL do LMStudio para embeddings
        """
        self.host = host
        self.port = port
        self.lmstudio_url = lmstudio_url
        self.client = None
        self.collection = None
        self.embedding_function = LMStudioEmbeddingFunction(lmstudio_url)
        self.processor = DatabaseDocumentProcessor()
    
    def connect(self) -> bool:
        """
        Conecta aos serviços ChromaDB e LMStudio
        
        Returns:
            True se conectou com sucesso
        """
        try:
            print(f"🔌 Conectando ao ChromaDB em {self.host}:{self.port}...")
            self.client = chromadb.HttpClient(host=self.host, port=self.port)
            
            # Testa a conexão fazendo um heartbeat
            heartbeat = self.client.heartbeat()
            print(f"✅ Conexão com ChromaDB estabelecida! Heartbeat: {heartbeat}")
            
            # Testa conexão com LMStudio
            print(f"🔌 Testando conexão com LMStudio em {self.lmstudio_url}...")
            test_response = requests.get(f"{self.lmstudio_url}/v1/models", timeout=10)
            if test_response.status_code == 200:
                models = test_response.json()
                print(f"✅ Conexão com LMStudio estabelecida! Modelos disponíveis: {len(models.get('data', []))}")
            else:
                print(f"⚠️ LMStudio respondeu com status {test_response.status_code}")
            
            return True
            
        except Exception as e:
            print(f"❌ Erro ao conectar: {e}")
            return False
    
    def create_collection(self, collection_name: str = "sistema_comercial") -> bool:
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
            
            print(f"✅ Coleção '{collection_name}' pronta com embedding personalizado!")
            return True
            
        except Exception as e:
            print(f"❌ Erro ao criar/obter coleção: {e}")
            return False
    
    def load_and_index_documents(self, data_folder: str = None) -> bool:
        """
        Carrega e indexa documentos de todos os arquivos JSON
        
        Args:
            data_folder: Pasta contendo os arquivos JSON (se None, usa pasta relativa ao script)
            
        Returns:
            True se bem-sucedido
        """
        try:
            # Se não especificado, usa pasta db relativa ao script que chama
            if data_folder is None:
                # Pega o diretório do script que está executando (não este módulo)
                import inspect
                caller_frame = inspect.stack()[1]
                caller_dir = os.path.dirname(os.path.abspath(caller_frame.filename))
                data_folder = os.path.join(caller_dir, "db")
            
            print(f"📄 Carregando dados dos arquivos JSON da pasta {data_folder}...")
            
            # Carrega dados dos 3 arquivos
            structure_data = self.processor.load_json_file(os.path.join(data_folder, "database_structure.json"))
            rules_data = self.processor.load_json_file(os.path.join(data_folder, "business_rules.json"))
            services_data = self.processor.load_json_file(os.path.join(data_folder, "system_services.json"))
            
            if not any([structure_data, rules_data, services_data]):
                print("❌ Nenhum arquivo pôde ser carregado")
                return False
            
            # Extrai documentos de cada arquivo
            structure_docs = self.processor.extract_database_structure_documents(structure_data) if structure_data else []
            rules_docs = self.processor.extract_business_rules_documents(rules_data) if rules_data else []
            services_docs = self.processor.extract_services_documents(services_data) if services_data else []
            
            all_documents = structure_docs + rules_docs + services_docs
            
            if not all_documents:
                print("⚠️ Nenhum documento extraído dos arquivos JSON")
                return False
            
            print(f"📝 Adicionando {len(all_documents)} documentos ao ChromaDB...")
            print(f"   - {len(structure_docs)} documentos de estrutura de banco")
            print(f"   - {len(rules_docs)} documentos de regras de negócio")
            print(f"   - {len(services_docs)} documentos de serviços do sistema")
            
            # Adiciona documentos em lotes para melhor performance
            batch_size = 15
            for i in range(0, len(all_documents), batch_size):
                batch = all_documents[i:i+batch_size]
                
                ids = [doc['id'] for doc in batch]
                texts = [doc['text'] for doc in batch]
                metadatas = [doc['metadata'] for doc in batch]
                
                self.collection.add(
                    documents=texts,
                    metadatas=metadatas,
                    ids=ids
                )
                
                print(f"   ✅ Lote {i//batch_size + 1} adicionado ({len(batch)} docs)")
            
            print(f"✅ Todos os {len(all_documents)} documentos foram adicionados!")
            return True
            
        except Exception as e:
            print(f"❌ Erro ao carregar e adicionar documentos: {e}")
            return False
    
    def query(self, query_text: str, n_results: int = 5, context: str = "all") -> List[Dict]:
        """
        Busca documentos similares na coleção
        
        Args:
            query_text: Texto da consulta
            n_results: Número máximo de resultados
            context: Contexto para filtrar ('all', 'business_rules', 'database_struct', 'system_services')
            
        Returns:
            Lista de documentos encontrados
        """
        try:
            print(f"🔍 Buscando: '{query_text}' no contexto: {context}")
            
            # Configura filtros baseado no contexto
            where_filter = None
            if context != "all":
                # Mapeia os contextos para os tipos de documentos
                context_mapping = {
                    'business_rules': 'business_rule',
                    'database_struct': ['table', 'column', 'database_info'],
                    'system_services': 'service'
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
            
            print(f"✅ Encontrados {len(formatted_results)} resultados")
            return formatted_results
            
        except Exception as e:
            print(f"❌ Erro na busca: {e}")
            return []
    
    def get_collection_stats(self) -> Dict:
        """
        Obtém estatísticas da coleção atual e lista todas as coleções
        
        Returns:
            Dicionário com estatísticas
        """
        try:
            print(f"📊 Obtendo estatísticas da coleção...")
            
            # Obtém todas as coleções disponíveis
            all_collections = []
            try:
                collections = self.client.list_collections()
                for collection in collections:
                    all_collections.append({
                        'name': collection.name,
                        'count': collection.count(),
                        'id': collection.id if hasattr(collection, 'id') else collection.name
                    })
                print(f"📚 Encontradas {len(all_collections)} coleções")
            except Exception as e:
                print(f"⚠️ Erro ao listar coleções: {e}")
                all_collections = []
            
            if not self.collection:
                print("⚠️ Coleção atual não definida. Retornando lista de coleções sem criar padrão.")
                return {
                    'total_documentos': 0,
                    'collection_name': None,
                    'embedding_model': 'text-embedding-nomic-embed-text-v1.5',
                    'last_updated': get_local_datetime().strftime('%Y-%m-%d %H:%M:%S'),
                    'tipos_documento': {},
                    'fontes_dados': {},
                    'collections': all_collections,
                    'error': 'Coleção não definida'
                }
            
            total_docs = self.collection.count()
            print(f"📈 Total de documentos na coleção: {total_docs}")
            
            # Conta documentos por tipo e fonte
            type_counts = {}
            source_counts = {}
            
            if total_docs > 0:
                # Busca uma amostra para contar tipos
                sample_results = self.collection.get(limit=2000, include=['metadatas'])
                
                for metadata in sample_results['metadatas']:
                    doc_type = metadata.get('type', 'unknown')
                    doc_source = metadata.get('source', 'unknown')
                    
                    type_counts[doc_type] = type_counts.get(doc_type, 0) + 1
                    source_counts[doc_source] = source_counts.get(doc_source, 0) + 1
                
                print(f"📋 Tipos de documento: {type_counts}")
                print(f"📄 Fontes de dados: {source_counts}")
            else:
                print("⚠️ Nenhum documento encontrado na coleção")
            
            # Retorna formato esperado pelo frontend
            result = {
                'total_documentos': total_docs,
                'collection_name': self.collection.name if self.collection else 'sistema_comercial',
                'embedding_model': 'text-embedding-nomic-embed-text-v1.5',
                'last_updated': get_local_datetime().strftime('%Y-%m-%d %H:%M:%S'),
                'tipos_documento': type_counts,
                'fontes_dados': source_counts,
                'collections': all_collections
            }
            
            print(f"✅ Estatísticas retornadas: {result}")
            return result
        except Exception as e:
            print(f"❌ Erro ao obter estatísticas: {e}")
            return {
                'total_documentos': 0,
                'collection_name': 'sistema_comercial',
                'embedding_model': 'text-embedding-nomic-embed-text-v1.5',
                'last_updated': get_local_datetime().strftime('%Y-%m-%d %H:%M:%S'),
                'tipos_documento': {},
                'fontes_dados': {},
                'collections': [],
                'error': str(e)
            }
    
    def delete_collection(self, collection_name: str = "sistema_comercial") -> bool:
        """
        Deleta uma coleção
        
        Args:
            collection_name: Nome da coleção a ser deletada
            
        Returns:
            True se deletou com sucesso
        """
        try:
            print(f"🗂️ Antes de deletar, listando coleções disponíveis...")
            try:
                cols_before = [c.name for c in self.client.list_collections()]
                print(f"📚 Coleções antes: {cols_before}")
            except Exception as e:
                print(f"⚠️ Falha ao listar coleções antes da deleção: {e}")

            print(f"🗑️ Solicitando delete da coleção '{collection_name}' via client.delete_collection()...")
            self.client.delete_collection(collection_name)

            try:
                cols_after = [c.name for c in self.client.list_collections()]
                print(f"📚 Coleções depois: {cols_after}")
            except Exception as e:
                print(f"⚠️ Falha ao listar coleções depois da deleção: {e}")

            print(f"✅ Coleção '{collection_name}' deletada (cliente não levantou exceção).")
            return True
        except Exception as e:
            print(f"❌ Erro ao deletar coleção: {e}")
            return False
    
    def update_database_structure(self, database_structure: Dict[str, Any], data_folder: str = None) -> bool:
        """
        Atualiza a estrutura do banco de dados salvando no arquivo JSON e recarregando
        
        Args:
            database_structure: Nova estrutura do banco de dados
            data_folder: Pasta onde salvar o JSON (se None, usa pasta padrão)
            
        Returns:
            True se atualizou com sucesso
        """
        try:
            # Define o caminho do arquivo
            if data_folder is None:
                # Usa o caminho padrão baseado na estrutura do projeto
                import inspect
                caller_frame = inspect.stack()[1]
                caller_dir = os.path.dirname(os.path.abspath(caller_frame.filename))
                # Navega para a pasta de testes do projeto
                project_root = os.path.dirname(caller_dir)
                data_folder = os.path.join(project_root, "tests", "chromadb", "db")
            
            db_path = os.path.join(data_folder, "database_structure.json")
            
            # Salva o JSON atualizado
            with open(db_path, 'w', encoding='utf-8') as f:
                json.dump(database_structure, f, indent=2, ensure_ascii=False)
            
            print(f"💾 Estrutura do banco salva em: {db_path}")
            
            # Recria a coleção e recarrega os dados
            if not self.create_collection():
                return False
                
            if not self.load_and_index_documents(data_folder):
                return False
            
            print("✅ Estrutura do banco atualizada e dados recarregados com sucesso!")
            return True
            
        except Exception as e:
            print(f"❌ Erro ao atualizar estrutura do banco: {e}")
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
                print(f"✅ Coleção '{collection_name}' definida como atual!")
                return True
            except Exception as e:
                print(f"⚠️ Coleção '{collection_name}' não existe. Não criaremos automaticamente nesta chamada. Erro: {e}")
                # Não criar automaticamente aqui para evitar recriação inesperada após deleção
                return False

        except Exception as e:
            print(f"❌ Erro ao definir coleção: {e}")
            return False