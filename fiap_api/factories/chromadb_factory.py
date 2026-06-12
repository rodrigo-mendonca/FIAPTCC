#!/usr/bin/env python3
"""
ChromaDB Factory - Gerenciamento de banco de dados vetorial
Responsável por: criação de coleções, adição, deleção, consultas e processamento de documentos
"""

import os
import math
from chromadb import EmbeddingFunction, Documents
import chromadb
import requests
import yaml
from typing import List, Dict, Any
from datetime import datetime
from pathlib import Path
from .document_optimizer import DocumentOptimizer


class RemoteEmbeddingFunction(EmbeddingFunction):
    """
    Função de embedding via API OpenAI-compatível.

    Serve tanto para o LMStudio (endpoint local) quanto para a Jina AI
    (https://api.jina.ai): ambos expõem POST /v1/embeddings, aceitam
    Authorization: Bearer <api_key> e respondem no formato {"data":[{"embedding":[...]}]}.
    O provider é escolhido apenas pela configuração (endpoint/model/api_key).
    """

    def __init__(self, endpoint: str, model: str, api_key: str = "",
                 embedding_dimension: int = 768, task: str = None,
                 dimensions: int = None, batch_size: int = 32):
        self.endpoint = (endpoint or "").rstrip("/")
        self.model = model
        self.api_key = api_key or ""
        self.embedding_dimension = embedding_dimension
        self.task = task              # opcional (Jina): ex. 'retrieval.passage'
        self.dimensions = dimensions  # opcional (Jina v3): trunca o vetor
        self.batch_size = batch_size

    @staticmethod
    def _normalize(embedding: List[float]) -> List[float]:
        # Norma unitária: com vetores normalizados, distância L2 e cosseno ficam
        # equivalentes, evitando que a escala do vetor distorça a busca.
        norm = math.sqrt(sum(v * v for v in embedding))
        return [v / norm for v in embedding] if norm > 0 else embedding

    def _embed_batch(self, inputs: List[str]) -> List[List[float]]:
        """Gera embeddings de um lote de textos em uma única requisição."""
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"

        payload = {"input": inputs, "model": self.model}
        if self.task:
            payload["task"] = self.task
        if self.dimensions:
            payload["dimensions"] = self.dimensions

        response = requests.post(
            f"{self.endpoint}/v1/embeddings",
            headers=headers,
            json=payload,
            timeout=60,
        )

        if response.status_code != 200:
            raise RuntimeError(
                f"Falha ao gerar embedding ({response.status_code}): {response.text}"
            )

        data = response.json().get("data")
        if not data or len(data) != len(inputs):
            raise RuntimeError(f"Resposta de embedding inválida: {response.text[:300]}")

        # Garante alinhamento com a ordem dos inputs
        data.sort(key=lambda d: d.get("index", 0))
        return [self._normalize(item["embedding"]) for item in data]

    def __call__(self, input: Documents) -> List[List[float]]:
        """
        Gera embeddings para uma lista de textos, em lotes.

        NÃO usa fallback de vetor-zero: um vetor de zeros polui a coleção (fica
        "distante" de tudo e quebra a busca). Em caso de falha, propaga a exceção
        para que indexação/consulta falhem de forma visível.
        """
        embeddings = []
        for i in range(0, len(input), self.batch_size):
            batch = list(input[i:i + self.batch_size])
            try:
                embeddings.extend(self._embed_batch(batch))
            except Exception as e:
                print(f"⚠️ Erro ao gerar embeddings (lote {i // self.batch_size}): {e}")
                raise
        return embeddings


# Compatibilidade retroativa: o nome antigo continua válido.
LMStudioEmbeddingFunction = RemoteEmbeddingFunction


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
        Documentos otimizados para reduzir tokens mantendo relevância
        
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
            
            # Otimiza documento da tabela
            table_doc = DocumentOptimizer.optimize_table_document(
                table_name, 
                file_data,
                max_text_length=200
            )
            
            if table_doc:
                documents.append(table_doc)
            
            # Otimiza documentos das colunas importantes
            colunas = file_data.get('colunas_importantes', [])
            col_docs = DocumentOptimizer.optimize_column_documents(
                table_name,
                colunas,
                max_text_length=150
            )
            documents.extend(col_docs)
        
        return documents
    
    @staticmethod
    def extract_business_rules_documents(yaml_files_data: List[Dict]) -> List[Dict[str, Any]]:
        """
        Extrai documentos das regras de negócio de múltiplos arquivos YAML
        Documentos otimizados para reduzir tokens
        
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
                
                # Otimiza documento da regra
                regra_doc = DocumentOptimizer.optimize_business_rule_document(
                    nome,
                    regra,
                    idx,
                    max_text_length=200
                )
                
                if regra_doc:
                    documents.append(regra_doc)
        
        return documents
    
    @staticmethod
    def extract_services_documents(yaml_files_data: List[Dict]) -> List[Dict[str, Any]]:
        """
        Extrai documentos das rotinas de sistema de múltiplos arquivos YAML
        Documentos otimizados para reduzir tokens
        
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
                
                # Otimiza documento da rotina
                rotina_doc = DocumentOptimizer.optimize_service_document(
                    nome_rotina,
                    rotina,
                    idx,
                    max_text_length=200
                )
                
                if rotina_doc:
                    documents.append(rotina_doc)
        
        return documents

    @staticmethod
    def extract_user_routines_documents(yaml_files_data: List[Dict]) -> List[Dict[str, Any]]:
        """
        Extrai documentos das rotinas de usuário de múltiplos arquivos YAML
        Documentos otimizados para reduzir tokens
        
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
                
                nome_rotina = rotina.get('nome', f'rotina_usuario_{idx}')
                
                # Otimiza documento da rotina
                rotina_doc = DocumentOptimizer.optimize_user_routine_document(
                    nome_rotina,
                    rotina,
                    idx,
                    max_text_length=200
                )
                
                if rotina_doc:
                    documents.append(rotina_doc)
        
        return documents


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
        import os as os_module

        # Carregar parâmetros do ChromaDB
        if host is None:
            host = os_module.getenv("CHROMADB_HOST")
        if port is None:
            port_str = os_module.getenv("CHROMADB_PORT")
            port = int(port_str)

        # Provider de embeddings: 'lmstudio' (default) ou 'jina'
        provider = (os_module.getenv("EMBEDDINGS_PROVIDER") or "lmstudio").lower()
        api_key = os_module.getenv("EMBEDDINGS_API_KEY", "")

        # Endpoint/modelo com defaults por provider (parâmetros explícitos têm prioridade,
        # depois EMBEDDINGS_*, depois LMSTUDIO_* por compatibilidade)
        if provider == "jina":
            endpoint = endpoint or os_module.getenv("EMBEDDINGS_ENDPOINT") or "https://api.jina.ai"
            embeddings_model = embeddings_model or os_module.getenv("EMBEDDINGS_MODEL") or "jina-embeddings-v3"
        else:  # lmstudio
            endpoint = endpoint or os_module.getenv("EMBEDDINGS_ENDPOINT") or os_module.getenv("LMSTUDIO_ENDPOINT")
            embeddings_model = (embeddings_model or os_module.getenv("EMBEDDINGS_MODEL")
                                or os_module.getenv("LMSTUDIO_MODEL") or "nomic-embed-text")

        # Parâmetros opcionais (úteis para a Jina v3)
        task = os_module.getenv("EMBEDDINGS_TASK") or None
        dims_env = os_module.getenv("EMBEDDINGS_DIMENSIONS")
        dimensions = int(dims_env) if dims_env and dims_env.isdigit() else None

        self.host = host
        self.port = port
        self.provider = provider
        self.lmstudio_url = endpoint
        self.embeddings_model = embeddings_model
        self.client = None
        self.collection = None
        self.embedding_function = RemoteEmbeddingFunction(
            endpoint=endpoint,
            model=embeddings_model,
            api_key=api_key,
            task=task,
            dimensions=dimensions,
        )
        self.processor = DatabaseDocumentProcessor()

        print(f"[INIT] ChromaDBClient inicializado: host={self.host}, port={self.port}, "
              f"provider={self.provider}, endpoint={endpoint}, model={self.embeddings_model}")
    
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
                metadata={
                    "description": "Sistema comercial - estrutura, regras e serviços",
                    # Usa distância de cosseno (0 = idêntico, 2 = oposto) em vez do
                    # default L2, que com embeddings de alta dimensão produz valores
                    # grandes e faz documentos relevantes parecerem "distantes".
                    "hnsw:space": "cosine",
                },
            )
            
            print(f"[OK] Coleção '{collection_name}' pronta com embedding personalizado!")
            return True
            
        except Exception as e:
            print(f"[OK] Erro ao criar/obter coleção: {e}")
            return False
    
    def query(self, query_text: str, n_results: int, context: str = "all", similarity_threshold: float = 0.0) -> List[Dict]:
        """
        Busca documentos similares na coleção, ordenados por similaridade.

        Args:
            query_text: Texto da consulta
            n_results: Número máximo de resultados (None ou <= 0 para sem limite)
            context: Contexto para filtrar ('all', 'business_rules', 'database_struct', 'system_services', 'user_routines')
            similarity_threshold: Similaridade MÍNIMA (0.0 a 1.0) para um documento
                ser incluído. 1.0 = idêntico, 0.0 = sem relação. Padrão 0.0
                (não descarta nada; apenas ordena por relevância).

        Returns:
            Lista de documentos com similarity >= threshold, ordenada da maior
            para a menor similaridade.
        """
        try:
            print(f"[OK] Buscando: '{query_text}' no contexto: {context}")
            print(f"[OK] Similaridade mínima exigida: {similarity_threshold}")
            
            # Configura filtros baseado no contexto
            where_filter = None
            if context != "all":
                # Mapeia os contextos para os tipos de documentos. Inclui todas as
                # variações realmente gravadas pelos otimizadores e pelo upload
                # (ex.: serviços salvos como 'rotina_sistema'/'service', colunas
                # como 'column'/'field') para o filtro não descartar tudo.
                context_mapping = {
                    'business_rules': ['business_rule'],
                    'database_struct': ['table', 'column', 'field', 'database_info'],
                    'system_services': ['service', 'rotina_sistema'],
                    'user_routines': ['rotina_usuario']
                }

                if context in context_mapping:
                    where_filter = {"type": {"$in": context_mapping[context]}}
            
            # Se n_results for None ou <= 0, interpretamos como 'sem limite' e usamos o total de documentos
            if n_results is None or (isinstance(n_results, int) and n_results <= 0):
                try:
                    total_docs = self.collection.count() if self.collection else 0
                    # segurança: se coleção vazia ou count falhar, usa um limite alto
                    n_results = total_docs if total_docs and total_docs > 0 else 10000
                except Exception:
                    n_results = 10000
            print(f"[DEBUG] Número de resultados solicitado: {n_results}")
            # Executa a query com ou sem filtro
            query_params = {
                "query_texts": [query_text],
                "n_results": n_results,
                "include": ['documents', 'metadatas', 'distances']
            }
            
            if where_filter:
                query_params["where"] = where_filter
                print(f"📋 Aplicando filtro de contexto: {where_filter}")
            
            results = self.collection.query(**query_params)

            formatted_results = []
            total_returned = 0

            if results['documents'] and results['documents'][0]:
                total_returned = len(results['documents'][0])

                for i, doc in enumerate(results['documents'][0]):
                    distance = results['distances'][0][i]

                    # Converte distância de cosseno em similaridade de cosseno padrão.
                    # distância 0 -> 1.0 (idêntico); distância 1 -> 0.0 (sem relação).
                    similarity = round(max(0.0, 1.0 - distance), 3)

                    print(f"[OK] Documento {results['ids'][0][i]} | distância: {round(distance, 4)} | similaridade: {similarity}")

                    # Mantém apenas documentos com similaridade >= limiar
                    if similarity >= similarity_threshold:
                        result = {
                            'id': results['ids'][0][i],
                            'content': doc,
                            'metadata': results['metadatas'][0][i],
                            'similarity': similarity,
                            'distance': round(distance, 4),
                            'type': results['metadatas'][0][i].get('type', 'unknown')
                        }
                        formatted_results.append(result)

            # Ordena do mais relevante (maior similaridade) para o menos relevante
            formatted_results.sort(key=lambda r: r['similarity'], reverse=True)

            print(f"[OK] ChromaDB retornou {total_returned} documentos")
            print(f"[OK] Após filtro (similarity >= {similarity_threshold}): {len(formatted_results)} resultados relevantes")

            if total_returned > 0 and len(formatted_results) == 0:
                print(f"⚠️  AVISO: Nenhum documento passou no filtro. Considere reduzir o similarity_threshold.")

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

    def reindex_collection_from_folder(self, base_path: str, collection_name: str) -> Dict[str, Any]:
        """
        Reconstrói UMA coleção a partir de todos os YAML do disco.

        Ao contrário de DatabaseDocumentProcessor.load_and_index_documents (que
        cria uma coleção por tipo), este método consolida base_dados,
        regras_negocio, servicos e rotinas_usuario na MESMA coleção nomeada,
        que é o modelo usado pela busca do app. Usa a embedding_function do
        cliente e a métrica de cosseno.

        Args:
            base_path: Pasta raiz que contém as subpastas de YAML
            collection_name: Coleção alvo (será criada se não existir)

        Returns:
            Resumo com total de documentos e contagem por tipo
        """
        summary = {'total_documents': 0, 'by_type': {}, 'errors': []}

        if not self.create_collection(collection_name):
            summary['errors'].append(f"Falha ao criar/obter coleção '{collection_name}'")
            return summary

        extractors = {
            'base_dados': self.processor.extract_database_structure_documents,
            'regras_negocio': self.processor.extract_business_rules_documents,
            'servicos': self.processor.extract_services_documents,
            'rotinas_usuario': self.processor.extract_user_routines_documents,
        }

        for doc_type, extractor in extractors.items():
            try:
                folder_path = os.path.join(base_path, doc_type)
                yaml_files = self.processor.load_yaml_files_from_folder(folder_path)
                if not yaml_files:
                    continue

                documents = extractor(yaml_files)
                if not documents:
                    continue

                batch_size = 15
                for i in range(0, len(documents), batch_size):
                    batch = documents[i:i + batch_size]
                    self.collection.add(
                        ids=[doc['id'] for doc in batch],
                        documents=[doc['text'] for doc in batch],
                        metadatas=[doc.get('metadata', {}) for doc in batch],
                    )

                summary['by_type'][doc_type] = len(documents)
                summary['total_documents'] += len(documents)
                print(f"[REINDEX] {len(documents)} documentos de '{doc_type}' indexados em '{collection_name}'")
            except Exception as e:
                err = f"Erro ao reindexar '{doc_type}': {e}"
                summary['errors'].append(err)
                print(f"[REINDEX] {err}")

        return summary
