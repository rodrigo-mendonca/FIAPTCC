#!/usr/bin/env python3
"""
ChromaDB Factory - Gerenciamento de banco de dados vetorial
Responsável por: criação de coleções, adição, deleção, consultas e processamento de documentos
"""

import os
from chromadb import EmbeddingFunction, Documents
import chromadb
import requests
import yaml
from typing import List, Dict, Any, Optional
from datetime import datetime, timezone, timedelta
from pathlib import Path


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
    def load_yaml_file(file_path: str) -> Dict:
        """Carrega um arquivo YAML"""
        try:
            # Garantir que o caminho seja relativo à localização do arquivo
            if not os.path.isabs(file_path):
                current_dir = os.path.dirname(os.path.abspath(__file__))
                file_path = os.path.join(current_dir, file_path)
            
            with open(file_path, 'r', encoding='utf-8') as f:
                return yaml.safe_load(f) or {}
        except Exception as e:
            print(f"[OK] Erro ao carregar arquivo YAML {file_path}: {e}")
            return {}
    
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
                
                print(f"   📄 Carregando {yaml_file.name}...")
                content = DatabaseDocumentProcessor.load_yaml_file(str(yaml_file))
                if content:
                    data_list.append(content)
            
            print(f"   [OK] {len(data_list)} arquivos YAML carregados")
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
            # Validar que é um dicionário
            if not isinstance(file_data, dict):
                print(f"[EXTRACT] ⚠️ Ignorando dados inválidos (não é dict): {type(file_data)}")
                continue
            
            # O arquivo YAML tem 'tabela' como string (nome da tabela) no nível raiz
            if 'tabela' not in file_data:
                continue
            
            table_name = file_data.get('tabela', '')
            
            # Se table_name não é string, pular
            if not isinstance(table_name, str):
                print(f"[EXTRACT] ⚠️ 'tabela' é {type(table_name)}, esperado string. Pulando...")
                continue
            
            # Extrai as chaves (chaves primária e estrangeiras)
            chaves = file_data.get('chaves', {})
            pk = ""
            if isinstance(chaves, dict):
                pk_list = chaves.get('pk', [])
                if isinstance(pk_list, list) and pk_list:
                    pk = ", ".join(pk_list)
            
            # Documento principal da tabela
            table_doc = {
                'id': f"table_{table_name.lower().replace(' ', '_')}",
                'text': f"Tabela {table_name}: {file_data.get('descricao_curta', '')}. "
                       f"Database: {file_data.get('database', 'não definido')}. "
                       f"Total de registros: {file_data.get('total_registros', '0')}. "
                       f"Última atualização: {file_data.get('ultima_atualizacao', 'não informada')}. "
                       f"Chave primária: {pk if pk else 'não definida'}.",
                'metadata': {
                    'type': 'table',
                    'table_name': table_name,
                    'database': file_data.get('database', ''),
                    'primary_key': pk,
                    'total_records': str(file_data.get('total_registros', '0')),
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
                               f"Nulo: {coluna.get('nulo', 'não definido')}. "
                               f"Exemplo: {coluna.get('exemplo_significativo', 'não informado')}.",
                        'metadata': {
                            'type': 'column',
                            'table_name': table_name,
                            'column_name': col_name,
                            'column_type': coluna.get('tipo', ''),
                            'nullable': str(coluna.get('nulo', '')),
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
            # Validar que é um dicionário
            if not isinstance(file_data, dict):
                print(f"[EXTRACT] ⚠️ Ignorando arquivo de regras inválido (não é dict): {type(file_data)}")
                continue
            
            if 'regras_negocio' not in file_data:
                continue
            
            regras = file_data.get('regras_negocio', [])
            
            # Validar que regras é uma lista
            if not isinstance(regras, list):
                print(f"[EXTRACT] ⚠️ regras_negocio não é lista: {type(regras)}")
                continue
            
            # Documentos das regras individuais
            for idx, regra in enumerate(regras):
                if not isinstance(regra, dict):
                    print(f"[EXTRACT] ⚠️ Regra {idx} não é dict: {type(regra)}")
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
            # Validar que é um dicionário
            if not isinstance(file_data, dict):
                print(f"[EXTRACT] ⚠️ Ignorando arquivo de serviços inválido (não é dict): {type(file_data)}")
                continue
                
            if 'rotinas' not in file_data:
                continue
            
            rotinas = file_data.get('rotinas', [])
            
            # Validar que rotinas é uma lista
            if not isinstance(rotinas, list):
                print(f"[EXTRACT] ⚠️ rotinas não é lista: {type(rotinas)}")
                continue
            
            # Documentos das rotinas individuais
            for idx, rotina in enumerate(rotinas):
                # Validar que é um dicionário
                if not isinstance(rotina, dict):
                    print(f"[EXTRACT] ⚠️ Serviço {idx} não é dict: {type(rotina)}")
                    continue
                
                # Monta texto de frequência de forma legível
                frequencia = rotina.get('frequencia', 'indefinida')
                detalhes_frequencia = []
                
                if frequencia == 'diaria' and rotina.get('horario'):
                    detalhes_frequencia.append(f"diariamente às {rotina.get('horario')}")
                elif frequencia == 'semanal' and rotina.get('dia_semana'):
                    detalhes_frequencia.append(f"semanalmente às {rotina.get('dia_semana')}")
                elif frequencia == 'mensal' and rotina.get('dia_mes'):
                    detalhes_frequencia.append(f"mensalmente no dia {rotina.get('dia_mes')}")
                elif frequencia == 'a_cada_hora':
                    if rotina.get('minuto'):
                        detalhes_frequencia.append(f"a cada hora no minuto {rotina.get('minuto')}")
                    else:
                        detalhes_frequencia.append("a cada hora")
                elif frequencia == 'a_cada_6_horas':
                    detalhes_frequencia.append("a cada 6 horas")
                elif frequencia == 'a_cada_30_minutos':
                    detalhes_frequencia.append("a cada 30 minutos")
                elif frequencia == 'a_cada_15_minutos':
                    detalhes_frequencia.append("a cada 15 minutos")
                elif frequencia == 'a_cada_5_minutos':
                    detalhes_frequencia.append("a cada 5 minutos")
                elif frequencia == 'tempo_real':
                    detalhes_frequencia.append("tempo real")
                elif frequencia == 'continu':
                    detalhes_frequencia.append("contínuo")
                elif frequencia == 'sob_demanda':
                    detalhes_frequencia.append("sob demanda")
                
                frequencia_texto = detalhes_frequencia[0] if detalhes_frequencia else frequencia
                
                nome_rotina = rotina.get('nome', f'rotina_{idx}')
                rotina_doc = {
                    'id': f"rotina_{nome_rotina.lower().replace(' ', '_')}_{idx}",
                    'text': f"Rotina: {nome_rotina}. "
                           f"Descrição: {rotina.get('descricao', '')}. "
                           f"Tipo: {rotina.get('tipo_servico', 'indefinido')}. "
                           f"Frequência: {frequencia_texto}. "
                           f"Duração: {rotina.get('duracao_estimada', 'não informada')}. "
                           f"Prioridade: {rotina.get('prioridade', 'indefinida')}.",
                    'metadata': {
                        'type': 'rotina_sistema',
                        'nome_rotina': nome_rotina,
                        'tipo_servico': rotina.get('tipo_servico', ''),
                        'frequencia': rotina.get('frequencia', ''),
                        'prioridade': rotina.get('prioridade', ''),
                        'duracao_estimada': rotina.get('duracao_estimada', ''),
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
            # Validar que é um dicionário
            if not isinstance(file_data, dict):
                print(f"[EXTRACT] ⚠️ Ignorando arquivo de rotinas de usuário inválido (não é dict): {type(file_data)}")
                continue
                
            if 'rotinas_usuario' not in file_data:
                continue
            
            rotinas = file_data.get('rotinas_usuario', [])
            
            # Validar que rotinas é uma lista
            if not isinstance(rotinas, list):
                print(f"[EXTRACT] ⚠️ rotinas_usuario não é lista: {type(rotinas)}")
                continue
            
            # Documentos das rotinas individuais
            for idx, rotina in enumerate(rotinas):
                # Validar que é um dicionário
                if not isinstance(rotina, dict):
                    print(f"[EXTRACT] ⚠️ Rotina de usuário {idx} não é dict: {type(rotina)}")
                    continue
                
                # Monta texto com informações dos passos
                passos_texto = ""
                if rotina.get('passos'):
                    passos_lista = []
                    for passo in rotina.get('passos', []):
                        if isinstance(passo, dict):
                            passos_lista.append(f"{passo.get('passo', '')}: {passo.get('detalhes', '')}")
                    passos_texto = " | ".join(passos_lista)
                
                # Monta texto com papéis necessários
                papeis_raw = rotina.get('papeis_necessarios', [])
                papeis_texto = ", ".join(papeis_raw) if isinstance(papeis_raw, list) else str(papeis_raw)
                
                # Monta texto com módulos envolvidos
                modulos_raw = rotina.get('modulos_envolvidos', [])
                modulos_texto = ", ".join(modulos_raw) if isinstance(modulos_raw, list) else str(modulos_raw)
                
                # Monta texto com validações
                validacoes_raw = rotina.get('validacoes_importantes', [])
                validacoes_texto = " | ".join(validacoes_raw) if isinstance(validacoes_raw, list) else str(validacoes_raw)
                
                # Monta texto com dicas
                dicas_raw = rotina.get('dicas', [])
                dicas_texto = " | ".join(dicas_raw) if isinstance(dicas_raw, list) else str(dicas_raw)
                
                nome_rotina = rotina.get('nome', f'rotina_usuario_{idx}')
                rotina_doc = {
                    'id': f"rotina_usuario_{nome_rotina.lower().replace(' ', '_')}_{idx}",
                    'text': f"Rotina de Usuário: {nome_rotina}. "
                           f"Descrição: {rotina.get('descricao', '')}. "
                           f"Frequência: {rotina.get('frequencia', 'indefinida')}. "
                           f"Tempo Estimado: {rotina.get('tempo_estimado', 'não informado')}. "
                           f"Papéis: {papeis_texto}. "
                           f"Módulos: {modulos_texto}. "
                           f"Passos: {passos_texto}. "
                           f"Validações: {validacoes_texto}. "
                           f"Dicas: {dicas_texto}.",
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
        # Se parâmetros não foram fornecidos, carregar do env
        if endpoint is None or embeddings_model is None:
            from .env_factory import EnvFactory
            try:
                embeddings_params = EnvFactory.get_embeddings_params()
                if endpoint is None:
                    endpoint = embeddings_params.endpoint
                if embeddings_model is None:
                    embeddings_model = embeddings_params.model
            except Exception as e:
                print(f"[WARN] Erro ao carregar parâmetros de embeddings do env: {e}")
                endpoint = endpoint or "http://localhost:1234"
                embeddings_model = embeddings_model or "text-embedding-nomic-embed-text-v1.5"
        
        self.host = host or "localhost"
        self.port = port or 8200
        self.lmstudio_url = endpoint
        self.embeddings_model = embeddings_model
        self.client = None
        self.collection = None
        self.embedding_function = LMStudioEmbeddingFunction(self.lmstudio_url, self.embeddings_model)
        self.processor = DatabaseDocumentProcessor()
    
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
                # Mesmo assim, continua a tentar outras coisas
            
            # Testa conexão com LMStudio
            print(f"[CONNECT] Testando conexão com LMStudio em {self.lmstudio_url}...")
            try:
                test_response = requests.get(f"{self.lmstudio_url}/models", timeout=10)
                if test_response.status_code == 200:
                    models = test_response.json()
                    print(f"[CONNECT] ✓ LMStudio disponível! Modelos: {len(models.get('data', []))}")
                else:
                    print(f"[CONNECT] ⚠️ LMStudio status {test_response.status_code}")
            except Exception as lm_error:
                print(f"[CONNECT] ⚠️ LMStudio não respondeu: {lm_error}")
            
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
    
    def load_and_index_documents(self, data_folder: str = None) -> bool:
        """
        Carrega e indexa documentos de todos os arquivos YAML das pastas
        
        Args:
            data_folder: Pasta pai contendo as subpastas de dados (base_dados, regras_negocio, servicos)
            
        Returns:
            True se bem-sucedido
        """
        try:
            print(f"\n[INDEX] === INICIANDO INDEXAÇÃO DE DOCUMENTOS ===")
            
            # Verificar se collection está selecionada
            if not self.collection:
                print("[INDEX] ✗ ERRO: Nenhuma coleção selecionada!")
                return False
            
            print(f"[INDEX] Coleção alvo: {self.collection.name}")
            
            # Se não especificado, usa pasta relativa ao script que chama
            if data_folder is None:
                # Pega o diretório do script que está executando (não este módulo)
                import inspect
                caller_frame = inspect.stack()[1]
                caller_dir = os.path.dirname(os.path.abspath(caller_frame.filename))
                data_folder = os.path.join(caller_dir, "data")
            
            print(f"[INDEX] Pasta de dados: {data_folder}")
            
            # Verificar se a pasta existe
            if not os.path.exists(data_folder):
                print(f"[INDEX] ✗ Pasta não encontrada: {data_folder}")
                return False
            
            # Carrega dados dos 4 tipos de pastas
            print("\n[INDEX] 📂 Carregando base_dados...")
            db_structure_folder = os.path.join(data_folder, "base_dados")
            structure_yaml_files = self.processor.load_yaml_files_from_folder(db_structure_folder)
            print(f"[INDEX]    → {len(structure_yaml_files)} arquivos encontrados")
            
            print("\n[INDEX] 📂 Carregando regras_negocio...")
            business_rules_folder = os.path.join(data_folder, "regras_negocio")
            rules_yaml_files = self.processor.load_yaml_files_from_folder(business_rules_folder)
            print(f"[INDEX]    → {len(rules_yaml_files)} arquivos encontrados")
            
            print("\n[INDEX] 📂 Carregando servicos...")
            services_folder = os.path.join(data_folder, "servicos")
            services_yaml_files = self.processor.load_yaml_files_from_folder(services_folder)
            print(f"[INDEX]    → {len(services_yaml_files)} arquivos encontrados")
            
            print("\n[INDEX] 📂 Carregando rotinas_usuario...")
            user_routines_folder = os.path.join(data_folder, "rotinas_usuario")
            user_routines_yaml_files = self.processor.load_yaml_files_from_folder(user_routines_folder)
            print(f"[INDEX]    → {len(user_routines_yaml_files)} arquivos encontrados")
            
            if not any([structure_yaml_files, rules_yaml_files, services_yaml_files, user_routines_yaml_files]):
                print("[INDEX] ✗ Nenhum arquivo YAML encontrado em nenhuma das pastas!")
                return False
            
            # Extrai documentos de cada pasta
            print("\n[INDEX] 🔄 Extraindo documentos...")
            structure_docs = self.processor.extract_database_structure_documents(structure_yaml_files) if structure_yaml_files else []
            rules_docs = self.processor.extract_business_rules_documents(rules_yaml_files) if rules_yaml_files else []
            services_docs = self.processor.extract_services_documents(services_yaml_files) if services_yaml_files else []
            user_routines_docs = self.processor.extract_user_routines_documents(user_routines_yaml_files) if user_routines_yaml_files else []
            
            all_documents = structure_docs + rules_docs + services_docs + user_routines_docs
            
            if not all_documents:
                print("[INDEX] ✗ Nenhum documento foi extraído dos arquivos YAML")
                return False
            
            print(f"\n[INDEX] ✓ {len(all_documents)} documentos extraídos:")
            print(f"[INDEX]    - {len(structure_docs)} de estrutura de banco")
            print(f"[INDEX]    - {len(rules_docs)} de regras de negócio")
            print(f"[INDEX]    - {len(services_docs)} de serviços do sistema")
            print(f"[INDEX]    - {len(user_routines_docs)} de rotinas de usuário")
            
            # Adiciona documentos em lotes para melhor performance
            batch_size = 15
            total_added = 0
            print(f"\n[INDEX] 📤 Adicionando documentos em lotes...")
            
            for i in range(0, len(all_documents), batch_size):
                batch = all_documents[i:i+batch_size]
                
                ids = [doc['id'] for doc in batch]
                texts = [doc['text'] for doc in batch]
                metadatas = [doc['metadata'] for doc in batch]
                
                try:
                    self.collection.add(
                        documents=texts,
                        metadatas=metadatas,
                        ids=ids
                    )
                    total_added += len(batch)
                    batch_num = (i // batch_size) + 1
                    print(f"[INDEX]    ✓ Lote {batch_num}: {len(batch)} documentos adicionados")
                except Exception as batch_error:
                    print(f"[INDEX]    ✗ Erro ao adicionar lote {(i//batch_size + 1)}: {batch_error}")
                    # Continua com os próximos lotes
            
            print(f"\n[INDEX] ✓✓✓ SUCESSO! {total_added} documentos foram indexados na coleção '{self.collection.name}'!")
            return True
            
        except Exception as e:
            print(f"[INDEX] ✗✗✗ ERRO ao carregar e indexar documentos: {e}")
            import traceback
            traceback.print_exc()
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
            print(f"[DEBUG] Iniciando get_collection_stats...")
            
            # Dados básicos
            collection_name = str(self.collection.name) if self.collection else None
            print(f"[DEBUG] Nome da coleção: {collection_name}")
            
            # Tenta contar documentos
            total_docs = 0
            try:
                if self.collection:
                    total_docs = int(self.collection.count())
                    print(f"[DEBUG] Total de docs: {total_docs}")
            except Exception as count_err:
                print(f"[DEBUG] Erro ao contar docs: {count_err}")
            
            # Tipos e fontes (simples)
            type_counts = {}
            source_counts = {}
            
            if total_docs > 0 and self.collection:
                try:
                    print(f"[DEBUG] Buscando amostra de metadados...")
                    sample = self.collection.get(limit=100, include=['metadatas'])
                    print(f"[DEBUG] Amostra recebida")
                    
                    if sample and 'metadatas' in sample:
                        for metadata in sample['metadatas']:
                            if metadata and isinstance(metadata, dict):
                                doc_type = str(metadata.get('type', 'unknown'))
                                doc_source = str(metadata.get('source', 'unknown'))
                                type_counts[doc_type] = type_counts.get(doc_type, 0) + 1
                                source_counts[doc_source] = source_counts.get(doc_source, 0) + 1
                except Exception as sample_err:
                    print(f"[DEBUG] Erro ao processar amostra: {sample_err}")
            
            # SEMPRE lista todas as coleções, independente de ter uma selecionada
            print(f"[DEBUG] Listando coleções...")
            all_collections = []
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
                        print(f"[DEBUG] Erro ao processar coleção {col}: {col_err}")
                print(f"[DEBUG] Coleções encontradas: {len(all_collections)}")
            except Exception as list_err:
                print(f"[DEBUG] Erro ao listar coleções: {list_err}")
            
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
            
            print(f"[DEBUG] Stats prontos, retornando: {result}")
            return result
            
        except Exception as e:
            print(f"[ERRO] Exceção em get_collection_stats: {e}")
            import traceback
            traceback.print_exc()
            
            # Tenta retornar algo útil mesmo em erro
            try:
                collection_name = str(self.collection.name) if self.collection else None
            except:
                collection_name = None
            
            # Tenta listar coleções mesmo em erro
            all_collections = []
            try:
                collections = self.client.list_collections()
                for col in collections:
                    try:
                        all_collections.append({
                            'name': str(col.name),
                            'count': int(col.count()),
                            'id': str(col.name)
                        })
                    except:
                        pass
            except:
                pass
            
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
    
    def update_database_structure(self, table_name: str, table_structure: Dict[str, Any], data_folder: str = None) -> bool:
        """
        Atualiza a estrutura de uma tabela específica salvando em arquivo YAML
        
        Args:
            table_name: Nome da tabela a ser atualizada
            table_structure: Nova estrutura da tabela
            data_folder: Pasta onde salvar o YAML (se None, usa pasta padrão)
            
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
                data_folder = os.path.join(project_root, "tests", "chromadb", "data", "database_structure")
            
            # Cria o conteúdo YAML
            yaml_content = {
                'tabela': table_structure
            }
            
            table_file_path = os.path.join(data_folder, f"{table_name}.yaml")
            
            # Salva o YAML atualizado
            os.makedirs(data_folder, exist_ok=True)
            with open(table_file_path, 'w', encoding='utf-8') as f:
                yaml.dump(yaml_content, f, default_flow_style=False, allow_unicode=True, sort_keys=False)
            
            print(f"[OK] Estrutura da tabela {table_name} salva em: {table_file_path}")
            print("[OK] Estrutura da tabela atualizada com sucesso!")
            return True
            
        except Exception as e:
            print(f"[OK] Erro ao atualizar estrutura da tabela: {e}")
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

    def add_documents(self, documents: List[Dict[str, Any]], collection_name: str = None) -> bool:
        """
        Adiciona múltiplos documentos ao ChromaDB na coleção atual ou em `collection_name`.

        Cada documento deve ter as chaves: 'id', 'text' e 'metadata'.
        """
        try:
            if collection_name:
                if not self.set_collection(collection_name):
                    created = self.create_collection(collection_name)
                    if not created:
                        print(f"[OK] Falha ao criar coleção '{collection_name}' para adicionar documentos")
                        return False
                    if not self.set_collection(collection_name):
                        print(f"[OK] Falha ao definir coleção '{collection_name}' após criação")
                        return False

            if not self.collection:
                if not self.create_collection():
                    print("[OK] Nenhuma coleção definida e falha ao criar padrão")
                    return False

            batch_size = 15
            for i in range(0, len(documents), batch_size):
                batch = documents[i:i+batch_size]
                ids = [doc.get('id') for doc in batch]
                texts = [doc.get('text') for doc in batch]
                metadatas = [doc.get('metadata') for doc in batch]

                self.collection.add(
                    documents=texts,
                    metadatas=metadatas,
                    ids=ids
                )

            print(f"[OK] {len(documents)} documentos adicionados na coleção '{self.collection.name}'")
            return True
        except Exception as e:
            print(f"[OK] Erro ao adicionar documentos em lote: {e}")
            return False

    def ingest_database_to_collection(self, database_data: Dict[str, Any], collection_name: str) -> bool:
        """
        Faz ingestão completa de dados de database no ChromaDB
        
        Args:
            database_data: Dados do banco de dados para ingerir
            collection_name: Nome da coleção para armazenar
            
        Returns:
            True se ingestão foi bem-sucedida
        """
        try:
            from factories.embeddings_factory import EmbeddingsUtility
            
            # Tentar deletar coleção existente
            try:
                self.delete_collection(collection_name)
            except:
                pass
            
            # Criar nova coleção
            if not self.create_collection(collection_name):
                return False
            
            documents = []
            metadatas = []
            ids = []
            
            db_info = database_data.get("database", {})
            
            # 1. Ingerir informações das tabelas
            tables = db_info.get("tables", {})
            for table_name, table_data in tables.items():
                # Documento da tabela
                table_text = EmbeddingsUtility.create_searchable_text(table_data, table_name)
                documents.append(table_text)
                metadatas.append({
                    "type": "table",
                    "table_name": table_name,
                    "business_area": table_data.get("area_negocio", "general"),
                    "importance": "high"
                })
                ids.append(f"table_{table_name}")
                
                # 2. Ingerir campos importantes
                fields = table_data.get("fields", {})
                for field_name, field_data in fields.items():
                    if field_data.get("pesquisavel", True) or field_data.get("tipo") in ["decimal", "integer"] or "saldo" in field_name.lower() or "valor" in field_name.lower():
                        field_text = EmbeddingsUtility.create_field_searchable_text(field_name, field_data, table_name)
                        documents.append(field_text)
                        metadatas.append({
                            "type": "field",
                            "table_name": table_name,
                            "field_name": field_name,
                            "data_type": field_data.get("tipo", "unknown"),
                            "business_area": table_data.get("area_negocio", "general"),
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
            embeddings = EmbeddingsUtility.create_simple_embeddings(documents)
            
            self.collection.add(
                documents=documents,
                metadatas=metadatas,
                ids=ids,
                embeddings=embeddings
            )
            
            print(f"[OK] Ingestão concluída: {len(documents)} documentos adicionados")
            return True
            
        except Exception as e:
            print(f"[OK] Erro na ingestão: {e}")
            import traceback
            traceback.print_exc()
            return False

    def search_database_schema(self, query: str, limit: int = 5) -> Dict[str, Any]:
        """
        Busca informações no schema da database
        
        Args:
            query: Texto da consulta
            limit: Número máximo de resultados
            
        Returns:
            Dicionário com resultados da busca
        """
        try:
            if not self.collection:
                return {
                    "results": [],
                    "total_found": 0,
                    "query": query,
                    "error": "Nenhuma coleção definida"
                }
            
            from factories.embeddings_factory import EmbeddingsUtility
            
            # Gerar embedding para a query
            query_embeddings = EmbeddingsUtility.create_simple_embeddings([query])
            
            # Buscar documentos similares
            results = self.collection.query(
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
                "collection": self.collection.name if self.collection else None
            }
            
        except Exception as e:
            print(f"[OK] Erro na busca: {e}")
            return {
                "results": [],
                "total_found": 0,
                "query": query,
                "error": str(e)
            }
