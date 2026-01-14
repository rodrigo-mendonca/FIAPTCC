"""
Fábrica de Embeddings - Configuração e inicialização de modelos de embedding
Suporta: LMStudio, OpenAI, Azure OpenAI
"""

import os
from typing import Optional, List, Dict, Any
import hashlib
import re
import unicodedata

class EmbeddingsConfig:
    """Configuração de Embeddings a partir de variáveis de ambiente (padrão unificado)"""
    
    def __init__(self):
        # Provider: lmstudio, openai, azure
        self.provider = os.getenv("EMBEDDINGS_PROVIDER", "lmstudio").lower()
        
        # Configuração unificada
        self.model = os.getenv("EMBEDDINGS_MODEL", "text-embedding-nomic-embed-text-v1.5")
        self.api_key = os.getenv("EMBEDDINGS_API_KEY", "")
        self.api_version = os.getenv("EMBEDDINGS_API_VERSION", "")
        self.endpoint = os.getenv("EMBEDDINGS_ENDPOINT", "")
        
        # Fallback para variáveis antigas (compatibilidade)
        if not self.endpoint and self.provider == "lmstudio":
            self.endpoint = os.getenv("LMSTUDIO_URL", "http://192.168.50.30:1234")
        
        if not self.endpoint and self.provider == "azure":
            self.endpoint = os.getenv("AZURE_OPENAI_ENDPOINT", "")
        
        if not self.api_key and self.provider == "openai":
            self.api_key = os.getenv("OPENAI_API_KEY", "")
        
        if not self.api_key and self.provider == "azure":
            self.api_key = os.getenv("AZURE_OPENAI_API_KEY", "")
        
        if not self.api_version and self.provider == "azure":
            self.api_version = os.getenv("AZURE_OPENAI_API_VERSION", "2024-02-15-preview")
        
    def validate(self):
        """Valida a configuração"""
        if self.provider == "openai" and not self.api_key:
            raise ValueError("EMBEDDINGS_API_KEY é obrigatório para provider 'openai'")
        
        if self.provider == "azure" and (not self.api_key or not self.endpoint):
            raise ValueError("EMBEDDINGS_API_KEY e EMBEDDINGS_ENDPOINT são obrigatórios para provider 'azure'")
        
        if self.provider not in ["lmstudio", "openai", "azure"]:
            raise ValueError(f"Provider '{self.provider}' não suportado. Use: lmstudio, openai, azure")


class EmbeddingsFactory:
    """Factory para criar instância de Embeddings"""
    
    @staticmethod
    def create():
        """Cria instância de Embeddings baseado na configuração"""
        config = EmbeddingsConfig()
        config.validate()
        
        if config.provider == "lmstudio":
            return EmbeddingsFactory._create_lmstudio(config)
        elif config.provider == "openai":
            return EmbeddingsFactory._create_openai(config)
        elif config.provider == "azure":
            return EmbeddingsFactory._create_azure(config)
    
    @staticmethod
    def _create_lmstudio(config: EmbeddingsConfig):
        """Cria embeddings via LMStudio (usando OpenAI API)"""
        from langchain.embeddings import OpenAIEmbeddings
        
        return OpenAIEmbeddings(
            base_url=config.endpoint + "/v1",
            api_key="not-needed",
            model=config.model,
        )
    
    @staticmethod
    def _create_openai(config: EmbeddingsConfig):
        """Cria embeddings OpenAI"""
        from langchain.embeddings import OpenAIEmbeddings
        
        return OpenAIEmbeddings(
            api_key=config.api_key,
            model=config.model,
        )
    
    @staticmethod
    def _create_azure(config: EmbeddingsConfig):
        """Cria embeddings Azure OpenAI"""
        from langchain.embeddings import OpenAIEmbeddings
        
        return OpenAIEmbeddings(
            api_type="azure",
            api_key=config.api_key,
            api_base=config.endpoint,
            api_version=config.api_version,
            model=config.model,
        )


class EmbeddingsUtility:
    """Utilitários para criar embeddings e textos pesquisáveis em português"""
    
    @staticmethod
    def create_simple_embeddings(texts: List[str]) -> List[List[float]]:
        """Cria embeddings simples para textos em português usando análise de características linguísticas"""
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
    
    @staticmethod
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
    
    @staticmethod
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
