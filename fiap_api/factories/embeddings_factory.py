"""
Fábrica de Embeddings - Configuração e inicialização de modelos de embedding
Suporta: LMStudio, OpenAI, Azure OpenAI

Fornece embeddings compatíveis com LangChain e ChromaDB via Chroma.from_documents()
"""

from typing import Optional, Dict, Any
from .env_factory import EnvFactory

class EmbeddingsConfig:
    """Configuração de Embeddings a partir de variáveis de ambiente (padrão unificado)"""
    
    def __init__(self):
        params = EnvFactory.get_embeddings_params()
        
        # Provider: lmstudio, openai, azure
        self.provider = params.provider
        
        # Configuração unificada
        self.model = params.model
        self.api_key = params.api_key
        self.api_version = params.api_version
        self.endpoint = params.endpoint
        
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
        """Cria embeddings via LMStudio (usando OpenAI API compatível)"""
        from langchain_openai import OpenAIEmbeddings
        
        return OpenAIEmbeddings(
            base_url=config.endpoint,
            api_key="not-needed",
            model=config.model,
        )
    
    @staticmethod
    def _create_openai(config: EmbeddingsConfig):
        """Cria embeddings OpenAI"""
        from langchain_openai import OpenAIEmbeddings
        
        return OpenAIEmbeddings(
            api_key=config.api_key,
            model=config.model,
        )
    
    @staticmethod
    def _create_azure(config: EmbeddingsConfig):
        """Cria embeddings Azure OpenAI"""
        from langchain_openai import OpenAIEmbeddings
        
        return OpenAIEmbeddings(
            api_type="azure",
            api_key=config.api_key,
            api_base=config.endpoint,
            api_version=config.api_version,
            model=config.model,
        )

