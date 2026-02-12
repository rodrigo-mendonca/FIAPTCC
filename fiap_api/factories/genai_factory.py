"""
Fábrica de GenAI - Configuração e inicialização de modelos de geração de texto
Suporta: LMStudio, OpenAI, Azure OpenAI
"""

from typing import Optional
from enum import Enum
import json
from .env_factory import EnvFactory


class GenAIProvider(str, Enum):
    """Provedores suportados de GenIA"""
    LMSTUDIO = "lmstudio"
    OPENAI = "openai"
    ANTHROPIC = "anthropic"
    OLLAMA = "ollama"
    GOOGLE = "google"
    AZURE = "azure"
    HUGGINGFACE = "huggingface"
    CUSTOM = "custom"


class GenAIConfig:
    """Configuração centralizada de modelos de GenIA"""
    
    def __init__(self):
        params = EnvFactory.get_genai_params()
        
        # Provider padrão
        self.provider = params.provider
        
        # Configurações gerais
        self.model_name = params.model
        self.api_key = params.api_key
        self.base_url = params.endpoint
        self.api_version = params.api_version
        
        # Configurações de LMStudio (padrão)
        self.lmstudio_url = params.endpoint if params.provider == "lmstudio" else ""
        self.lmstudio_api_key = params.api_key if params.provider == "lmstudio" else ""
        
        # Configurações de temperatura e outros parâmetros
        self.temperature = params.temperature
        self.max_tokens = params.max_tokens
        self.top_p = params.top_p
    
    @property
    def is_lmstudio(self) -> bool:
        """Verifica se está usando LMStudio"""
        return self.provider == GenAIProvider.LMSTUDIO.value or self.provider == "lmstudio"
    
    @property
    def is_openai(self) -> bool:
        """Verifica se está usando OpenAI"""
        return self.provider == GenAIProvider.OPENAI.value or self.provider == "openai"
    
    @property
    def is_anthropic(self) -> bool:
        """Verifica se está usando Anthropic"""
        return self.provider == GenAIProvider.ANTHROPIC.value or self.provider == "anthropic"
    
    @property
    def is_ollama(self) -> bool:
        """Verifica se está usando Ollama"""
        return self.provider == GenAIProvider.OLLAMA.value or self.provider == "ollama"
    
    @property
    def is_azure(self) -> bool:
        """Verifica se está usando Azure"""
        return self.provider == GenAIProvider.AZURE.value or self.provider == "azure"
    
    def get_api_url(self) -> str:
        """Retorna a URL da API baseada no provider"""
        if self.is_lmstudio:
            return f"{self.lmstudio_url}/v1"
        elif self.base_url:
            return self.base_url
        else:
            raise ValueError(f"Base URL não configurada para o provider {self.provider}")
    
    def get_headers(self) -> dict:
        """Retorna headers padrão para requisições da API"""
        headers = {
            "Content-Type": "application/json",
        }
        
        if self.is_lmstudio:
            headers["Authorization"] = f"Bearer {self.lmstudio_api_key}"
        elif self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        
        return headers
    
    def validate(self):
        """Valida a configuração"""
        if self.is_openai and not self.api_key:
            raise ValueError("GENAI_API_KEY é obrigatório para provider 'openai'")
        
        if self.is_azure and (not self.api_key or not self.base_url):
            raise ValueError("GENAI_API_KEY e base_url são obrigatórios para provider 'azure'")
        
        if self.provider not in ["lmstudio", "openai", "azure", "anthropic", "ollama", "google", "huggingface", "custom"]:
            raise ValueError(f"Provider '{self.provider}' não suportado.")
    
    def __repr__(self) -> str:
        return (
            f"GenAIConfig(provider={self.provider}, "
            f"model={self.model_name})"
        )


class GenAIFactory:
    """Factory para criar instância de GenAI"""
    
    @staticmethod
    def create():
        """Cria instância de GenAI baseado na configuração"""
        config = GenAIConfig()
        config.validate()
        
        if config.is_lmstudio:
            return GenAIFactory._create_lmstudio(config)
        elif config.is_openai:
            return GenAIFactory._create_openai(config)
        elif config.is_azure:
            return GenAIFactory._create_azure(config)
    
    @staticmethod
    def _create_lmstudio(config: GenAIConfig):
        """Cria cliente LMStudio via OpenAI API"""
        from langchain_openai import ChatOpenAI
        
        return ChatOpenAI(
            base_url=config.get_api_url(),
            api_key="not-needed",
            model=config.model_name,
            temperature=config.temperature,
            max_tokens=config.max_tokens,
            top_p=config.top_p,
        )
    
    @staticmethod
    def _create_openai(config: GenAIConfig):
        """Cria cliente OpenAI"""
        from langchain_openai import ChatOpenAI
        
        return ChatOpenAI(
            api_key=config.api_key,
            model=config.model_name,
            temperature=config.temperature,
            max_tokens=config.max_tokens,
            top_p=config.top_p,
        )
    
    @staticmethod
    def _create_azure(config: GenAIConfig):
        """Cria cliente Azure OpenAI"""
        from langchain_openai import AzureChatOpenAI
        
        return AzureChatOpenAI(
            azure_endpoint=config.base_url,
            azure_deployment=config.model_name,
            api_version=config.api_version,
            api_key=config.api_key,
            temperature=config.temperature,
            max_tokens=config.max_tokens,
            top_p=config.top_p,
        )


class ChatResponseGenerator:
    """Gerador de respostas de chat com suporte a streaming"""
    
    @staticmethod
    def build_messages(user_message: str, system_prompt: str, context_history: list = None) -> list:
        """
        Constrói lista de mensagens para a API do LMStudio
        
        Args:
            user_message: Mensagem do usuário
            system_prompt: Prompt do sistema
            context_history: Histórico de mensagens anteriores
            
        Returns:
            Lista de mensagens formatadas
        """
        messages = []
        
        # Adicionar prompt do sistema
        messages.append({
            "role": "system",
            "content": system_prompt
        })
        
        # Adicionar histórico de contexto
        if context_history:
            for msg in context_history[-10:]:  # Limitar a últimas 10 mensagens
                if isinstance(msg, dict):
                    messages.append(msg)
                else:
                    # Se for objeto com atributos (ex: Pydantic model)
                    messages.append({
                        "role": getattr(msg, 'role', 'user'),
                        "content": getattr(msg, 'content', str(msg))
                    })
        
        # Adicionar mensagem atual
        messages.append({
            "role": "user",
            "content": user_message
        })
        
        return messages
    
    @staticmethod
    def prepare_system_prompt_with_context(base_prompt: str, chromadb_context: str = "") -> str:
        """
        Prepara prompt do sistema com contexto do ChromaDB
        
        Args:
            base_prompt: Prompt base do sistema
            chromadb_context: Contexto extraído do ChromaDB
            
        Returns:
            Prompt completo com contexto
        """
        full_prompt = base_prompt
        
        if chromadb_context:
            full_prompt += f"\n\nCONTEXTO DA BASE DE CONHECIMENTO:\n{chromadb_context}"
        
        return full_prompt
