"""
Fábrica de Ambiente - Centraliza a leitura de variáveis de ambiente
Fornece métodos específicos para cada factory e métodos comuns compartilhados
"""

import os
from dataclasses import dataclass


class MissingEnvironmentVariable(Exception):
    """Exceção quando uma variável de ambiente obrigatória está faltando"""
    pass


# ==================== DATACLASSES PARA ESTRUTURAÇÃO ====================

@dataclass
class GenAIEnvParams:
    """Parâmetros estruturados de GenAI"""
    provider: str
    model: str
    api_key: str
    endpoint: str
    api_version: str
    temperature: float
    max_tokens: int
    top_p: float


@dataclass
class EmbeddingsEnvParams:
    """Parâmetros estruturados de Embeddings"""
    provider: str
    model: str
    api_key: str
    api_version: str
    endpoint: str


class EnvFactory:
    """Centraliza a leitura de variáveis de ambiente para toda a aplicação"""

    @staticmethod
    def _require_env(var_name: str, custom_message: str = None) -> str:
        """
        Obtém uma variável de ambiente obrigatória
        
        Args:
            var_name: Nome da variável de ambiente
            custom_message: Mensagem customizada de erro (opcional)
            
        Returns:
            Valor da variável
            
        Raises:
            MissingEnvironmentVariable: Se a variável não estiver preenchida
        """
        value = os.getenv(var_name)
        if not value or value.strip() == "":
            message = (
                custom_message or
                f"❌ Erro: Variável de ambiente '{var_name}' é obrigatória e não foi preenchida.\n"
                f"   Configure o arquivo .env com: {var_name}=<valor>"
            )
            raise MissingEnvironmentVariable(message)
        return value

    @staticmethod
    def get_genai_params() -> GenAIEnvParams:
        """
        Retorna todos os parâmetros de GenAI em um objeto estruturado
        
        Returns:
            GenAIEnvParams com todos os parâmetros do GenAI
            
        Raises:
            MissingEnvironmentVariable: Se alguma variável obrigatória estiver faltando
        """
        provider = EnvFactory._require_env("GENAI_PROVIDER").lower()
        model = EnvFactory._require_env("GENAI_MODEL")
        temperature = EnvFactory._require_env("GENAI_TEMPERATURE")
        max_tokens = EnvFactory._require_env("GENAI_MAX_TOKENS")
        top_p = EnvFactory._require_env("GENAI_TOP_P")
        
        # Validar parâmetros específicos por provider
        if provider == "lmstudio":
            endpoint = EnvFactory._require_env(
                "GENAI_ENDPOINT",
                "❌ Erro: GENAI_ENDPOINT é obrigatório para provider 'lmstudio'"
            )
            # LMStudio usa chave padrão ou não usa
            api_key = os.getenv("GENAI_API_KEY", "lm-studio")
            api_version = ""
            
        elif provider == "openai":
            endpoint = EnvFactory._require_env(
                "GENAI_ENDPOINT",
                "❌ Erro: GENAI_ENDPOINT é obrigatório para provider 'openai'"
            )
            api_key = EnvFactory._require_env(
                "GENAI_API_KEY",
                "❌ Erro: GENAI_API_KEY é obrigatório para provider 'openai'"
            )
            api_version = ""
            
        elif provider == "azure":
            endpoint = EnvFactory._require_env(
                "GENAI_ENDPOINT",
                "❌ Erro: GENAI_ENDPOINT é obrigatório para provider 'azure'"
            )
            api_key = EnvFactory._require_env(
                "GENAI_API_KEY",
                "❌ Erro: GENAI_API_KEY é obrigatório para provider 'azure'"
            )
            api_version = EnvFactory._require_env(
                "GENAI_API_VERSION",
                "❌ Erro: GENAI_API_VERSION é obrigatório para provider 'azure'"
            )
        else:
            raise MissingEnvironmentVariable(
                f"❌ Erro: Provider GenAI inválido '{provider}'. "
                f"Use: 'lmstudio', 'openai' ou 'azure'"
            )
        
        return GenAIEnvParams(
            provider=provider,
            model=model,
            api_key=api_key,
            endpoint=endpoint,
            api_version=api_version,
            temperature=float(temperature),
            max_tokens=int(max_tokens),
            top_p=float(top_p),
        )
    
    @staticmethod
    def get_embeddings_params() -> EmbeddingsEnvParams:
        """
        Retorna todos os parâmetros de Embeddings em um objeto estruturado
        
        Returns:
            EmbeddingsEnvParams com todos os parâmetros de Embeddings
            
        Raises:
            MissingEnvironmentVariable: Se alguma variável obrigatória estiver faltando
        """
        provider = EnvFactory._require_env("EMBEDDINGS_PROVIDER").lower()
        model = EnvFactory._require_env("EMBEDDINGS_MODEL")
        
        # Validar parâmetros específicos por provider
        if provider == "lmstudio":
            endpoint = EnvFactory._require_env(
                "EMBEDDINGS_ENDPOINT",
                "❌ Erro: EMBEDDINGS_ENDPOINT é obrigatório para provider 'lmstudio'"
            )
            api_key = os.getenv("EMBEDDINGS_API_KEY", "")
            api_version = ""
            
        elif provider == "openai":
            endpoint = ""
            api_key = EnvFactory._require_env(
                "EMBEDDINGS_API_KEY",
                "❌ Erro: EMBEDDINGS_API_KEY é obrigatório para provider 'openai'"
            )
            api_version = ""
            
        elif provider == "azure":
            endpoint = EnvFactory._require_env(
                "EMBEDDINGS_ENDPOINT",
                "❌ Erro: EMBEDDINGS_ENDPOINT é obrigatório para provider 'azure'"
            )
            api_key = EnvFactory._require_env(
                "EMBEDDINGS_API_KEY",
                "❌ Erro: EMBEDDINGS_API_KEY é obrigatório para provider 'azure'"
            )
            api_version = EnvFactory._require_env(
                "EMBEDDINGS_API_VERSION",
                "❌ Erro: EMBEDDINGS_API_VERSION é obrigatório para provider 'azure'"
            )
        else:
            raise MissingEnvironmentVariable(
                f"❌ Erro: Provider Embeddings inválido '{provider}'. "
                f"Use: 'lmstudio', 'openai' ou 'azure'"
            )
        
        return EmbeddingsEnvParams(
            provider=provider,
            model=model,
            api_key=api_key,
            api_version=api_version,
            endpoint=endpoint,
        )
