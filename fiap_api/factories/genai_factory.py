"""
Fábrica de GenAI - Configuração e inicialização de modelos de geração de texto
Suporta: LMStudio, OpenAI, Azure OpenAI

Integra:
- Cliente GenAI (ChatOpenAI, AzureChatOpenAI)
- Streaming de respostas com contexto ChromaDB
- Construção de prompts com contexto
"""

from typing import Optional, List, Dict, Any, AsyncGenerator
from enum import Enum
import json
import httpx
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
    
    @staticmethod
    async def generate_streaming_response(
        message: str,
        system_prompt: str,
        context: Optional[List[Dict[str, str]]] = None,
        use_chromadb: bool = True,
        chromadb_client = None,
        chromadb_context: str = "",
        chromadb_default_results: int = 50,
        collection_name: Optional[str] = None
    ) -> AsyncGenerator[str, None]:
        """
        Gera resposta especializada com streaming usando GenAI com contexto do ChromaDB
        
        Args:
            message: Mensagem do usuário
            system_prompt: Prompt do sistema
            context: Histórico de mensagens anteriores
            use_chromadb: Se deve usar ChromaDB para contexto
            chromadb_client: Cliente ChromaDB (se use_chromadb=True)
            chromadb_context: Contexto pré-gerado do ChromaDB
            chromadb_default_results: Número de resultados do ChromaDB
            collection_name: Nome da coleção ChromaDB
            
        Yields:
            Chunks da resposta em formato SSE
        """
        # Obter parâmetros de GenAI
        try:
            genai_params = EnvFactory.get_genai_params()
        except Exception as e:
            yield f"data: {json.dumps({'error': f'Erro ao carregar configuração GenAI: {e}'})}\n\n"
            return

        context_from_db = chromadb_context
        
        # Buscar contexto no ChromaDB se solicitado e não fornecido
        if use_chromadb and not chromadb_context:
            if not chromadb_client:
                error_msg = "Banco de dados (ChromaDB) não inicializado"
                yield f"data: {json.dumps({'error': error_msg})}\n\n"
                return
            
            try:
                # Define a coleção a ser usada
                target_collection = collection_name if collection_name else ""
                
                if not target_collection or not target_collection.strip():
                    error_msg = "Nenhuma coleção especificada. Por favor, selecione uma coleção."
                    yield f"data: {json.dumps({'error': error_msg})}\n\n"
                    return
                
                # Tenta definir a coleção
                collection_set = chromadb_client.set_collection(target_collection)
                if not collection_set:
                    error_msg = f"Coleção '{target_collection}' não encontrada no banco de dados"
                    yield f"data: {json.dumps({'error': error_msg})}\n\n"
                    return
                
                try:
                    # Buscar contexto relevante
                    results = chromadb_client.query(message, n_results=chromadb_default_results)
                    
                    if results and len(results) > 0:
                        # Construir contexto com todos os resultados
                        context_parts = [f"[{i}] {result['type'].upper()}: {result['content']} ({result['similarity']:.3f})" for i, result in enumerate(results, 1)]
                        context_from_db = "\n".join(context_parts)
                    else:
                        error_msg = f"Nenhum dado encontrado na coleção '{target_collection}'. A base de dados pode estar vazia ou danificada."
                        yield f"data: {json.dumps({'error': error_msg})}\n\n"
                        return
                        
                except AttributeError as ae:
                    error_msg = f"Base de dados está com problema ao tentar acessar a coleção '{target_collection}'"
                    yield f"data: {json.dumps({'error': error_msg})}\n\n"
                    return
                    
                except Exception as qe:
                    error_msg = f"Erro ao consultar o banco de dados: {str(qe)}"
                    yield f"data: {json.dumps({'error': error_msg})}\n\n"
                    return
                    
            except Exception as e:
                error_msg = f"Erro ao acessar o banco de dados: {str(e)}"
                yield f"data: {json.dumps({'error': error_msg})}\n\n"
                return
        
        # Construir mensagens
        messages = []
        
        # Adicionar prompt do sistema com contexto
        full_system_prompt = ChatResponseGenerator.prepare_system_prompt_with_context(
            system_prompt, 
            context_from_db
        )
        
        messages.append({
            "role": "system",
            "content": full_system_prompt
        })
        
        # Adicionar contexto da conversa
        if context:
            for msg in context[-10:]:  # Limitar histórico
                messages.append({
                    "role": msg.get("role", "user"),
                    "content": msg.get("content", "")
                })
        
        # Adicionar mensagem atual
        messages.append({
            "role": "user",
            "content": message
        })
        
        # Fazer requisição ao GenAI com streaming
        max_tokens_for_response = min(genai_params.max_tokens, 1024)
        
        payload = {
            "model": genai_params.model,
            "messages": messages,
            "stream": True,
            "temperature": genai_params.temperature,
            "max_tokens": max_tokens_for_response
        }
        
        headers = {
            "Content-Type": "application/json",
        }
        
        # Adicionar autorização se houver API key
        if genai_params.api_key:
            headers["Authorization"] = f"Bearer {genai_params.api_key}"
        
        # Construir URL da API
        api_url = f"{genai_params.endpoint}/chat/completions"
        
        async with httpx.AsyncClient(timeout=60.0) as client:
            async with client.stream(
                "POST",
                api_url,
                json=payload,
                headers=headers
            ) as response:
                
                if response.status_code != 200:
                    error_detail = f"Erro HTTP {response.status_code}"
                    yield f"data: {json.dumps({'error': error_detail})}\n\n"
                    return
                
                buffer = ""
                total_content = ""
                
                async for chunk in response.aiter_bytes():
                    try:
                        # Decodifica o chunk
                        chunk_str = chunk.decode('utf-8')
                        buffer += chunk_str
                        
                        # Processa linhas completas
                        lines = buffer.split('\n')
                        buffer = lines[-1]
                        
                        for line in lines[:-1]:
                            line = line.strip()
                            
                            if not line:
                                continue
                            
                            if line.startswith("event:"):
                                continue
                            
                            if line.startswith("data: "):
                                data_str = line[6:]
                                if data_str.strip() == "[DONE]":
                                    return
                                
                                try:
                                    data = json.loads(data_str)
                                    
                                    if "error" in data:
                                        error_msg = data["error"]
                                        if isinstance(error_msg, dict):
                                            error_msg = error_msg.get("message", str(error_msg))
                                        yield f"data: {json.dumps({'error': f'Erro do servidor IA: {error_msg}'})}\n\n"
                                        return
                                    
                                    if "choices" in data and len(data["choices"]) > 0:
                                        delta = data["choices"][0].get("delta", {})
                                        if "content" in delta:
                                            content = delta["content"]
                                            total_content += content
                                            yield f"data: {json.dumps({'content': content})}\n\n"
                                        
                                except json.JSONDecodeError:
                                    continue
                                
                    except UnicodeDecodeError:
                        continue
                    except Exception as e:
                        yield f"data: {json.dumps({'error': f'Erro ao processar streaming: {str(e)}'})}\n\n"
                        return
                
                # Processa buffer final
                if buffer.strip():
                    if buffer.startswith("data: "):
                        data_str = buffer[6:].strip()
                        if data_str and data_str != "[DONE]":
                            try:
                                data = json.loads(data_str)
                                if "choices" in data and len(data["choices"]) > 0:
                                    delta = data["choices"][0].get("delta", {})
                                    if "content" in delta:
                                        content = delta["content"]
                                        total_content += content
                                        yield f"data: {json.dumps({'content': content})}\n\n"
                            except json.JSONDecodeError:
                                pass
                
                if total_content == "":
                    yield f"data: {json.dumps({'error': 'Nenhuma resposta foi recebida da IA. Verifique se o servidor GenAI está respondendo corretamente.'})}\n\n"

