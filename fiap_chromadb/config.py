# ChromaDB Configuration File
# Este arquivo pode ser usado para configurações customizadas

import os

# Configurações do servidor
chroma_server_host = "0.0.0.0"
chroma_server_http_port = 8200
chroma_server_cors_allow_origins = os.getenv("CHROMA_CORS_ORIGINS", "").split(",") if os.getenv("CHROMA_CORS_ORIGINS") else []

# Configurações de autenticação (opcional)
# chroma_server_authn_credentials_file = "/chroma/auth/credentials.txt"
# chroma_server_authn_provider = "basic"

# Configurações de persistência
persist_directory = "/chroma/data"

# Configurações de logging
chroma_server_log_config = {
    "version": 1,
    "disable_existing_loggers": False,
    "formatters": {
        "default": {
            "format": "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
        }
    },
    "handlers": {
        "default": {
            "formatter": "default",
            "class": "logging.StreamHandler",
            "stream": "ext://sys.stdout"
        }
    },
    "root": {
        "level": "INFO",
        "handlers": ["default"]
    }
}

# Configurações de embeddings (opcional)
# Você pode configurar diferentes modelos de embedding aqui
# Por padrão, ChromaDB usa sentence-transformers/all-MiniLM-L6-v2

# Configurações de memória e performance
# chroma_segment_cache_policy = "LRU"
# chroma_segment_cache_max_size_bytes = 1000000000  # 1GB
