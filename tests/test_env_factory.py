#!/usr/bin/env python3
"""
Testes para env_factory
Verifica se as variáveis de ambiente obrigatórias estão preenchidas
"""

import os
import sys

# Adiciona o diretório fiap_api ao path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from fiap_api.factories.env_factory import EnvFactory, MissingEnvironmentVariable


if __name__ == "__main__":
    # Teste simples sem pytest
    print("🧪 Testando env_factory...")
    
    # Limpar variáveis
    for key in list(os.environ.keys()):
        if key.startswith("GENAI_") or key.startswith("EMBEDDINGS_"):
            del os.environ[key]
    
    print("\n❌ Teste 1: Verificando erro quando GENAI_PROVIDER está faltando...")
    try:
        EnvFactory.get_genai_params()
        print("   FALHOU: Deveria ter lançado exceção")
    except MissingEnvironmentVariable as e:
        print(f"   ✅ PASSOU: {e}")
    
    print("\n✅ Teste 2: Configurando variáveis e testando...")
    os.environ["GENAI_PROVIDER"] = "lmstudio"
    os.environ["GENAI_MODEL"] = "gpt-3.5-turbo"
    os.environ["GENAI_ENDPOINT"] = "http://localhost:1234"
    os.environ["GENAI_TEMPERATURE"] = "0.7"
    os.environ["GENAI_MAX_TOKENS"] = "2048"
    os.environ["GENAI_TOP_P"] = "0.95"
    
    try:
        params = EnvFactory.get_genai_params()
        print(f"   ✅ PASSOU: {params}")
    except Exception as e:
        print(f"   ❌ FALHOU: {e}")
