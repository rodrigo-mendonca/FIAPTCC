# ChromaDB Module
import sys
import os

# Adiciona o diretório do fiap_api ao path para importar as factories
fiap_api_path = os.path.join(os.path.dirname(__file__), '..', 'fiap_api')
if fiap_api_path not in sys.path:
    sys.path.insert(0, fiap_api_path)

from factories.chromadb_factory import ChromaDBClient

__all__ = ['ChromaDBClient']