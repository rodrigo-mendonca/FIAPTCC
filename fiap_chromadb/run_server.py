#!/usr/bin/env python
"""
ChromaDB Local Server Runner
Inicia o servidor ChromaDB localmente sem Docker
"""

import subprocess
import sys
import os

def run_chromadb_server():
    """Inicia o servidor ChromaDB"""
    try:
        # Usar chroma run se disponível
        cmd = [sys.executable, "-m", "chromadb.server"]
        print(f"Iniciando ChromaDB na porta 8200...")
        print(f"Comando: {' '.join(cmd)}")
        
        # Define variáveis de ambiente
        env = os.environ.copy()
        env['CHROMA_HOST'] = '0.0.0.0'
        env['CHROMA_PORT'] = '8200'
        env['ALLOW_RESET'] = 'true'
        env['ALLOW_DELETE_COLLECTIONS'] = 'true'
        
        # Tenta usar chroma run
        result = subprocess.run(
            ["chroma", "run", "--port", "8200", "--host", "0.0.0.0"],
            env=env
        )
        sys.exit(result.returncode)
    except FileNotFoundError:
        # Se chroma CLI não estiver disponível, tenta com chromadb server
        try:
            print("chroma CLI não encontrada, tentando chromadb.server...")
            result = subprocess.run(
                [sys.executable, "-m", "chromadb.server"],
                env=env
            )
            sys.exit(result.returncode)
        except Exception as e:
            print(f"Erro ao iniciar ChromaDB: {e}")
            print("\nPor favor, instale chromadb com:")
            print("  pip install chromadb")
            sys.exit(1)

if __name__ == "__main__":
    run_chromadb_server()
