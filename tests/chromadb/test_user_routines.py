#!/usr/bin/env python3
"""
Teste das Rotinas de Usuário com ChromaDB
Demonstra como as rotinas de usuário são carregadas e consultadas
"""

import sys
import os

# Adiciona path da API
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'fiap_api'))

from factories.chromadb_factory import ChromaDBClient

def test_user_routines():
    """Testa carregamento e consulta de rotinas de usuário"""
    
    print("=" * 80)
    print("🧪 TESTE: ROTINAS DE USUÁRIO NO CHROMADB")
    print("=" * 80)
    
    # Inicializa cliente ChromaDB
    client = ChromaDBClient()
    
    # Carrega dados
    print("\n📚 Carregando documentos...")
    data_folder = os.path.join(os.path.dirname(__file__), 'data')
    
    if client.load_and_index_documents(data_folder):
        print("✅ Documentos carregados com sucesso!")
        
        # Teste 1: Buscar rotinas de vendas
        print("\n" + "=" * 80)
        print("📋 TESTE 1: Buscar rotinas de processamento de pedidos")
        print("=" * 80)
        
        results = client.query(
            "como processar um novo pedido de cliente",
            n_results=3,
            context="user_routines"
        )
        
        for i, result in enumerate(results, 1):
            print(f"\n🔹 Resultado {i}:")
            print(f"   ID: {result.get('id')}")
            print(f"   Similaridade: {result.get('similarity', 0):.2%}")
            print(f"   Tipo: {result.get('metadata', {}).get('type')}")
            print(f"   Rotina: {result.get('metadata', {}).get('nome_rotina')}")
            print(f"   Frequência: {result.get('metadata', {}).get('frequencia')}")
            print(f"   Preview: {result.get('content', '')[:200]}...")
        
        # Teste 2: Buscar rotinas financeiras
        print("\n" + "=" * 80)
        print("📋 TESTE 2: Buscar rotinas financeiras")
        print("=" * 80)
        
        results = client.query(
            "como fazer reconciliação bancária e processar folha de pagamento",
            n_results=3,
            context="user_routines"
        )
        
        for i, result in enumerate(results, 1):
            print(f"\n🔹 Resultado {i}:")
            print(f"   Rotina: {result.get('metadata', {}).get('nome_rotina')}")
            print(f"   Frequência: {result.get('metadata', {}).get('frequencia')}")
            print(f"   Tempo: {result.get('metadata', {}).get('tempo_estimado')}")
            print(f"   Papéis: {result.get('metadata', {}).get('papeis_necessarios')}")
            print(f"   Similaridade: {result.get('similarity', 0):.2%}")
        
        # Teste 3: Buscar rotinas de estoque
        print("\n" + "=" * 80)
        print("📋 TESTE 3: Buscar rotinas de gerenciamento de estoque")
        print("=" * 80)
        
        results = client.query(
            "contagem cíclica de inventário e gerar ordem de compra",
            n_results=3,
            context="user_routines"
        )
        
        for i, result in enumerate(results, 1):
            print(f"\n🔹 Resultado {i}:")
            print(f"   Rotina: {result.get('metadata', {}).get('nome_rotina')}")
            print(f"   Frequência: {result.get('metadata', {}).get('frequencia')}")
            print(f"   Módulos: {result.get('metadata', {}).get('modulos_envolvidos')}")
            print(f"   Similaridade: {result.get('similarity', 0):.2%}")
        
        # Teste 4: Buscar em todos os contextos
        print("\n" + "=" * 80)
        print("📋 TESTE 4: Buscar em TODOS os contextos (estrutura + regras + serviços + rotinas)")
        print("=" * 80)
        
        results = client.query(
            "processar pedido e atualizar estoque",
            n_results=5,
            context="all"
        )
        
        print(f"\n📊 Total de resultados: {len(results)}")
        for i, result in enumerate(results, 1):
            tipo = result.get('metadata', {}).get('type', 'desconhecido')
            print(f"   {i}. [{tipo:20}] {result.get('id', '')[:50]} - Similaridade: {result.get('similarity', 0):.2%}")
        
        print("\n" + "=" * 80)
        print("✅ TESTES CONCLUÍDOS COM SUCESSO!")
        print("=" * 80)
        
    else:
        print("❌ Falha ao carregar documentos")
        return False
    
    return True

if __name__ == "__main__":
    success = test_user_routines()
    sys.exit(0 if success else 1)
