#!/usr/bin/env python3
"""
Teste completo do ChromaDB usando dados do sistema comercial
Este script utiliza o ChromaDBClient para testar busca semântica
"""

import sys
import os

# Adiciona o diretório raiz do projeto ao Python path
project_root = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
sys.path.insert(0, project_root)

from fiap_chromadb import ChromaDBClient

def main():
    """Função principal que executa os testes"""
    print("🚀 Teste ChromaDB - Sistema Comercial Completo")
    print("📊 Modelo: text-embedding-nomic-embed-text-v1.5")
    print("📄 Fontes: database_structure.json + business_rules.json + system_services.json")
    print("=" * 70)
    
    # Inicializa o cliente ChromaDB
    client = ChromaDBClient()
    
    # Conecta aos serviços
    if not client.connect():
        print("❌ Falha na conexão")
        sys.exit(1)
    
    # Cria coleção
    if not client.create_collection():
        print("❌ Falha ao criar coleção")
        sys.exit(1)
    
    # Carrega e indexa todos os documentos
    if not client.load_and_index_documents():
        print("❌ Falha ao carregar dados")
        sys.exit(1)
    
    # Mostra estatísticas
    print("\n📊 Estatísticas da coleção:")
    stats = client.get_collection_stats()
    for key, value in stats.items():
        if isinstance(value, dict):
            print(f"  {key}:")
            for k, v in value.items():
                print(f"    {k}: {v}")
        else:
            print(f"  {key}: {value}")
    
    # Consultas de teste expandidas incluindo os 3 tipos de dados
    queries = [
        # Estrutura do banco
        "Como funciona o cadastro de clientes?",
        "Quais colunas estão disponíveis na tabela de vendas?",
        "Qual é a chave primária da tabela produtos?",
        "Quais relacionamentos existem entre tabelas?",
        "Que índices foram criados na tabela clientes?",
        
        # Regras de negócio
        "Quais são as regras de validação para CPF?",
        "Como funciona o cálculo de juros e multa?",
        "Que validações existem para produtos?",
        "Quais são as regras para vendas canceladas?",
        "Como é controlado o estoque mínimo?",
        
        # Serviços do sistema
        "Quando é executado o backup do banco de dados?",
        "Como funciona o serviço de contas vencidas?",
        "Que serviços rodam diariamente no sistema?",
        "Como é feito o monitoramento de performance?",
        "Quais relatórios são gerados automaticamente?",
        
        # Consultas integradas
        "Como é calculado o saldo do cliente?",
        "Qual a diferença entre pessoa física e jurídica?",
        "Como funciona o controle de inadimplência?",
        "Quando são aplicados juros por atraso?",
        "Como é feita a baixa automática de estoque?"
    ]
    
    print("\n🔍 Executando consultas sobre o sistema:")
    print("-" * 60)
    
    for query in queries:
        print(f"\n❓ Pergunta: {query}")
        results = client.query(query, n_results=3)
        
        if results:
            for i, result in enumerate(results, 1):
                source_icon = {"database_structure": "🏗️", "business_rules": "📋", "system_services": "⚙️"}.get(result['metadata'].get('source'), "📄")
                print(f"  {i}. {source_icon} [{result['type']}] Similaridade: {result['similarity']:.3f}")
                print(f"     {result['content'][:120]}...")
                if result['metadata'].get('table_name'):
                    print(f"     Tabela: {result['metadata']['table_name']}")
                if result['metadata'].get('source'):
                    print(f"     Fonte: {result['metadata']['source']}")
        else:
            print("  ❌ Nenhum resultado encontrado")
    
    print("\n✅ Teste concluído com sucesso!")
    print("🎯 Sistema comercial completo indexado e consultável via busca semântica")
    print("📚 Estrutura + Regras + Serviços integrados no ChromaDB")
    print("=" * 70)

if __name__ == "__main__":
    main()