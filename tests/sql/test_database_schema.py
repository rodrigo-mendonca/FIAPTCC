import os
import sys
import pytest

# Add the project root to Python path for imports
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

class TestDatabaseSchema:
    """Test cases for database schema defined in init.sql"""
    
    def test_init_sql_contains_mcp_extension(self):
        """Test that init.sql contains MCP extension creation"""
        with open('fiap_sqldb/init.sql', 'r') as f:
            content = f.read()
            
        assert '-- Enable MCP extension' in content
        assert 'CREATE EXTENSION IF NOT EXISTS mcp;' in content
    
    def test_all_tables_created(self):
        """Test that all expected tables are defined in init.sql"""
        with open('fiap_sqldb/init.sql', 'r') as f:
            content = f.read()
        
        # Check for all table definitions
        assert 'CREATE TABLE IF NOT EXISTS usuarios' in content
        assert 'CREATE TABLE IF NOT EXISTS clientes' in content
        assert 'CREATE TABLE IF NOT EXISTS produtos' in content
        assert 'CREATE TABLE IF NOT EXISTS vendas' in content
        assert 'CREATE TABLE IF NOT EXISTS contas_receber' in content
    
    def test_table_structures(self):
        """Test that table structures match expected schema"""
        with open('fiap_sqldb/init.sql', 'r') as f:
            content = f.read()
        
        # Check specific column definitions for usuarios
        assert 'id SERIAL PRIMARY KEY' in content
        assert 'nome TEXT NOT NULL' in content
        assert 'email TEXT UNIQUE NOT NULL' in content
        assert 'telefone TEXT' in content
        assert 'data_cadastro DATE DEFAULT CURRENT_DATE' in content
        
        # Check clientes table structure
        assert 'endereco TEXT' in content
        
        # Check produtos table structure
        assert 'descricao TEXT' in content
        assert 'preco REAL NOT NULL' in content
        assert 'categoria TEXT' in content
        
        # Check vendas table foreign keys and structure
        assert 'cliente_id INTEGER NOT NULL' in content
        assert 'produto_id INTEGER NOT NULL' in content
        assert 'quantidade INTEGER NOT NULL' in content
        assert 'valor_total REAL NOT NULL' in content
        assert 'data_venda DATE DEFAULT CURRENT_DATE' in content
        
        # Check contas_receber table structure
        assert 'cliente_id INTEGER NOT NULL' in content
        assert 'venda_id INTEGER' in content
        assert 'valor_original REAL NOT NULL' in content
        assert 'valor_atual REAL NOT NULL' in content
        assert 'data_emissao TIMESTAMP DEFAULT CURRENT_TIMESTAMP' in content
        assert 'data_vencimento TIMESTAMP NOT NULL' in content
        assert 'status CHAR(1) NOT NULL' in content
        
    def test_foreign_key_constraints(self):
        """Test that foreign key constraints are properly defined"""
        with open('fiap_sqldb/init.sql', 'r') as f:
            content = f.read()
        
        # Check for foreign key references
        assert 'FOREIGN KEY (cliente_id) REFERENCES clientes (id)' in content
        assert 'FOREIGN KEY (produto_id) REFERENCES produtos (id)' in content
        assert 'FOREIGN KEY (venda_id) REFERENCES vendas (id)' in content
    
    def test_database_setup_migration_complete(self):
        """Test that database setup has been properly migrated"""
        # Verify init.sql contains all the schema definitions from database_setup.py
        with open('fiap_sqldb/init.sql', 'r') as f:
            content = f.read()
        
        expected_elements = [
            '-- Enable MCP extension',
            'CREATE EXTENSION IF NOT EXISTS mcp;',
            'CREATE TABLE IF NOT EXISTS usuarios',
            'CREATE TABLE IF NOT EXISTS clientes',
            'CREATE TABLE IF NOT EXISTS produtos',
            'CREATE TABLE IF NOT EXISTS vendas',
            'CREATE TABLE IF NOT EXISTS contas_receber'
        ]
        
        for element in expected_elements:
            assert element in content, f"Missing {element} in init.sql"

if __name__ == "__main__":
    pytest.main([__file__, "-v"])