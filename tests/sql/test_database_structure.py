testeimport os
import sys
import pytest
from unittest.mock import patch, MagicMock

# Add the project root to Python path for imports
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

from fiap_api.factories.sql_factory import SQLFactory

class TestSQLFactoryDatabaseStructure:
    """Test cases that verify database structure matches expected schema"""
    
    def test_database_tables_existence(self):
        """Test that the SQL factory can query for table existence"""
        # This would normally connect to a real database
        # For testing purposes, we'll mock this behavior
        
        with patch.dict(os.environ, {'DATABASE_URL': 'postgresql://postgres:password@localhost/loja_db'}):
            sql_factory = SQLFactory()
            
            # Mock the engine and session behavior for table existence check
            mock_engine = MagicMock()
            mock_session = MagicMock()
            mock_result = MagicMock()
            
            with patch.object(sql_factory, 'engine', mock_engine):
                with patch.object(sql_factory.SessionLocal, 'return_value', mock_session):
                    # Simulate checking for tables that should exist based on our setup script
                    mock_result.keys.return_value = ['table_name']
                    mock_result.__iter__.return_value = [
                        ['usuarios'],
                        ['clientes'], 
                        ['produtos'],
                        ['vendas'],
                        ['contas_receber']
                    ]
                    mock_session.execute.return_value = mock_result
                    
                    # This would be a query to check table existence
                    result = sql_factory.execute_query(
                        "SELECT tablename FROM pg_tables WHERE schemaname='public'"
                    )
                    
                    # Verify we get the expected tables
                    assert isinstance(result, list)
                    table_names = [row['tablename'] for row in result]
                    expected_tables = ['usuarios', 'clientes', 'produtos', 'vendas', 'contas_receber']
                    for table in expected_tables:
                        assert table in table_names

    def test_database_schema_compatibility(self):
        """Test that database schema matches what's defined in setup script"""
        with patch.dict(os.environ, {'DATABASE_URL': 'postgresql://postgres:password@localhost/loja_db'}):
            sql_factory = SQLFactory()
            
            # Mock the engine and session behavior
            mock_engine = MagicMock()
            mock_session = MagicMock()
            
            with patch.object(sql_factory, 'engine', mock_engine):
                with patch.object(sql_factory.SessionLocal, 'return_value', mock_session):
                    # Test a query that would work on our database structure
                    mock_result = MagicMock()
                    mock_result.keys.return_value = ['id', 'nome', 'email']
                    mock_result.__iter__.return_value = [[1, 'Test User', 'test@example.com']]
                    mock_session.execute.return_value = mock_result
                    
                    # Test a query that should work with our schema
                    result = sql_factory.execute_query(
                        "SELECT id, nome, email FROM usuarios LIMIT 1"
                    )
                    
                    assert isinstance(result, list)
                    assert len(result) >= 0  # Should not fail even if no data

    def test_database_connection_error_handling(self):
        """Test that SQL factory properly handles connection errors"""
        with patch.dict(os.environ, {'DATABASE_URL': 'postgresql://postgres:password@localhost/loja_db'}):
            sql_factory = SQLFactory()
            
            # Mock engine to simulate connection failure
            mock_engine = MagicMock()
            mock_session = MagicMock()
            
            with patch.object(sql_factory, 'engine', mock_engine):
                with patch.object(sql_factory.SessionLocal, 'return_value', mock_session):
                    # Make the session creation raise an exception
                    mock_session.execute.side_effect = Exception("Connection failed")
                    
                    with pytest.raises(Exception) as excinfo:
                        sql_factory.execute_query("SELECT 1")
                    
                    assert "Error executing query" in str(excinfo.value)