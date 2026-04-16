import os
import sys
import pytest
from unittest.mock import patch, MagicMock

# Add the project root to Python path for imports
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

from fiap_api.factories.sql_factory import SQLFactory

class TestSQLFactoryDatabaseIntegration:
    """Test cases for SQL Factory with database integration"""
    
    def test_sql_factory_initialization_with_valid_url(self):
        """Test that SQL factory initializes correctly with a valid DATABASE_URL"""
        # Set up the environment variable
        db_url = 'postgresql://postgres:password@localhost/loja_db'
        with patch.dict(os.environ, {'DATABASE_URL': db_url}):
            sql_factory = SQLFactory()
            
            assert sql_factory.db_path == db_url
            assert hasattr(sql_factory, 'engine')
            assert hasattr(sql_factory, 'SessionLocal')
    
    def test_get_engine_returns_engine_instance(self):
        """Test that get_engine method returns a valid engine instance"""
        with patch.dict(os.environ, {'DATABASE_URL': 'postgresql://postgres:password@localhost/loja_db'}):
            sql_factory = SQLFactory()
            
            engine = sql_factory.get_engine()
            
            # Should return the engine object
            assert engine is not None
            assert hasattr(engine, 'execute')
    
    def test_execute_query_with_simple_select(self):
        """Test execute_query with a simple SELECT statement"""
        # This test would require an actual database connection
        # For now we'll mock it to ensure proper method structure
        
        with patch.dict(os.environ, {'DATABASE_URL': 'postgresql://postgres:password@localhost/loja_db'}):
            sql_factory = SQLFactory()
            
            # Mock the engine and session behavior
            mock_engine = MagicMock()
            mock_session = MagicMock()
            mock_result = MagicMock()
            
            # Configure the mocks to simulate a successful query execution
            with patch.object(sql_factory, 'engine', mock_engine):
                with patch.object(sql_factory.SessionLocal, 'return_value', mock_session):
                    # Mock result data - simulating what we'd get from database
                    mock_result.keys.return_value = ['id', 'nome', 'email']
                    mock_result.__iter__.return_value = [[1, 'Test User', 'test@example.com']]
                    mock_session.execute.return_value = mock_result
                    
                    # Execute a test query
                    result = sql_factory.execute_query("SELECT id, nome, email FROM usuarios LIMIT 1")
                    
                    # Verify the results
                    assert isinstance(result, list)
                    assert len(result) == 1
                    assert 'id' in result[0]
                    assert 'nome' in result[0]
                    assert 'email' in result[0]
                    assert result[0]['id'] == 1
                    assert result[0]['nome'] == 'Test User'
                    assert result[0]['email'] == 'test@example.com'

    def test_execute_query_with_exception_handling(self):
        """Test that execute_query properly handles database connection errors"""
        with patch.dict(os.environ, {'DATABASE_URL': 'postgresql://postgres:password@localhost/loja_db'}):
            sql_factory = SQLFactory()
            
            # Mock engine and session to raise an exception
            mock_engine = MagicMock()
            mock_session = MagicMock()
            
            with patch.object(sql_factory, 'engine', mock_engine):
                with patch.object(sql_factory.SessionLocal, 'return_value', mock_session):
                    # Make execute raise an exception
                    mock_session.execute.side_effect = Exception("Database connection failed")
                    
                    # Test that the exception is properly raised
                    with pytest.raises(Exception) as excinfo:
                        sql_factory.execute_query("SELECT * FROM usuarios")
                    
                    assert "Error executing query" in str(excinfo.value)