import os
import sys
import pytest
from unittest.mock import patch, MagicMock

# Add the project root to Python path for imports
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

from fiap_api.factories.sql_factory import SQLFactory

class TestSQLFactoryWithDatabaseSetup:
    """Test cases that specifically verify compatibility with database_setup.py"""
    
    def test_sql_factory_can_be_initialized(self):
        """Test that SQL factory can be initialized without errors"""
        # This tests the basic initialization logic
        with patch.dict(os.environ, {'DATABASE_URL': 'postgresql://postgres:password@localhost/loja_db'}):
            sql_factory = SQLFactory()
            
            assert sql_factory is not None
            assert hasattr(sql_factory, 'db_path')
            assert hasattr(sql_factory, 'engine')
            assert hasattr(sql_factory, 'SessionLocal')
    
    def test_sql_factory_has_correct_methods(self):
        """Test that SQL factory has the expected methods"""
        with patch.dict(os.environ, {'DATABASE_URL': 'postgresql://postgres:password@localhost/loja_db'}):
            sql_factory = SQLFactory()
            
            # Check for required methods
            assert hasattr(sql_factory, 'execute_query')
            assert callable(getattr(sql_factory, 'execute_query'))
            assert hasattr(sql_factory, 'get_engine')
            assert callable(getattr(sql_factory, 'get_engine'))
    
    def test_execute_query_structure(self):
        """Test that execute_query returns expected data structure"""
        with patch.dict(os.environ, {'DATABASE_URL': 'postgresql://postgres:password@localhost/loja_db'}):
            sql_factory = SQLFactory()
            
            # Mock the engine and session behavior
            mock_engine = MagicMock()
            mock_session = MagicMock()
            
            with patch.object(sql_factory, 'engine', mock_engine):
                with patch.object(sql_factory.SessionLocal, 'return_value', mock_session):
                    # Mock result that simulates what we'd get from database queries
                    mock_result = MagicMock()
                    mock_result.keys.return_value = ['id', 'nome']
                    mock_result.__iter__.return_value = [[1, 'Test']]
                    mock_session.execute.return_value = mock_result
                    
                    # Execute a query - this should not raise an exception
                    result = sql_factory.execute_query("SELECT id, nome FROM usuarios LIMIT 1")
                    
                    # Verify the structure of results
                    assert isinstance(result, list)
                    if len(result) > 0:
                        assert 'id' in result[0]
                        assert 'nome' in result[0]
    
    def test_database_url_validation(self):
        """Test that DATABASE_URL environment variable is properly validated"""
        # Test with valid URL
        with patch.dict(os.environ, {'DATABASE_URL': 'postgresql://postgres:password@localhost/loja_db'}):
            sql_factory = SQLFactory()
            assert sql_factory.db_path == 'postgresql://postgres:password@localhost/loja_db'
        
        # Test with empty URL (should raise error)
        with patch.dict(os.environ, {'DATABASE_URL': ''}):
            with pytest.raises(ValueError) as excinfo:
                sql_factory = SQLFactory()
            
            assert "DATABASE_URL environment variable is not set" in str(excinfo.value)

if __name__ == "__main__":
    # Run tests directly if script is executed
    pytest.main([__file__, "-v"])